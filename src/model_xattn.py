#!/usr/bin/env python3
"""Two-tower encoder-decoder classifier for Task A (stereotype yes/no).

The video context (title + description) is the *encoder memory* and the comment is
the *decoder input*: each decoder block first refines the comment against itself
(self-attention, full - not causal), then attends into the video memory
(cross-attention, Q=comment, K=V=video). No generation: pool the comment-side
outputs and read a 2-way logit.

    comment  -> encoder -> H_c ─┐
    video    -> encoder -> H_v ─┤
    Z_0 = H_c  --[ N x DecoderBlock ]-->  Z_N
        DecoderBlock:  self-attn(Z, Z, Z)  ->  cross-attn(Q=Z, K=V=H_v)  ->  FFN
    Z_N  -> attn-pool ─┐
         mean-pool(H_v) ─┴─ fusion (none|concat|gated) -> head

The self-attention is what makes this a decoder rather than a bare cross-attention
stack: it lets the comment's own tokens interact (coreference, negation scope)
before they query the video. It is full attention - a causal mask would be wrong
for a classification head that reads the whole comment.

How the video reaches the head is chosen by `fusion` (default `gated`): `none` reads
the pooled comment only, so cross-attention is the *only* channel from the video
tower; `concat` appends a mean-pooled video vector to the head input; `gated` adds
`sigmoid(W.[pooled_c; mean-pool(H_v)]) * video_proj(mean-pool(H_v))`, a learned gate
(bias initialised to -1.5, i.e. mostly closed) on that same parallel term. The
`concat`/`gated` term is a comment-independent path - a shortcut that reaches the head
without being filtered through the comment's own attention, so it can cover for a
cross-attention that learns nothing. The choice is measured in
`src/diag_pooled_v.py` (inference-time ablation for `concat` + attention entropy +
probes + an input-level video perturbation), not by the historical seed-42 pair
(`0.7213` vs `0.7396`, 1.83pt), which sits inside the run-to-run spread. The old
boolean `use_pooled_v=True/False` is kept as a compatibility keyword mapping to
`concat`/`none`.

The decoder blocks are **warm-started** from the top encoder layers by default
(`warm_start_decoder_from_backbone`: self-attn, cross-attn, FFN and LayerNorm), so
they start from pretrained weights instead of random init; `warm_start=False` (CLI
`--no-warm-start`) restores random init and is the control for that choice.

`forward` rejects an all-padded *video* row up front (see `_validate_video_mask`):
with every key masked, cross-attention is undefined - the training path returns
zeros silently, and the `return_attn=True` diagnostic path returns NaN weights and
a NaN output, on top of the NaN backward pass. The comment side keeps a defensive
fallback instead (the data layer rejects those rows first); its encoder behaviour on
an all-zero mask is verified per backbone in `src/probe_allpad.py`.

The video tower has its own encoder instance by default (`separate_encoders`), so
the two text styles - terse/formal title+description vs. informal, often abusive
comments - can adapt independently in the blocks that training re-opens, and the
two towers cannot fight over shared weights. Both copies start from the same
pretrained checkpoint and every frozen block stays bit-identical between them, so
the towers can only diverge in the `unfreeze_last_n` blocks. Set
`separate_encoders=False` (trainer: `--shared-encoder`) to go back to one shared
encoder, which roughly halves the parameter count.

**Nothing is cached.** `forward()` always runs both towers, so a video that
appears in several rows is re-encoded on every row, and a shared encoder does not
change that by itself - a cache also needs the *video* weights to be stationary.
That holds while the video tower is fully frozen, and at inference once training
is done; it does not hold across optimizer steps of a trainable video tower,
where a persistent representation goes stale. Caching is therefore left out
until the model's own behaviour is established (`src/report_cost.py` measures the
parameter cost, which does *not* generalise across backbones: ~242M vs ~125M for
MiniLM, but ~642M vs ~335M for mmBERT; the two towers alone are ~614M vs ~307M
for mmBERT, before the decoder stack).
"""

import math

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_BACKBONE = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Parameter-name prefixes that belong to the encoder towers. With
# `separate_encoders` the video tower is a second instance named `encoder_v`, so
# anything that partitions "backbone vs head" must know about both.
ENCODER_PREFIXES = ('encoder.', 'encoder_v.')

# The cross-attention stack. These are *freshly initialised* Transformer layers
# trained on top of a pretrained encoder, so they are a third thing: neither
# backbone (pretrained, wants a tiny step) nor head (a single linear layer,
# tolerates a large one). `trainable_groups` gives them their own LR for exactly
# that reason.
BLOCK_PREFIX = 'blocks.'


class XAttnBlock(nn.Module):
    """Pre-LN cross-attention + FFN, one layer.

    Query comes from the comment tower; key/value come from the video tower.
    `key_padding_mask` marks padded video tokens (True = ignore), which keeps
    padded positions from entering the softmax.

    Kept for backward compatibility and for the comparison in `probe_decoder.py`;
    the shipped fusion stack is `DecoderBlock`.
    """

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        if d_model % heads:
            raise ValueError(f'd_model {d_model} must be divisible by heads {heads}')
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, h_q, h_kv, kv_pad_mask, return_attn: bool = False):
        """`return_attn=True` also hands back the (B, heads, Lq, Lkv) attention
        probabilities. That forces the slow attention path, so only ask for it
        when the weights are actually wanted (diagnostics, not training)."""
        kv = self.norm_kv(h_kv)
        if return_attn:
            a, weights = self.attn(self.norm_q(h_q), kv, kv, key_padding_mask=kv_pad_mask,
                                   need_weights=True, average_attn_weights=False)
        else:
            a, weights = self.attn(self.norm_q(h_q), kv, kv,
                                   key_padding_mask=kv_pad_mask, need_weights=False)
        h_q = h_q + a
        out = h_q + self.ffn(self.norm_ffn(h_q))
        return (out, weights) if return_attn else out


class DecoderBlock(nn.Module):
    """Transformer decoder block: self-attn on comment + cross-attn to video.

    - Self-attn: full attention (no causal). Q=K=V = h (comment-side), mask_c
    - Cross-attn: Q = h1, K=V = H_v (video memory), mask_v
    - Pre-LN
    """

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        if d_model % heads:
            raise ValueError(f'd_model {d_model} must be divisible by heads {heads}')
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h, H_v, mask_c=None, mask_v=None, return_attn: bool = False):
        # 1) self-attn
        h_self_in = self.norm_self(h)
        # A row whose *every* comment token is padding has no valid key at all:
        # softmax over an all -inf row is NaN, in the forward value and in the
        # backward pass. The data layer rejects such rows, but the model keeps its
        # own belt: keep one key alive for those rows so the softmax is finite,
        # then zero the whole row back out. `torch.where` with a finite branch is
        # safe for gradients (unlike clamping after a NaN softmax).
        if mask_c is not None:
            has_token = (~mask_c).any(dim=1, keepdim=True)      # (B, 1)
            safe_mask = mask_c
            empty_rows = ~has_token.squeeze(1)                  # (B,)
            if bool(empty_rows.any()):
                safe_mask = mask_c.clone()
                safe_mask[empty_rows, 0] = False
            a_s, _ = self.self_attn(h_self_in, h_self_in, h_self_in,
                                    key_padding_mask=safe_mask,
                                    need_weights=False)
            a_s = torch.where(has_token.unsqueeze(-1), a_s, torch.zeros_like(a_s))
        else:
            a_s, _ = self.self_attn(h_self_in, h_self_in, h_self_in,
                                    need_weights=False)
        h1 = h + a_s

        # 2) cross-attn
        kv = self.norm_kv(H_v)
        qn = self.norm_q(h1)
        if return_attn:
            a_c, weights = self.cross_attn(qn, kv, kv,
                                           key_padding_mask=mask_v,
                                           need_weights=True,
                                           average_attn_weights=False)
        else:
            a_c, weights = self.cross_attn(qn, kv, kv,
                                           key_padding_mask=mask_v,
                                           need_weights=False)
        h2 = h1 + a_c

        # 3) ffn
        h3 = h2 + self.ffn(self.norm_ffn(h2))
        return (h3, weights) if return_attn else h3


def warm_start_decoder_from_backbone(blocks: nn.ModuleList, encoder: nn.Module) -> int:
    """Clones self-attention, cross-attention, and FFN weights from the top layers
    of the pretrained encoder into the freshly initialized DecoderBlocks.

    Returns the count of blocks successfully warm-started.
    """
    import re
    layers = {}
    for name, mod in encoder.named_modules():
        m = re.search(r'(?:^|\.)(?:layers|layer|blocks)\.(\d+)$', name)
        if m:
            layers[int(m.group(1))] = mod

    if not layers:
        return 0

    sorted_indices = sorted(layers.keys())
    top_layers = [layers[i] for i in sorted_indices[-len(blocks):]]
    count = 0

    with torch.no_grad():
        for blk, enc_layer in zip(blocks, top_layers):
            # Check Bert / Roberta style
            if hasattr(enc_layer, 'attention') and hasattr(enc_layer.attention, 'self'):
                self_attn = enc_layer.attention.self
                out_dense = enc_layer.attention.output.dense
                q_w, k_w, v_w = self_attn.query.weight, self_attn.key.weight, self_attn.value.weight
                in_w = torch.cat([q_w, k_w, v_w], dim=0)
                if blk.self_attn.in_proj_weight.shape == in_w.shape:
                    blk.self_attn.in_proj_weight.copy_(in_w)
                    if self_attn.query.bias is not None and blk.self_attn.in_proj_bias is not None:
                        in_b = torch.cat([self_attn.query.bias, self_attn.key.bias, self_attn.value.bias], dim=0)
                        blk.self_attn.in_proj_bias.copy_(in_b)
                    blk.self_attn.out_proj.weight.copy_(out_dense.weight)
                    if out_dense.bias is not None and blk.self_attn.out_proj.bias is not None:
                        blk.self_attn.out_proj.bias.copy_(out_dense.bias)

                    # Cross-attention initialization (from the same semantic space)
                    blk.cross_attn.in_proj_weight.copy_(in_w)
                    if blk.cross_attn.in_proj_bias is not None and self_attn.query.bias is not None:
                        blk.cross_attn.in_proj_bias.copy_(in_b)
                    blk.cross_attn.out_proj.weight.copy_(out_dense.weight)
                    if out_dense.bias is not None and blk.cross_attn.out_proj.bias is not None:
                        blk.cross_attn.out_proj.bias.copy_(out_dense.bias)

                # FFN
                if hasattr(enc_layer, 'intermediate') and hasattr(enc_layer, 'output'):
                    if blk.ffn[0].weight.shape == enc_layer.intermediate.dense.weight.shape:
                        blk.ffn[0].weight.copy_(enc_layer.intermediate.dense.weight)
                        if enc_layer.intermediate.dense.bias is not None:
                            blk.ffn[0].bias.copy_(enc_layer.intermediate.dense.bias)
                        blk.ffn[3].weight.copy_(enc_layer.output.dense.weight)
                        if enc_layer.output.dense.bias is not None:
                            blk.ffn[3].bias.copy_(enc_layer.output.dense.bias)

                # LayerNorms
                if hasattr(enc_layer.attention.output, 'LayerNorm') and hasattr(enc_layer.output, 'LayerNorm'):
                    ln_attn = enc_layer.attention.output.LayerNorm
                    ln_out = enc_layer.output.LayerNorm
                    blk.norm_self.weight.copy_(ln_attn.weight)
                    blk.norm_self.bias.copy_(ln_attn.bias)
                    blk.norm_q.weight.copy_(ln_attn.weight)
                    blk.norm_q.bias.copy_(ln_attn.bias)
                    blk.norm_kv.weight.copy_(ln_attn.weight)
                    blk.norm_kv.bias.copy_(ln_attn.bias)
                    blk.norm_ffn.weight.copy_(ln_out.weight)
                    blk.norm_ffn.bias.copy_(ln_out.bias)
                count += 1

            # Check ModernBERT style
            elif hasattr(enc_layer, 'attn') and hasattr(enc_layer.attn, 'Wqkv'):
                in_w = enc_layer.attn.Wqkv.weight
                if blk.self_attn.in_proj_weight.shape == in_w.shape:
                    blk.self_attn.in_proj_weight.copy_(in_w)
                    if enc_layer.attn.Wqkv.bias is not None and blk.self_attn.in_proj_bias is not None:
                        blk.self_attn.in_proj_bias.copy_(enc_layer.attn.Wqkv.bias)
                    blk.self_attn.out_proj.weight.copy_(enc_layer.attn.Wo.weight)
                    if enc_layer.attn.Wo.bias is not None and blk.self_attn.out_proj.bias is not None:
                        blk.self_attn.out_proj.bias.copy_(enc_layer.attn.Wo.bias)

                    blk.cross_attn.in_proj_weight.copy_(in_w)
                    if enc_layer.attn.Wqkv.bias is not None and blk.cross_attn.in_proj_bias is not None:
                        blk.cross_attn.in_proj_bias.copy_(enc_layer.attn.Wqkv.bias)
                    blk.cross_attn.out_proj.weight.copy_(enc_layer.attn.Wo.weight)
                    if enc_layer.attn.Wo.bias is not None and blk.cross_attn.out_proj.bias is not None:
                        blk.cross_attn.out_proj.bias.copy_(enc_layer.attn.Wo.bias)

                if hasattr(enc_layer, 'mlp') and hasattr(enc_layer.mlp, 'Wi'):
                    if blk.ffn[0].weight.shape == enc_layer.mlp.Wi.weight.shape:
                        blk.ffn[0].weight.copy_(enc_layer.mlp.Wi.weight)
                        blk.ffn[3].weight.copy_(enc_layer.mlp.Wo.weight)
                count += 1

    return count


class XAttnClassifier(nn.Module):
    """Comment-query / video-memory cross-attention classifier."""

    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        use_pooled_v: bool | None = None,
        fusion: str = 'gated',
        warm_start: bool = True,
        separate_encoders: bool = True,
        keep_frozen_dropout: bool = False,
        pool_init_scale: float = 0.02,
    ):
        super().__init__()
        # Force float32: the decoder blocks and head are fp32, and AMP keeps weights
        # fp32 (only the forward runs under autocast), so a checkpoint shipped in
        # bfloat16 - e.g. vllm-sr/mmbert-embed-32k-2d-matryoshka - would otherwise make
        # the encoder output bf16 and crash the first fp32 matmul (BFloat16 != float).
        self.encoder = AutoModel.from_pretrained(backbone).float()
        # With separate encoders the video tower gets its own copy of the same
        # pretrained checkpoint. Both start identical; the two towers can only
        # diverge in the blocks that `unfreeze_last_n` re-opens, because every
        # other block stays frozen (and therefore bit-identical) in both copies.
        self.separate_encoders = separate_encoders
        # False (default): frozen encoder subtrees are switched to eval(), so the
        # frozen half of the tower is deterministic in training too. True keeps
        # their Dropout firing, i.e. the old stochastic-regularizer behaviour.
        self.keep_frozen_dropout = keep_frozen_dropout
        if separate_encoders:
            self.encoder_v = AutoModel.from_pretrained(backbone).float()
        d = self.encoder.config.hidden_size
        self.d_model = d

        # Fusion configuration: 'none', 'concat', or 'gated' (adaptive)
        if use_pooled_v is not None:
            self.fusion = 'concat' if use_pooled_v else 'none'
        else:
            self.fusion = fusion
        self.use_pooled_v = (self.fusion == 'concat')

        self.blocks = nn.ModuleList(
            DecoderBlock(d, heads, dropout=dropout, ffn_mult=ffn_mult) for _ in range(layers)
        )
        if warm_start:
            warm_start_decoder_from_backbone(self.blocks, self.encoder)

        # Learnable attention-pool query over the comment-side outputs.
        self.pool_query = nn.Parameter(torch.randn(d) * pool_init_scale)

        # Adaptive Gated Fusion components
        if self.fusion == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.gate[0].bias, -1.5)
            self.video_proj = nn.Linear(d, d)

        head_in = 2 * d if self.fusion == 'concat' else d
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def train(self, mode: bool = True):
        """`train(True)` also re-applies the frozen-module invariant.

        `nn.Module.train` walks every child, so it re-enables Dropout (and
        attention-prob Dropout) inside the *frozen* encoder blocks that
        `eval_frozen_modules` had switched off. A frozen tower then injects a
        fresh random mask into every step, which is neither what "frozen" means
        nor what the "frozen blocks stay bit-identical between the two towers"
        argument for the separate-encoder design needs.

        Doing it here means the invariant holds for every caller. It used to live
        in the training loop only, so any other entry point - or a loop that
        forgot the second line - silently got a stochastic frozen tower. Callers
        that want the old stochastic-regulator behaviour set
        `keep_frozen_dropout=True`, which turns this off.
        """
        super().train(mode)
        if mode and not self.keep_frozen_dropout:
            from model import eval_frozen_modules
            for enc in self.encoders():
                eval_frozen_modules(enc)
        return self

    def encoders(self):
        """The encoder module(s) actually in use (one shared, or one per tower)."""
        return [self.encoder, self.encoder_v] if self.separate_encoders else [self.encoder]

    def encode(self, ids, mask, tower: str = 'comment'):
        enc = self.encoder_v if (tower == 'video' and self.separate_encoders) else self.encoder
        trainable = any(p.requires_grad for p in enc.parameters())
        if not trainable and not self.keep_frozen_dropout:
            enc.eval()
        with torch.set_grad_enabled(trainable):
            return enc(input_ids=ids, attention_mask=mask).last_hidden_state

    def fuse(self, pooled_c, pooled_v=None):
        """Combines comment and video representations according to self.fusion."""
        if self.fusion == 'none' or pooled_v is None:
            return pooled_c
        if self.fusion == 'concat':
            return torch.cat([pooled_c, pooled_v], dim=-1)
        if self.fusion == 'gated':
            g = self.gate(torch.cat([pooled_c, pooled_v], dim=-1))
            return pooled_c + g * self.video_proj(pooled_v)
        raise ValueError(f'Unknown fusion mode {self.fusion}')

    @staticmethod
    def _validate_video_mask(video_mask):
        """Reject an all-padded video row at the model boundary.

        The training data layer already refuses such rows (see
        `train_xattn.assert_nonempty_masks`), but the model can be called from
        anywhere - inference, `diag_pooled_v.py`, a probe - and an all-padded
        video row is genuinely undefined:

        - `need_weights=False` (the training path) answers with zeros, silently;
        - `need_weights=True` (the `return_attn=True` diagnostic path) puts an
          all-`-inf` row through the softmax, which is NaN - NaN weights and a NaN
          comment output - and the backward pass is NaN too.

        A valid token can be a special token, so this needs no non-empty title or
        description: it only requires the combined, tokenised video input to keep
        at least one unmasked token. Rejecting here means the NaN path is
        unreachable by construction, on every backend, instead of only being
        papered over on the optimized path.
        """
        if video_mask.ndim != 2 or video_mask.shape[1] == 0:
            raise ValueError(
                f'video_mask must have shape [batch, seq_len] with seq_len > 0, '
                f'got {tuple(video_mask.shape)}')
        per_row = video_mask.bool().any(dim=1)
        if not bool(per_row.all()):
            bad = (~per_row).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f'{len(bad)} video row(s) have no valid token (first: row {bad[0]}). '
                f'Every video context must contain at least one unmasked token: with '
                f'every key masked, cross-attention is undefined - the training path '
                f'answers with zeros and the need_weights=True diagnostic path with NaN.')

    def forward(self, comment_ids, comment_mask, video_ids, video_mask,
                return_attn: bool = False):
        self._validate_video_mask(video_mask)
        h_c = self.encode(comment_ids, comment_mask, tower='comment')
        h_v = self.encode(video_ids, video_mask, tower='video')

        kv_pad = video_mask == 0          # True = ignore this key
        q_pad = comment_mask == 0
        attn = []
        for block in self.blocks:
            if return_attn:
                h_c, w = block(h_c, h_v, q_pad, kv_pad, return_attn=True)
                attn.append(w)
            else:
                h_c = block(h_c, h_v, q_pad, kv_pad)

        # Attention pooling over the comment tokens.
        scores = (h_c @ self.pool_query) / math.sqrt(self.d_model)
        scores = scores.masked_fill(q_pad, float('-inf'))
        has_token = (~q_pad).any(dim=1, keepdim=True)          # (B, 1)
        scores = torch.where(has_token, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(q_pad, 0.0)
        pooled_c = torch.einsum('bl,bl d->bd', weights, h_c)

        # Mean-pooled video for late or gated fusion
        pooled_v = None
        if self.fusion != 'none':
            vmask = video_mask.unsqueeze(-1).to(h_v.dtype)
            pooled_v = (h_v * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)

        features = self.fuse(pooled_c, pooled_v)
        logits = self.head(features)
        if not return_attn:
            return logits
        return logits, {'attn': attn, 'pool_weights': weights,
                        'pooled_c': pooled_c, 'pooled_v': pooled_v,
                        'features': features}


def build_model(backbone: str, unfreeze_layers: int = 0, **kwargs) -> XAttnClassifier:
    """Builds the classifier and freezes the encoder tower(s) except the last
    `unfreeze_layers` blocks (0 = fully frozen; the head always trains).

    Returns the model in one consistent state: the trainable parts in train mode,
    the frozen parts in eval(). `from_pretrained` leaves the encoder in eval()
    while the freshly created blocks/head are in train(), so without the
    `net.train()` below the *trainable* encoder blocks would start with dropout
    off. `train()` is what applies the frozen-module invariant (see
    `XAttnClassifier.train`), and `keep_frozen_dropout=True` disables it, leaving
    the previous stochastic-frozen-blocks behaviour.

    Inference callers should still call `net.eval()` explicitly.
    """
    from model import unfreeze_last_n

    net = XAttnClassifier(backbone, **kwargs)
    for enc in net.encoders():
        unfreeze_last_n(enc, unfreeze_layers)
    net.train()
    return net


def trainable_groups(net: XAttnClassifier):
    """(backbone, blocks, head) parameter groups for a three-LR optimizer.

    Three groups, because the three parts want very different step sizes:

    - **backbone** - a pretrained encoder being fine-tuned, so a small LR
      (2e-5) is enough and a large one destroys it;
    - **blocks** - the cross-attention stack: *randomly initialised* Transformer
      layers that must learn from scratch, so they need a real training LR, but
      nowhere near the head's: with AdamW the per-step move is ~LR regardless of
      gradient scale, so a freshly initialised block (weights ~N(0, 0.02)) fed
      lr 1e-3 moves ~5% of its own weight scale every step;
    - **head** - `LayerNorm + Linear(d, 2)` plus the pooling query, a handful of
      parameters that happily take 1e-3.

    The blocks used to be lumped in with the head (anything not under
    `ENCODER_PREFIXES`), i.e. the whole 5.33M-parameter attention stack was
    trained at `--head-lr`. Splitting them out is what `--xattn-lr` controls.

    Frozen encoder params are excluded, so the backbone group is empty (and can
    be skipped) when `unfreeze_layers == 0`. With `separate_encoders` both tower
    encoders belong to the backbone group.
    """
    backbone = [p for enc in net.encoders() for p in enc.parameters() if p.requires_grad]
    blocks = [p for n, p in net.named_parameters()
              if p.requires_grad and n.startswith(BLOCK_PREFIX)]
    head = [p for n, p in net.named_parameters()
            if p.requires_grad and not n.startswith(ENCODER_PREFIXES + (BLOCK_PREFIX,))]
    return backbone, blocks, head
