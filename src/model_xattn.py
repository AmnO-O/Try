#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
         (--pooled-v) mean-pool(H_v) ──┴─ head

The self-attention is what makes this a decoder rather than a bare cross-attention
stack: it lets the comment's own tokens interact (coreference, negation scope)
before they query the video. It is full attention - a causal mask would be wrong
for a classification head that reads the whole comment.

The head reads the pooled comment. Cross-attention is therefore the *only* channel
from the video tower, which is the default; `use_pooled_v=True` (CLI `--pooled-v`)
opts back into late fusion by appending a mean-pooled video vector to the head
input. That second term is a parallel, comment-independent path - a shortcut that
reaches the head without being filtered through the comment's own attention, so it
can also cover for a cross-attention that learns nothing. It is off by default
because a single run at seed 42 with it removed came out
`sep_u2_l3 0.7213` vs `sep_u2_l3_npv 0.7396`, and because dropping it removes one
of the two ways a video-level signal can reach the head; the honest evidence for
that choice is `src/diag_pooled_v.py` (inference-time ablation + attention
entropy + probes), not the 1.83pt, which sits inside the run-to-run spread.

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
        use_pooled_v: bool = False,
        separate_encoders: bool = True,
        keep_frozen_dropout: bool = False,
        pool_init_scale: float = 0.02,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
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
            self.encoder_v = AutoModel.from_pretrained(backbone)
        d = self.encoder.config.hidden_size
        self.d_model = d
        self.use_pooled_v = use_pooled_v

        self.blocks = nn.ModuleList(
            DecoderBlock(d, heads, dropout=dropout, ffn_mult=ffn_mult) for _ in range(layers)
        )
        # Learnable attention-pool query over the comment-side outputs.
        #
        # The 0.02 is a scale choice, not a fitted constant, and it has a visible
        # consequence: the score is `(h_c @ q) / sqrt(d)`, so with `q` at 0.02 and
        # a MiniLM hidden state (RMS ~ 1) the scores start in a narrow band around
        # 0 and the softmax over Lq tokens starts close to uniform, i.e. close to
        # mean pooling. That is not a bug - the query trains - but it does mean the
        # *initial* pooling carries almost no token selection, so "the pool learned
        # something" has to be read relative to that baseline. `pool_init_scale`
        # exposes it for the controlled ablation: change only this, not the FFN.
        # `probe_pool_init.py` measures the initial entropy for several scales.
        self.pool_query = nn.Parameter(torch.randn(d) * pool_init_scale)
        # The head reads the pooled comment. With `use_pooled_v=True` it also
        # gets a mean-pooled video vector appended: a parallel, comment-independent
        # path, i.e. the late-fusion fallback if the cross-attention learns little.
        # Off by default, so cross-attention is the only channel from the video
        # tower (see the module docstring for why, and for what backs that up).
        head_in = 2 * d if use_pooled_v else d
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
        # A fully frozen encoder runs in eval() and without gradients: its tower
        # weights cannot change, so its dropout only adds pointless noise and its
        # activations would be cached anyway.
        #
        # Trainability is decided per encoder rather than from one model-wide
        # flag. A shared flag is wrong as soon as only one tower is re-opened,
        # and - worse - it was what made `keep_frozen_dropout` a silent no-op in
        # the fully frozen case: that branch called enc.eval() unconditionally,
        # so asking to keep frozen dropout had no effect at all.
        enc = self.encoder_v if (tower == 'video' and self.separate_encoders) else self.encoder
        trainable = any(p.requires_grad for p in enc.parameters())
        if not trainable and not self.keep_frozen_dropout:
            enc.eval()
        if not trainable:
            with torch.no_grad():
                return enc(input_ids=ids, attention_mask=mask).last_hidden_state
        return enc(input_ids=ids, attention_mask=mask).last_hidden_state

    def forward(self, comment_ids, comment_mask, video_ids, video_mask,
                return_attn: bool = False):
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
        # `-inf` rather than a large finite number: a finite fill would leave the
        # padded positions a share of the softmax mass, and the pool would then
        # quietly weight the encoder's pad-token outputs.
        scores = scores.masked_fill(q_pad, float('-inf'))
        # A fully padded comment has no finite score at all, and `softmax` of a row
        # of -inf is 0/0. Two things go wrong, and only the first is obvious: the
        # forward value is NaN, and softmax's backward multiplies by its own (NaN)
        # output, so the gradient is NaN too - the loss and every parameter go to
        # NaN and the run is dead. Clamping *after* the softmax (`nan_to_num`,
        # `torch.where`) repairs the forward value but not the gradient. So the
        # row's scores are made finite *before* the softmax, and its weights are
        # zeroed below: it pools to zeros, "no comment tokens to attend to".
        # The data layer rejects such rows; this is the belt to its braces.
        has_token = (~q_pad).any(dim=1, keepdim=True)          # (B, 1)
        scores = torch.where(has_token, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(q_pad, 0.0)
        pooled_c = torch.einsum('bl,bl d->bd', weights, h_c)

        # Mean-pooled video as a parallel, comment-independent path.
        pooled_v = None
        if self.use_pooled_v:
            vmask = video_mask.unsqueeze(-1).to(h_v.dtype)
            pooled_v = (h_v * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
            features = torch.cat([pooled_c, pooled_v], dim=-1)
        else:
            # Cross-attention is then the *only* channel from the video tower.
            features = pooled_c

        logits = self.head(features)
        if not return_attn:
            return logits
        # Diagnostics only: hand back the two pooled vectors so a pooled-v
        # ablation can be done by re-running the head instead of re-encoding.
        # `attn[l]` is (B, heads, Lq, Lkv) over the video keys and `weights` the
        # (B, Lq) comment-pooling distribution.
        return logits, {'attn': attn, 'pool_weights': weights,
                        'pooled_c': pooled_c, 'pooled_v': pooled_v}


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
