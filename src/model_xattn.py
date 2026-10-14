#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-tower + cross-attention classifier for Task A (stereotype yes/no).

The comment is the *query* and the video context (title + description) is the
*k/v memory*, i.e. exactly the cross-attention of a Transformer decoder, but
without any generation: pool the comment-side outputs and read a 2-way logit.

    comment  -> encoder -> H_c ─┐
                                ├─ cross-attn (Q=H_c, K=V=H_v) -> attn-pool ─┐
    video    -> encoder -> H_v ─┘                                mean-pool ──┴─ head

The head reads the pooled comment plus a mean-pooled video vector. That second
term is a *parallel, comment-independent* path (a late-fusion shortcut that also
keeps the model honest if the cross-attention learns nothing); `use_pooled_v`
drops it so the cross-attention becomes the only channel from the video tower.

The video tower has its own encoder instance by default (`separate_encoders`), so
the two text styles - terse/formal title+description vs. informal, often abusive
comments - can adapt independently in the blocks that training re-opens, and the
two towers cannot fight over shared weights. Both copies start from the same
pretrained checkpoint and every frozen block stays bit-identical between them, so
the towers can only diverge in the `unfreeze_last_n` blocks. Set
`separate_encoders=False` (trainer: `--shared-encoder`) to go back to one shared
encoder, which halves the parameter count (240.6M -> 122.98M) and makes the video
K/V cacheable per video.
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

    def forward(self, h_q, h_kv, kv_pad_mask):
        # Normalise the memory once: the same tensor is both K and V.
        kv = self.norm_kv(h_kv)
        a, _ = self.attn(self.norm_q(h_q), kv, kv,
                         key_padding_mask=kv_pad_mask, need_weights=False)
        h_q = h_q + a
        return h_q + self.ffn(self.norm_ffn(h_q))


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
        use_pooled_v: bool = True,
        separate_encoders: bool = True,
        keep_frozen_dropout: bool = False,
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
            XAttnBlock(d, heads, dropout=dropout, ffn_mult=ffn_mult) for _ in range(layers)
        )
        # Learnable attention-pool query over the comment-side outputs.
        self.pool_query = nn.Parameter(torch.randn(d) * 0.02)
        # The head reads the pooled comment and, unless `use_pooled_v=False`, a
        # mean-pooled video vector. `pooled_v` is a parallel, comment-independent
        # path: the only route by which a video-level signal reaches the head
        # without being filtered through the comment's own attention (and so the
        # fallback to plain late fusion if the cross-attention learns nothing).
        head_in = 2 * d if use_pooled_v else d
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def encoders(self):
        """The encoder module(s) actually in use (one shared, or one per tower)."""
        return [self.encoder, self.encoder_v] if self.separate_encoders else [self.encoder]

    def encode(self, ids, mask, tower: str = 'comment'):
        # A fully frozen encoder runs in eval() and without gradients: its tower
        # weights cannot change, so its dropout only adds pointless noise and its
        # activations would be cached anyway. `encoder_trainable` is set by
        # build_model() (default True keeps a directly-built model trainable).
        enc = self.encoder_v if (tower == 'video' and self.separate_encoders) else self.encoder
        if getattr(self, 'encoder_trainable', True):
            return enc(input_ids=ids, attention_mask=mask).last_hidden_state
        enc.eval()
        with torch.no_grad():
            return enc(input_ids=ids, attention_mask=mask).last_hidden_state

    def forward(self, comment_ids, comment_mask, video_ids, video_mask):
        h_c = self.encode(comment_ids, comment_mask, tower='comment')
        h_v = self.encode(video_ids, video_mask, tower='video')

        kv_pad = video_mask == 0          # True = ignore this key
        q_pad = comment_mask == 0
        for block in self.blocks:
            h_c = block(h_c, h_v, kv_pad)

        # Attention pooling over the (unmasked) comment tokens.
        scores = (h_c @ self.pool_query) / math.sqrt(self.d_model)
        scores = scores.masked_fill(q_pad, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled_c = torch.einsum('bl,bl d->bd', weights, h_c)

        # Mean-pooled video as a parallel, comment-independent path.
        if self.use_pooled_v:
            vmask = video_mask.unsqueeze(-1).to(h_v.dtype)
            pooled_v = (h_v * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
            features = torch.cat([pooled_c, pooled_v], dim=-1)
        else:
            # Cross-attention is then the *only* channel from the video tower.
            features = pooled_c

        return self.head(features)


def build_model(backbone: str, unfreeze_layers: int = 0, **kwargs) -> XAttnClassifier:
    """Builds the classifier and freezes the encoder tower(s) except the last
    `unfreeze_layers` blocks (0 = fully frozen; the head always trains)."""
    from model import eval_frozen_modules, unfreeze_last_n

    net = XAttnClassifier(backbone, **kwargs)
    for enc in net.encoders():
        unfreeze_last_n(enc, unfreeze_layers)
    net.encoder_trainable = any(p.requires_grad for enc in net.encoders()
                                for p in enc.parameters())
    if not net.keep_frozen_dropout:
        for enc in net.encoders():
            eval_frozen_modules(enc)
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
