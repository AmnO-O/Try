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

One encoder can be shared by both towers (default) so the video tower's K/V can
be cached per video: with many comments per video this is the efficiency win
over concatenating everything into one sequence. The design is dimension
agnostic (`d_model` comes from the backbone), so swapping the light MiniLM for
`jhu-clsp/mmbert-base` is just a different `--backbone`.
"""

import math

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_BACKBONE = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


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
        a, _ = self.attn(self.norm_q(h_q), self.norm_kv(h_kv), self.norm_kv(h_kv),
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
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
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

    def encode(self, ids, mask):
        # A fully frozen encoder runs in eval() and without gradients: its tower
        # weights cannot change, so its dropout only adds pointless noise and its
        # activations would be cached anyway. `encoder_trainable` is set by
        # build_model() (default True keeps a directly-built model trainable).
        if getattr(self, 'encoder_trainable', True):
            return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    def forward(self, comment_ids, comment_mask, video_ids, video_mask):
        h_c = self.encode(comment_ids, comment_mask)
        h_v = self.encode(video_ids, video_mask)

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
    """Builds the classifier and freezes the shared encoder except the last
    `unfreeze_layers` blocks (0 = fully frozen; the head always trains)."""
    from model import unfreeze_last_n

    net = XAttnClassifier(backbone, **kwargs)
    unfreeze_last_n(net.encoder, unfreeze_layers)
    net.encoder_trainable = any(p.requires_grad for p in net.encoder.parameters())
    return net


def trainable_groups(net: XAttnClassifier):
    """(backbone, head) parameter groups for a two-LR optimizer.

    Frozen encoder params are excluded so the backbone group is empty (and can
    be skipped) when `unfreeze_layers == 0`.
    """
    backbone = [p for p in net.encoder.parameters() if p.requires_grad]
    head = [p for n, p in net.named_parameters()
            if p.requires_grad and not n.startswith('encoder.')]
    return backbone, head
