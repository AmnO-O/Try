#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the model actually costs, per backbone and per encoder setting.

The `model_xattn` docstring used to state one pair of numbers - 240.6M separate,
122.98M shared - as if they described the architecture. They describe MiniLM, and
only MiniLM. mmBERT-base is a different order: a 256k vocab and 768 hidden make
the embedding table alone ~197M, so one ModernBERT tower is ~307M and the
separate-encoder default is ~614M *before* the cross-attention stack and head.
That is why the reviewer's memory point stands: separate encoders are a
deliberate experimental variable, not a free lunch.

Counts are built from the real classes on the meta device, so they are the
model's own parameter count without paying for the weights. The trainable column
uses `unfreeze_last_n(encoder, n)` on the same meta module.

Run:  python src/report_cost.py
"""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from transformers import AutoConfig, AutoModel  # noqa: E402

import model_xattn  # noqa: E402
from model import count_encoder_blocks, unfreeze_last_n  # noqa: E402

BACKBONES = [
    ('MiniLM-L12', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    ('mmBERT-base', 'jhu-clsp/mmBERT-base'),
]
HEAD = 4
LAYERS = 3
FFN = 4
# AdamW keeps a moment per trainable parameter on top of the parameter and its
# gradient: fp32 param + fp32 grad + 2 x fp32 moment = 4 x the parameter bytes.
TRAIN_MULTIPLIER = 4


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def stack_counts(d_model: int):
    """(cross-attn blocks, pool query, head) counts, from the real classes."""
    with torch.device('meta'):
        blocks = nn.ModuleList(model_xattn.XAttnBlock(d_model, HEAD, dropout=0.1, ffn_mult=FFN)
                               for _ in range(LAYERS))
        pool = nn.Parameter(torch.empty(d_model))
        head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1), nn.Linear(d_model, 2))
    return count(blocks), pool.numel(), count(head)


print(f'{"backbone":<12} {"blocks":>6} {"d":>5} | {"1 encoder":>10} {"stack":>8} | '
      f'{"shared":>10} {"separate":>10} | {"sep fp32":>9} {"sep train u2":>13}')
print('-' * 104)

for label, backbone in BACKBONES:
    config = AutoConfig.from_pretrained(backbone)
    d_model = config.hidden_size
    with torch.device('meta'):
        encoder = AutoModel.from_config(config)
    encoder_params = count(encoder)
    n_found = count_encoder_blocks(encoder)
    block_n, pool_n, head_n = stack_counts(d_model)
    stack = block_n + pool_n + head_n
    shared = encoder_params + stack
    separate = 2 * encoder_params + stack

    # Trainable encoder slice at the default --unfreeze-layers 2, from the helper
    # itself so this column cannot drift from what training actually opens.
    unfreeze_last_n(encoder, 2)
    unfrozen_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    unfreeze_last_n(encoder, 0)

    sep_touched = 2 * unfrozen_encoder + stack           # params with a gradient
    print(f'{label:<12} {n_found:>6} {d_model:>5} | {encoder_params:>10} {stack:>8} | '
          f'{shared:>10} {separate:>10} | {separate * 4 / 1024 ** 3:>7.2f}G '
          f'{sep_touched * TRAIN_MULTIPLIER / 1024 ** 3:>10.2f} GiB')

print()
print('Notes:')
print(f'  - "sep train u2" is the AdamW footprint (param + grad + 2 moments = '
      f'{TRAIN_MULTIPLIER}x) for the *trainable* part of the separate-encoder model')
print('    at the default --unfreeze-layers 2: the frozen blocks still need their')
print('    forward activations, so this is a lower bound on the real footprint.')
print('  - shared encoder: half the encoder bytes but the same stack; caching is')
print('    not implemented either way (see model_xattn.py).')
