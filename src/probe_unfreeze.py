#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does unfreeze_last_n() do what it says, on both backbones?

`unfreeze_last_n` only clears `requires_grad`; which parameters those are is
decided from parameter *names* (`...layer.11...` for BERT-style, `...layers.21...`
for ModernBERT) so that it keeps working across encoder layouts. The risk that
buys: a name it does not recognise is frozen silently, and the experiment quietly
becomes a different experiment. This probe does not take the names on trust - the
expected set of trainable parameters is derived from the module tree, and only
then compared with what the name-based helper produced.

Checks, for MiniLM (BERT-style, 12 blocks) and mmBERT (ModernBERT, 22 blocks):
  a. every block index is detected, and no phantom index appears;
  b. for n in {0, 1, 2, all}: the trainable set is exactly the last n blocks, plus
     the encoder's output normalisation where the architecture keeps it outside
     the block stack (ModernBERT's `final_norm`);
  c. the embeddings stay frozen at every n;
  d. the top of the tower ends up equivalent across the two architectures;
  e. both towers of a separate-encoder model get the same pattern;
  f. the optimizer groups cover every trainable parameter exactly once;
  g. n < 0 and n > block count fail loudly instead of silently.

Run:  python src/probe_unfreeze.py
Exit code 0 = every check holds.
"""
import gc
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from transformers import AutoModel  # noqa: E402

from model import count_encoder_blocks, layer_index_in_name, unfreeze_last_n  # noqa: E402

MINILM = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MMBERT = 'jhu-clsp/mmBERT-base'
SMALL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'   # cheap two-tower test
BACKBONES = [('MiniLM-L12   (BERT-style)', MINILM), ('mmBERT-base  (ModernBERT)', MMBERT)]

FAILURES = []


def check(ok, label, extra=''):
    print(f'    [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


def find_blocks(encoder):
    """The transformer block list, found structurally rather than by name."""
    best_name, best = None, None
    for name, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            if best is None or len(module) > len(best):
                best_name, best = name, module
    return best_name, best


def trainable(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def output_norm_params(model):
    """Parameters of a *direct child* of the encoder whose name says 'norm'.

    That is ModernBERT's `final_norm`. BERT-style encoders have no such module -
    the equivalent LayerNorm lives inside the last block - so this is empty.
    """
    direct = {name for name, _ in model.named_modules() if '.' not in name}
    return {name for name, _ in model.named_parameters()
            if name.split('.')[0] in direct and 'norm' in name.split('.')[0].lower()}


def expected_last_n(encoder, blocks, n):
    """What the last n blocks *plus the output norm* should make trainable."""
    if n <= 0:
        return set()
    name_of = {id(p): name for name, p in encoder.named_parameters()}
    want = {name_of[id(p)] for block in blocks[len(blocks) - n:] for p in block.parameters()}
    return want | output_norm_params(encoder)


outside_at_one = {}

for label, backbone in BACKBONES:
    print(f'\n=== {label} ===')
    base = AutoModel.from_pretrained(backbone)
    base.eval()
    block_name, blocks = find_blocks(base)
    n_blocks = len(blocks)
    print(f'  structure: blocks at {block_name!r}, {n_blocks} of them')
    print(f'  normalisation outside the block stack: '
          f'{sorted(output_norm_params(base)) or "none (BERT folds it into the last block)"}')

    idx = {i for i in (layer_index_in_name(n) for n, _ in base.named_parameters()) if i is not None}
    check(idx == set(range(n_blocks)),
          'every block index is detected, and no phantom index appears',
          f'{len(idx)} indices, {min(idx)}..{max(idx)}' if idx else 'none detected')
    check(count_encoder_blocks(base) == n_blocks,
          'count_encoder_blocks agrees with the module tree')

    for n in (0, 1, 2, n_blocks):
        enc = AutoModel.from_pretrained(backbone)
        enc.eval()
        _, enc_blocks = find_blocks(enc)
        unfreeze_last_n(enc, n)
        got, want = trainable(enc), expected_last_n(enc, enc_blocks, n)
        detail = (f'{len(want)} params' if got == want
                  else f'missing {sorted(want - got)[:3]} / extra {sorted(got - want)[:3]}')
        check(got == want,
              f'n = {n}: trainable == last {min(n, n_blocks)} block(s) + output norm', detail)
        check(not [name for name in got if 'embed' in name.lower()],
              f'n = {n}: the embeddings stay frozen')

    enc = AutoModel.from_pretrained(backbone)
    enc.eval()
    unfreeze_last_n(enc, 1)
    got, norm_names = trainable(enc), sorted(output_norm_params(enc))
    outside_at_one[label] = sorted(name for name in got if layer_index_in_name(name) is None)
    print(f'  n = 1 -> {len(got)} trainable params; of the output norm {norm_names or "[]"}, '
          f'trainable: {[name for name in norm_names if name in got] or "none"}')
    del enc, base, blocks
    gc.collect()

print('\n=== the top of the tower, across the two architectures ===')
for label in outside_at_one:
    print(f'  {label}: trainable params outside any block -> {outside_at_one[label] or "none"}')
check(len(outside_at_one) == 2, 'measured the top of the tower for both backbones')
check(outside_at_one[BACKBONES[0][0]] == [],
      'BERT-style: nothing outside the blocks is needed, its top norm is inside block N')
check(outside_at_one[BACKBONES[1][0]] == ['final_norm.weight'],
      'ModernBERT: the output norm is included with the last block',
      str(outside_at_one[BACKBONES[1][0]]))

print('\n=== separate encoders: both towers, and the optimizer groups ===')
import model_xattn  # noqa: E402

net = model_xattn.build_model(SMALL, unfreeze_layers=1, layers=1,
                              use_pooled_v=False, separate_encoders=True)
tr_comment, tr_video = trainable(net.encoder), trainable(net.encoder_v)
check(tr_comment == tr_video, 'the comment and the video tower get the same freeze pattern',
      f'{len(tr_comment)} vs {len(tr_video)} params')
check(bool(tr_comment), 'the last block of each tower is actually trainable')

groups = model_xattn.trainable_groups(net)
flat = [p for group in groups for p in group]
ids = [id(p) for p in flat]
all_trainable = {id(p) for p in net.parameters() if p.requires_grad}
frozen = {id(p) for p in net.parameters() if not p.requires_grad}
check(set(ids) == all_trainable, 'the optimizer groups cover every trainable parameter',
      f'{len(set(ids))} unique vs {len(all_trainable)} trainable')
check(len(ids) == len(set(ids)), 'no trainable parameter is listed in two groups')
check(not (set(ids) & frozen), 'no frozen parameter leaked into a group')
backbone_group, block_group, head_group = groups
check(bool(backbone_group) and bool(block_group) and bool(head_group),
      'all three groups are non-empty at unfreeze=1',
      f'backbone {len(backbone_group)} / blocks {len(block_group)} / head {len(head_group)}')

print('\n=== out-of-range n must fail loudly, not silently ===')
enc = AutoModel.from_pretrained(SMALL)
try:
    unfreeze_last_n(enc, -1)
    check(False, 'n = -1 raises', f'silently returned; {len(trainable(enc))} params trainable')
except ValueError:
    check(True, 'n = -1 raises')

enc = AutoModel.from_pretrained(SMALL)
_, blocks = find_blocks(enc)
n_blocks = len(blocks)
check(count_encoder_blocks(enc) == n_blocks, 'the small backbone is counted correctly',
      f'{n_blocks} blocks')
enc = AutoModel.from_pretrained(SMALL)
try:
    unfreeze_last_n(enc, n_blocks + 3)
    check(False, f'n = {n_blocks + 3} (over {n_blocks} blocks) raises',
          f'silently unfroze {len(trainable(enc))} params')
except ValueError:
    check(True, f'n = {n_blocks + 3} (over {n_blocks} blocks) raises')

enc = AutoModel.from_pretrained(SMALL)
_, blocks = find_blocks(enc)
unfreeze_last_n(enc, len(blocks))
check(trainable(enc) == expected_last_n(enc, blocks, len(blocks)),
      f'n = {len(blocks)} (exactly the block count) is accepted, not treated as out of range')

print()
if FAILURES:
    print(f'FAILURES ({len(FAILURES)}):')
    for failure in FAILURES:
        print('  -', failure)
    sys.exit(1)
print('all checks passed')
