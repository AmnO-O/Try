#!/usr/bin/env python3
"""All-padding rows: what breaks, in the forward pass *and* the backward pass.

Two claims to check:
  1. Comment pooling. `masked_fill(-inf)` + `softmax` is 0/0 on a row with no
     finite score. The NaN does not stay in the forward value: softmax's backward
     multiplies by its own NaN output, so the gradient is NaN too, the loss is NaN
     and the run dies.
  2. All-padded video. The training path (`need_weights=False`) answers with zeros;
     the `need_weights=True` diagnostic path answers with NaN - weights *and* the
     comment output.

The fix: make the row's scores finite *before* the softmax (repairing the value
after the softmax fixes the forward number but not the gradient), and reject such
rows at load time - and, for the video side, at the model boundary too, so a caller
outside the trainer (inference, diagnostics, a probe) cannot reach the NaN path.

Run:  python src/probe_allpad.py
Exit code 0 = every assertion holds.
"""
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

BACKBONE = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FAILURES = []


def check(ok, label, extra=''):
    print(f'  [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


class OldPool(nn.Module):
    """The pooling exactly as it was."""

    def __init__(self, d):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, h_c, q_pad):
        scores = (h_c @ self.pool_query) / math.sqrt(h_c.shape[-1])
        scores = scores.masked_fill(q_pad, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        return torch.einsum('bl,bl d->bd', weights, h_c), weights


class NewPool(nn.Module):
    """The pooling as it is now."""

    def __init__(self, d):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, h_c, q_pad):
        scores = (h_c @ self.pool_query) / math.sqrt(h_c.shape[-1])
        scores = scores.masked_fill(q_pad, float('-inf'))
        has_token = (~q_pad).any(dim=1, keepdim=True)
        scores = torch.where(has_token, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(q_pad, 0.0)
        return torch.einsum('bl,bl d->bd', weights, h_c), weights


print('=== 1. pooling, pure tensors: forward AND backward ===')
D = 8
torch.manual_seed(0)
h_base = torch.randn(3, 6, D)
q_pad = torch.zeros(3, 6, dtype=torch.bool)     # row 0: valid
q_pad[1] = True                                 # row 1: the malformed row
q_pad[2, 4:] = True                             # row 2: valid, partly padded


def run_pool(cls):
    torch.manual_seed(0)                        # identical pool_query in both classes
    module = cls(D)
    h = h_base.clone().requires_grad_(True)
    pooled, w = module(h, q_pad)
    pooled.sum().backward()
    return pooled, w.detach(), h.grad.clone()


old_pooled, old_w, old_hgrad = run_pool(OldPool)
new_pooled, new_w, new_hgrad = run_pool(NewPool)

print(f'  old: pooled row 1 = {old_pooled[1][:2].tolist()}  '
      f'| grad rows finite = {[bool(torch.isfinite(old_hgrad[i]).all()) for i in range(3)]}')
print(f'  new: pooled row 1 = {new_pooled[1][:2].tolist()}  '
      f'| grad rows finite = {[bool(torch.isfinite(new_hgrad[i]).all()) for i in range(3)]}')

check(bool(torch.isnan(old_pooled[1]).all()), '1a old: forward value of the all-padded row is NaN')
check(bool(torch.isnan(old_hgrad[1]).any()), '1b old: so is its gradient (the run-killer)')
check(bool(torch.isfinite(old_hgrad[0]).all()) and bool(torch.isfinite(old_hgrad[2]).all()),
      '1c old: the valid rows are unaffected, so the NaN is per-row')
check(bool(torch.isfinite(new_pooled).all()), '1d new: forward is finite for every row')
check(bool(torch.isfinite(new_hgrad).all()), '1e new: gradient is finite for every row')
check(bool((new_pooled[1] == 0).all()), '1f new: the all-padded row pools to exactly zeros')
check(bool((new_w[1] == 0).all()), '1g new: its pooling weights are exactly zeros')
check(torch.equal(new_pooled[0], old_pooled[0]) and torch.equal(new_pooled[2], old_pooled[2]),
      '1h new: the valid rows are bit-identical to the old implementation')
check(torch.equal(new_hgrad[0], old_hgrad[0]),
      '1i new: the valid rows\' gradients are bit-identical too')
check(bool((new_w[2, 4:] == 0).all()) and float(new_w[0].sum()) == 1.0,
      '1j masked positions still carry exactly zero weight')

print()
print('=== 2. MHA: an all-padded key row, both need_weights settings ===')
torch.manual_seed(0)
dm, hm, B, Lq, Lkv = 8, 2, 2, 3, 5
mha = nn.MultiheadAttention(dm, hm, batch_first=True, dropout=0.0).eval()
q_ = torch.randn(B, Lq, dm)
kv = torch.randn(B, Lkv, dm)
pad = torch.ones(B, Lkv, dtype=torch.bool)          # every key ignored
with torch.no_grad():
    out_fast, _ = mha(q_, kv, kv, key_padding_mask=pad, need_weights=False)
    _, w_diag = mha(q_, kv, kv, key_padding_mask=pad, need_weights=True,
                    average_attn_weights=False)
check(bool(torch.isfinite(out_fast).all()), '2a training path (need_weights=False) is finite')
check(bool((out_fast == 0).all()), '2b training path returns zeros, silently')
check(bool(torch.isnan(w_diag).any()), '2c diagnostic path (need_weights=True) returns NaN weights')
print('      -> the reviewer is right on both: silent zeros in training, NaN on the')
print('         diagnostic path, and neither one is loud.')

print()
print('=== 3. the data-layer guard ===')
import pandas as pd
from model import load_tokenizer
from train_xattn import PairedDataset, assert_nonempty_masks

check(assert_nonempty_masks('comment', [[1, 1], [1], [1, 1, 0]], ['a', 'b', 'c']) is None,
      '3a does not fire on normal masks')
try:
    assert_nonempty_masks('comment', [[1, 1], [0, 0, 0]], ['ok', 'the empty one'])
    check(False, '3b raises on an all-zero row')
except ValueError as err:
    msg = str(err)
    check('row 1' in msg and 'the empty one' in msg and 'comment' in msg,
          '3b raises and names the row and the text', msg)

tok = load_tokenizer(BACKBONE)
df = pd.DataFrame({'yt_comment': ['a real comment', '', '!!!'],
                   'yt_title': ['t', '', 'x'],
                   'yt_description': ['d', '', 'y'],
                   'st_y': [0, 1, 0]})
ds = PairedDataset(df, tok, comment_max_len=32, video_max_len=32)
check(all(any(m) for m in ds.comment_mask) and all(any(m) for m in ds.video_mask),
      '3c real tokenizer output never trips the guard (empty string included)',
      f'min comment tokens {min(sum(m) for m in ds.comment_mask)}')

print()
print('=== 4. end to end: one malformed row inside a real batch ===')
import model_xattn

net = model_xattn.build_model(BACKBONE, unfreeze_layers=2, heads=4, layers=1,
                              use_pooled_v=False, separate_encoders=False)
net.eval()
W = 32


def pack(seqs, masks):
    ids = torch.zeros(3, W, dtype=torch.long)
    msk = torch.zeros(3, W, dtype=torch.long)
    for r in range(3):
        ids[r, :len(seqs[r])] = torch.tensor(seqs[r])
        msk[r, :len(masks[r])] = torch.tensor(masks[r])
    return ids, msk


c_ids, c_m = pack(ds.comment_ids, ds.comment_mask)
v_ids, v_m = pack(ds.video_ids, ds.video_mask)
labels = torch.tensor([0, 1, 0])
c_bad = c_m.clone()
c_bad[1] = 0                                  # row 1: no comment tokens at all

with torch.no_grad():                         # like for like: same pass type
    ref = net(c_ids, c_m, v_ids, v_m)
    corrupted = net(c_ids, c_bad, v_ids, v_m)
row_diff = max((corrupted[0] - ref[0]).abs().max().item(),
               (corrupted[2] - ref[2]).abs().max().item())
bad_diff = (corrupted[1] - ref[1]).abs().max().item()

logits, aux = net(c_ids, c_bad, v_ids, v_m, return_attn=True)
loss = nn.CrossEntropyLoss()(logits, labels)
net.zero_grad()
loss.backward()
bad_grads = [n for n, p in net.named_parameters()
             if p.grad is not None and not bool(torch.isfinite(p.grad).all())]

check(bool(torch.isfinite(logits).all()), '4a logits are finite with a malformed row present')
check(bool(torch.isfinite(loss)), '4b loss is finite', f'{loss.item():.4f}')
check(not bad_grads, '4c no parameter received a NaN gradient',
      f'{len(bad_grads)} bad' if bad_grads else '')
check(bool((aux['pool_weights'][1] == 0).all()), '4d the malformed row has zero pooling weights')
check(row_diff == 0.0, '4e the two valid rows are unchanged, bit for bit',
      f'max abs diff {row_diff:.2e}')
check(bad_diff > 0, '4f only the malformed row moves off the reference',
      f'max abs diff {bad_diff:.2e}')

print()
print('=== 5. the model boundary rejects an all-padded video ===')
# Section 2 showed the raw MHA path is unsafe when every key is masked: zeros on
# the training path, NaN (weights and output) on the diagnostic path. The model
# now refuses such a row up front, so neither behaviour is reachable through the
# public API - on any backend, not just where the optimized path happens to run.
v_bad = v_m.clone()
v_bad[2] = 0                                        # row 2: no video tokens at all
for label, kwargs in (('training path', {}), ('diagnostic path', {'return_attn': True})):
    try:
        with torch.no_grad():
            net(c_ids, c_m, v_ids, v_bad, **kwargs)
        check(False, f'5a {label}: raises on an all-padded video row')
    except ValueError as err:
        msg = str(err)
        check('video' in msg and 'row 2' in msg,
              f'5a {label}: raises and names the issue', msg[:60] + '...')

# The shape guard, so a malformed mask is caught as clearly as an empty one.
for bad_mask, label in (
    (torch.ones(3, dtype=torch.long), 'a 1-D mask'),
    (torch.ones(3, 0, dtype=torch.long), 'a zero-length mask'),
):
    try:
        with torch.no_grad():
            net(c_ids, c_m, v_ids, bad_mask)
        check(False, f'5b rejects {label}')
    except ValueError:
        check(True, f'5b rejects {label}')

# A well-formed batch still runs, so the guard is not simply blocking everything.
with torch.no_grad():
    ok_logits = net(c_ids, c_m, v_ids, v_m)
check(bool(torch.isfinite(ok_logits).all()),
      '5c a well-formed batch still runs and is finite')
print('      -> the raw MHA is unsafe for an all-masked key row (section 2); the model')
print('         boundary rejects that row before the encoder or cross-attention, so')
print('         the NaN path is unreachable by construction.')

print()
print('=== 6. the comment fallback is safe on each supported backbone ===')
# The comment side keeps a defensive fallback (encoder on an all-zero mask, then
# the decoder/pooling guard). The preprocessing guard is the primary defence, but
# the fallback still runs the encoder on an all-zero mask, so that has to be finite
# or the fallback would be silently useless. Checked on both supported backbones.
from transformers import AutoModel, AutoTokenizer

for label, backbone in (('MiniLM-L12 (BERT-style)', BACKBONE),
                        ('mmBERT-base (ModernBERT)', 'jhu-clsp/mmBERT-base')):
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = AutoModel.from_pretrained(backbone)
    enc.eval()
    ids = torch.tensor([[tok.cls_token_id or 0, tok.sep_token_id or 0, 5, 6]])
    with torch.no_grad():
        hidden = enc(input_ids=ids, attention_mask=torch.zeros_like(ids)).last_hidden_state
    check(bool(hidden.isfinite().all()),
          f'6 encoder on an all-zero mask is finite [{label}]',
          f'hidden {tuple(hidden.shape)}')

print()
if FAILURES:
    print(f'FAILURES ({len(FAILURES)}):')
    for f in FAILURES:
        print('  -', f)
    sys.exit(1)
print('all checks passed')
