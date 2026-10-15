#!/usr/bin/env python3
"""Frozen-dropout invariant test for the xattn classifier.

The invariant: a Dropout that sits anywhere inside a subtree whose parameters are
ALL frozen must be in eval(), because that subtree cannot learn and a random mask
inside it only injects noise into a representation the model calls frozen.

Expected mode is derived from requires_grad alone, so it does not depend on which
train()/eval() the code happened to call - that is the point.

Also does a functional check on the fully frozen case: the frozen tower's output
must be bit-identical across two forwards in train mode (dropout off), and must
differ when --keep-frozen-dropout asks for the old behaviour.

Run:  python probe_frozen_state.py
Exit code 0 = invariant holds everywhere.
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import model_xattn
from model import eval_frozen_modules, unfreeze_last_n

BACKBONE = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

FAILURES = []


def frozen_modules(net):
    return {
        name for name, sub in net.named_modules()
        if (ps := list(sub.parameters(recurse=True))) and not any(p.requires_grad for p in ps)
    }


def in_frozen_region(name, frozen):
    if name in frozen:
        return True
    parts = name.split('.')
    for i in range(len(parts) - 1, 0, -1):
        if '.'.join(parts[:i]) in frozen:
            return True
    return False


def audit(net, tag, expect_frozen_dropout_off=True):
    """Check every Dropout against the mode required by the invariant."""
    frozen = frozen_modules(net)
    wrong = []
    n_frozen = n_live = 0
    for name, mod in net.named_modules():
        if not isinstance(mod, nn.Dropout):
            continue
        if in_frozen_region(name, frozen):
            n_frozen += 1
            should_train = not expect_frozen_dropout_off
        else:
            n_live += 1
            should_train = True
        if bool(mod.training) != should_train:
            wrong.append((name, 'train' if should_train else 'eval',
                          'train' if mod.training else 'eval'))
    ok = not wrong
    if not ok:
        FAILURES.append(f'{tag}: {len(wrong)} dropout(s) in the wrong mode')
    print(f'  [{"OK " if ok else "FAIL"}] {tag:<50} '
          f'frozen {n_frozen:>3} live {n_live:>3}'
          + ('' if ok else f'  <- {len(wrong)} wrong'))
    for name, exp, act in wrong[:3]:
        print(f'           {name}: expected {exp}, got {act}')
    return ok


def build(unfreeze, keep, shared=False, layers=1):
    return model_xattn.build_model(BACKBONE, unfreeze_layers=unfreeze, heads=4, layers=layers,
                                   use_pooled_v=False, separate_encoders=not shared,
                                   keep_frozen_dropout=keep)


def tower_fingerprint(net, tower):
    """Two forwards of the same ids; compare the tower's hidden states bitwise."""
    torch.manual_seed(0)
    ids = torch.randint(3, 1000, (2, 10))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = net.encode(ids, mask, tower=tower).clone()
        b = net.encode(ids, mask, tower=tower).clone()
    return bool(torch.equal(a, b))


print('=== 1. default (keep_frozen_dropout=False), unfreeze_layers=2 ===')
net = build(2, False)
audit(net, '1a after build_model')
net.train()
audit(net, '1b after net.train()  <-- reviewer claim')
eval_frozen_modules(net)          # must stay redundant, not break anything
audit(net, '1c train() then a redundant eval_frozen_modules')
net.eval()
frozen_all = all(not m.training for m in net.modules() if isinstance(m, nn.Dropout))
if not frozen_all:
    FAILURES.append('1d net.eval() left a Dropout in train mode')
print(f'  [{"OK " if frozen_all else "FAIL"}] 1d net.eval() turns every Dropout off')

print()
print('=== 2. fully frozen encoder (unfreeze_layers=0) ===')
net0 = build(0, False)
audit(net0, '2a after build_model')
net0.train()
audit(net0, '2b after net.train()  <-- this used to FAIL')
print(f'  [{"OK " if tower_fingerprint(net0, "video") else "FAIL"}] '
      f'2c frozen tower output is bit-identical across forwards')
if not tower_fingerprint(net0, 'video'):
    FAILURES.append('2c frozen tower is stochastic in train mode')

print()
print('=== 3. --keep-frozen-dropout: frozen dropout MUST fire ===')
netk0 = build(0, True)
netk0.train()
audit(netk0, '3a keep=True, unfreeze 0', expect_frozen_dropout_off=False)
netk = build(2, True)
netk.train()
audit(netk, '3b keep=True, unfreeze 2', expect_frozen_dropout_off=False)

print()
print('=== 3e. functional: keep_frozen_dropout=True must make the tower stochastic ===')
if tower_fingerprint(netk, 'video'):
    FAILURES.append('keep_frozen_dropout=True did not keep the frozen dropout stochastic')
    print('  [FAIL] frozen tower output identical - the flag is a no-op')
else:
    print('  [OK ] frozen tower output differs - the flag is live')

print()
print('=== 3f. --keep-frozen-dropout with unfreeze_layers=0 (was a silent no-op) ===')
netk0.train()
if tower_fingerprint(netk0, 'video'):
    FAILURES.append('keep_frozen_dropout=True is a no-op with a fully frozen encoder')
    print('  [FAIL] frozen tower output identical - flag ignored in the frozen case')
else:
    print('  [OK ] frozen tower output differs - flag honoured')

print()
print('=== 4. --freeze-epochs transition (0 -> 2 blocks, mid-run) ===')
netf = build(0, False)
netf.train()
audit(netf, '4a phase 1 (encoder frozen)')
for enc in netf.encoders():
    unfreeze_last_n(enc, 2)
netf.train()
audit(netf, '4b phase 2 after unfreeze + net.train()')
print('  [-- ] phase 2 tower fingerprint skipped: blocks 10-11 are trainable there, '
      'so the tower is stochastic by design')

print()
print('=== 5. --shared-encoder ===')
nets = build(0, False, shared=True)
audit(nets, '5a shared encoder, fully frozen')
nets.train()
audit(nets, '5b shared encoder after net.train()')

print()
print('=== 6. prompt baseline (MaskTargetedClassifier, src/train.py path) ===')
from transformers import AutoModel
from model import MaskTargetedClassifier

base_enc = AutoModel.from_pretrained(BACKBONE)
# A stand-in MLM head; the point is the *encoder* subtree, not the head.
dummy_head = nn.Linear(base_enc.config.hidden_size, base_enc.config.hidden_size)
mtc = MaskTargetedClassifier(base_enc, dummy_head, base_enc.config.hidden_size, num_classes=2)
unfreeze_last_n(mtc.encoder, 0)           # the fully frozen backbone, as train.py builds it
dummy_head.requires_grad_(False)
mtc.train()
eval_frozen_modules(mtc)                  # exactly what train.py line 353 does
stuck = [n for n, m in mtc.encoder.named_modules()
         if isinstance(m, nn.Dropout) and m.training]
if stuck:
    FAILURES.append(
        f'6: fully frozen baseline encoder left {len(stuck)} dropouts in train mode '
        f'after eval_frozen_modules(model) - the prompt baseline trained with a '
        f'stochastic "frozen" encoder')
    print(f'  [FAIL] frozen baseline encoder: {len(stuck)} dropouts still in train mode')
else:
    print('  [OK ] frozen baseline encoder fully in eval() after eval_frozen_modules(model)')

print()
if FAILURES:
    print(f'FAILURES ({len(FAILURES)}):')
    for f in FAILURES:
        print('  -', f)
    sys.exit(1)
print('all checks passed')
