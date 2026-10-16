#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is a *new* backbone usable in this model, and what does it cost?

`probe_unfreeze.py`, `probe_allpad.py` and `probe_frozen_state.py` answer this for
the two backbones the repo shipped with (MiniLM, mmBERT). This probe answers it for
any other checkpoint *before* a screen is spent on it, through the same code path
the trainer uses (`model_xattn.load_encoder`), so what is verified is what runs:

  a. the checkpoint loads, and reports its class, size and hidden dimension;
  b. the block stack is found by name (`count_encoder_blocks` > 0, indices
     0..n-1), and `d_model` divides by the number of attention heads;
  c. no block is causal: a decoder-style checkpoint (`is_causal = True`, e.g. a
     plain Qwen3/LLaMA embedding model) would make every token only see the
     tokens before it, which silently breaks the cross-attention decoder that
     reads *all* of them;
  d. `unfreeze_last_n` for n in {0, 1, 2, all}: the trainable set is exactly the
     last n blocks plus the output norm where that is a sibling of the stack
     (EuroBERT's `norm`, ModernBERT's `final_norm`; BERT folds it into block N),
     and the embeddings stay frozen at every n;
  e. a forward pass at the trainer's shapes returns a finite `last_hidden_state`;
  f. how many Dropout modules the encoder contains, and the largest p among
     them. Reported, not asserted: an encoder with none (EuroBERT / Qwen3 have
     both dropouts at 0.0) makes the frozen-module invariant in
     `XAttnClassifier.train` vacuous rather than broken - the frozen half is
     deterministic either way and `--keep-frozen-dropout` is a no-op - and it
     means unfreezing blocks adds no stochastic regularisation of its own. The
     same holds, less obviously, when the modules exist but every p is 0
     (mmBERT-base), which is why the check that used to be here was wrong;
  g. what the encoder does with an all-padded row. Reported, not asserted: the
     trainer rejects all-padded *video* rows before the encoder is reached
     (`XAttnClassifier._validate_video_mask`) and the data layer rejects them for
     comments, so this is about knowing whether the backbone would return NaN if
     one ever slipped through - the reason `probe_allpad.py` exists;
  h. one forward/backward step at the trainer's shapes, and the epoch cost that
     implies, so a screen is priced before it is started.

Run:  python src/probe_backbone.py --backbone jinaai/jina-embeddings-v5-text-nano
Exit code 0 = the checkpoint is usable. (f) and (g) print warnings, they do not fail.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from model import block_groups, count_encoder_blocks, layer_index_in_name, unfreeze_last_n  # noqa: E402

FAILURES = []


def check(ok, label, extra=''):
    print(f'    [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


def note(label, extra=''):
    print(f'    [--] {label}' + (f'  {extra}' if extra else ''))


def trainable_names(module):
    return {n for n, p in module.named_parameters() if p.requires_grad}


def output_norm_params(encoder):
    """Parameters of the encoder's output normalisation, when it sits outside the
    block stack. Matched on *module* names, which is what `unfreeze_last_n` uses:
    EuroBERT keeps it as a sibling module `norm`, so its single parameter is
    `norm.weight` - a name that contains a dot, which is why the rule has to be
    applied to the module and not to the parameter name."""
    names = set()
    for name, module in encoder.named_modules():
        if '.' not in name and 'norm' in name.lower():
            # `named_parameters` is relative to the module, so the prefix has to be
            # put back on: the module is top-level (`norm`), its parameter is
            # `norm.weight`.
            names |= {f'{name}.{param}' for param, _ in module.named_parameters(recurse=True)}
    return names


def expected_last_n(encoder, n_layers):
    """The trainable names `unfreeze_last_n(encoder, n_layers)` should leave.

    Derives the answer from the module tree instead of trusting the helper, the
    way `probe_unfreeze.py` does: the last n block indices by name, plus the
    output normalisation *when the architecture keeps it outside the stack*.
    """
    groups = block_groups(encoder)
    if n_layers == 0 or not groups:
        return set()
    keep = set(sorted(groups)[-n_layers:])
    wanted = {n for n, _ in encoder.named_parameters()
              if layer_index_in_name(n) in keep}
    wanted |= output_norm_params(encoder)
    return wanted


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', default='jinaai/jina-embeddings-v5-text-nano')
    p.add_argument('--heads', type=int, default=4, help='--heads the run would use')
    p.add_argument('--layers', type=int, default=3, help='--layers the run would use')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--comment-len', type=int, default=160, help='--max-len of the run')
    p.add_argument('--video-len', type=int, default=256, help='--video-max-len of the run')
    p.add_argument('--timing-steps', type=int, default=2)
    p.add_argument('--steps-per-epoch', type=int, default=262,
                   help='EN+IT+NL train split at batch 32 with the default val fraction '
                        '(0.1): 9288 rows - 10%% val ~ 8360 / 32. Lengthgrouped batches of '
                        'shorter rows are cheaper, so treat this as an upper bound; only '
                        'used for the cost estimate')
    p.add_argument('--shared-encoder', action='store_true',
                   help='measure the shared-encoder cost (half the towers); the trainer\'s '
                        'default is separate encoders, which is what is measured without this')
    p.add_argument('--no-timing', action='store_true', help='skip the fwd/bwd measurement')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import model_xattn

    print(f'=== {args.backbone} ===')
    t0 = time.time()
    encoder = model_xattn.load_encoder(args.backbone)
    encoder.eval()
    print(f'  loaded in {time.time() - t0:.1f}s as {type(encoder).__name__} '
          f'({args.backbone} -> model_type {model_xattn.encoder_model_type(args.backbone)!r})')

    # (a) shape of the checkpoint
    params = sum(p.numel() for p in encoder.parameters())
    d_model = encoder.config.hidden_size
    n_blocks = count_encoder_blocks(encoder)
    print(f'  {params / 1e6:.1f}M params | hidden {d_model} | {n_blocks} blocks')

    # (b) the name-based block search the freeze/unfreeze logic depends on
    print('\n--- the block stack ---')
    indices = {i for i in (layer_index_in_name(n) for n, _ in encoder.named_parameters())
               if i is not None}
    check(n_blocks > 0, 'at least one encoder block is found by name', f'{n_blocks} blocks')
    check(indices == set(range(n_blocks)),
          'every block index is detected, and no phantom index appears',
          f'{sorted(indices)[:3]}...{sorted(indices)[-3:]}' if indices else 'none')
    check(d_model % args.heads == 0,
          f'hidden {d_model} divides by --heads {args.heads}', f'd_model/heads = {d_model / args.heads:g}')

    # (c) a causal encoder would break a decoder that reads every token
    print('\n--- attention direction ---')
    causal_flags = {name for name, mod in encoder.named_modules()
                    if getattr(mod, 'is_causal', False)}
    if any(hasattr(mod, 'is_causal') for _, mod in encoder.named_modules()):
        check(not causal_flags,
              'every attention layer is non-causal (tokens see both directions)',
              f'causal: {sorted(causal_flags)[:3]}' if causal_flags else '')
    else:
        note('no `is_causal` flag on this architecture (BERT-style: always bi-directional)')

    # (d) the freeze boundary, including where the output norm lives
    print('\n--- unfreeze_last_n ---')
    base = model_xattn.load_encoder(args.backbone)
    base.eval()
    outside = sorted(output_norm_params(base))
    print(f'  normalisation outside the block stack: {outside or "none (folded into block N)"}')
    for n in (0, 1, 2, n_blocks):
        enc = model_xattn.load_encoder(args.backbone)
        enc.eval()
        unfreeze_last_n(enc, n)
        got, want = trainable_names(enc), expected_last_n(enc, n)
        detail = (f'{len(want)} params' if got == want else
                  f'missing {sorted(want - got)[:3]} / extra {sorted(got - want)[:3]}')
        check(got == want, f'n = {n}: trainable == last {min(n, n_blocks)} block(s) + output norm', detail)
        check(not [name for name in got if 'embed' in name.lower()],
              f'n = {n}: the embeddings stay frozen')
        del enc
    del base

    # (e) forward at the trainer's shapes
    print('\n--- forward at the trainer\'s shapes ---')
    from model import load_tokenizer
    tok = load_tokenizer(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batch = tok(['questo è un commento di prova in italiano'] * args.batch_size,
                return_tensors='pt', padding='max_length', truncation=True,
                max_length=args.comment_len)
    with torch.no_grad():
        out = encoder(**batch)
    hidden = out.last_hidden_state
    check(tuple(hidden.shape) == (args.batch_size, args.comment_len, d_model),
          'last_hidden_state has the shape the towers expect', str(tuple(hidden.shape)))
    check(bool(torch.isfinite(hidden).all()), 'the output is finite on a padded batch')
    note('tokenizer', f'{type(tok).__name__}, vocab {tok.vocab_size}, pad {tok.pad_token!r}, '
                      f'segment separator {tok.sep_token or tok.eos_token!r}')

    # (f) dropout inventory
    print('\n--- dropout ---')
    import torch.nn as nn
    drops = [(n, mod.p) for n, mod in encoder.named_modules() if isinstance(mod, nn.Dropout)]
    if drops:
        max_p = max(p for _, p in drops)
        note(f'{len(drops)} Dropout modules', f'e.g. {drops[0][0]}, max p = {max_p:g}')
        if max_p > 0:
            note('the frozen/train invariant is a real switch here',
                 'the trainable blocks are stochastic and the frozen ones are not')
        else:
            note('every Dropout module is switched off (p = 0)',
                 'an encoder whose dropout is present but at 0 is as deterministic as '
                 'one with none: --keep-frozen-dropout is a no-op and unfreezing adds '
                 'no regularisation')
    else:
        note('0 Dropout modules',
             'frozen and unfrozen blocks are equally deterministic here, so '
             '--keep-frozen-dropout is a no-op and unfreezing adds no regularisation')

    # (g) an all-padded row
    print('\n--- all-padded row (reported, not asserted) ---')
    pad = tok(['x'], return_tensors='pt')['input_ids'][0, :1]
    ids = (pad * torch.ones(1, args.comment_len, dtype=torch.long))
    with torch.no_grad():
        out = encoder(input_ids=ids, attention_mask=torch.zeros(1, args.comment_len, dtype=torch.long))
    finite = bool(torch.isfinite(out.last_hidden_state).all())
    note('all-padded row is finite' if finite else 'all-padded row returns NaN',
         'the trainer rejects these rows before the encoder; see probe_allpad.py')

    # (h) cost
    if not args.timing_steps:
        print()
        return finish()
    print(f'\n--- cost: one fwd/bwd step at batch {args.batch_size}, '
          f'{args.comment_len}/{args.video_len} tokens, --layers {args.layers} ---')
    del encoder
    net = model_xattn.build_model(args.backbone, unfreeze_layers=2, heads=args.heads,
                                  layers=args.layers, dropout=0.1,
                                  separate_encoders=not args.shared_encoder)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'  {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params, '
          f'{n_train / 1e6:.1f}M trainable at --unfreeze-layers 2')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.CrossEntropyLoss()
    ids = torch.randint(0, 1000, (args.batch_size, args.comment_len))
    vids = torch.randint(0, 1000, (args.batch_size, args.video_len))
    cm = torch.ones(args.batch_size, args.comment_len, dtype=torch.bool)
    vm = torch.ones(args.batch_size, args.video_len, dtype=torch.bool)
    labels = torch.randint(0, 2, (args.batch_size,))
    net.train()
    step = 0.0
    for i in range(args.timing_steps + 1):        # one warm-up, not counted
        t1 = time.time()
        opt.zero_grad()
        loss = crit(net(ids, cm, vids, vm), labels)
        loss.backward()
        opt.step()
        if i:
            step += time.time() - t1
    step /= max(args.timing_steps, 1)
    per_epoch = step * args.steps_per_epoch
    print(f'  {step:.1f}s/step -> ~{per_epoch / 60:.0f} min/epoch '
          f'at {args.steps_per_epoch} steps ({per_epoch * 15 / 3600:.1f}h for 15 epochs)')
    return finish()


def finish():
    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) FAILED:')
        for label in FAILURES:
            print(f'  - {label}')
        return 1
    print('every check holds.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
