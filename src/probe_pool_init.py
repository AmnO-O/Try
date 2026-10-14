#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the comment attention-pool uniform at initialization?

The reviewer's item 6: `pool_query = torch.randn(d) * 0.02` together with
`scores = (h_c @ pool_query) / sqrt(d_model)` under a near-unit hidden state
should make the initial scores small, hence the initial softmax distribution
close to uniform - i.e. the pool starts as roughly mean pooling and has to learn
to select. That is not a correctness bug; it is a claim about the initialization,
and it is measurable, so this measures it instead of asserting it:

  * the RMS of `h_c` (the encoder output on real tokenization);
  * the std of the scores at init;
  * the entropy of the pooling weights as a *fraction of its maximum* for that
    row, `H / log(Lq_valid)` - 1.0 is exactly uniform, 0.0 is a point mass;
  * the same for several `pool_init_scale` values with a fixed seed, which is
    the baseline the controlled ablation (change only this flag) needs.

Nothing is trained; the question is about init.

Run:  python src/probe_pool_init.py
Exit code 0 = every check holds.
"""

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import model_xattn  # noqa: E402
from model import load_tokenizer  # noqa: E402

BACKBONE = 'sentence-transformers/paraphrase-MiniLM-L3-v2'   # small, same d=384
SCALES = (0.02, 0.05, 0.1, 0.5, 1.0)
FAILURES = []

TEXTS = [
    'This is such a funny video, I laughed the whole time.',
    'Che schifo, questa persona merita solo insulti.',
    'Wat een belachelijk filmpje, echt niet te doen.',
    'Nobody should talk like that to another human being.',
    'Gli Italiani sono tutti pigri e non vogliono lavorare.',
    'Wat een dom volk, allemaal hetzelfde.',
]


def check(ok, label, extra=''):
    print(f'    [{"OK " if ok else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
    if not ok:
        FAILURES.append(label)
    return bool(ok)


def entropy_fraction(weights, valid):
    """H(weights) / log(n_valid), row-wise. 1.0 = uniform, 0.0 = point mass."""
    log_w = torch.log(weights.clamp_min(1e-30))
    h = -(weights * log_w).sum(dim=1)
    return h / torch.log(valid.clamp_min(2).to(h.dtype))


def main():
    tokenizer = load_tokenizer(BACKBONE)
    batch = tokenizer(TEXTS, padding=True, truncation=True, max_length=64,
                      return_tensors='pt')
    ids, mask = batch['input_ids'], batch['attention_mask']

    net = model_xattn.build_model(BACKBONE, unfreeze_layers=2, heads=4, layers=3,
                                  separate_encoders=False, use_pooled_v=False)
    net.eval()
    d_model = net.d_model

    print(f'=== tokenization: {ids.shape[0]} comments, {mask.shape[1]} slots '
          f'({int(mask.sum(1).min())}..{int(mask.sum(1).max())} valid) ===')

    # h_c itself, one forward with no gradient.
    with torch.no_grad():
        h_c = net.encode(ids, mask, tower='comment')
    rms = h_c[mask.bool()].pow(2).mean().sqrt()
    print(f'  hidden state on valid tokens: RMS {rms.item():.3f}')

    scale_entropy = {}
    scale_max = {}
    print(f'\n  {"pool_init_scale":>15} {"query RMS":>10} {"score std":>10} '
          f'{"H/log(Lq)":>10} {"max weight":>11}')
    for scale in SCALES:
        torch.manual_seed(0)
        with torch.no_grad():
            net.pool_query.data = torch.randn(d_model) * scale
        q = net.pool_query.detach()
        scores = (h_c @ q) / math.sqrt(d_model)
        with torch.no_grad():
            weights = torch.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=1)
        frac = entropy_fraction(weights, mask.sum(1)).mean().item()
        max_w = weights.max(dim=1).values.mean().item()
        valid_scores = scores[mask.bool()]
        scale_entropy[scale] = frac
        scale_max[scale] = max_w
        print(f'  {scale:>15} {q.pow(2).mean().sqrt().item():>10.5f} '
              f'{valid_scores.std().item():>10.5f} {frac:>10.4f} {max_w:>11.4f}')

    print()
    default = scale_entropy[0.02]
    check(default >= 0.999,
          'at the default scale 0.02 the pool is uniform (init = mean pooling)',
          f'H/log(Lq) = {default:.4f}')
    check(scale_entropy[1.0] < default - 0.005,
          'a larger query scale gives a genuinely sharper init, so the flag is live',
          f'H/log(Lq) {default:.4f} -> {scale_entropy[1.0]:.4f}')
    check(scale_max[1.0] > scale_max[0.02],
          'the sharpest token gets more mass as the scale rises',
          f'max weight {scale_max[0.02]:.4f} -> {scale_max[1.0]:.4f}')
    check(all(scale_entropy[a] >= scale_entropy[b] - 1e-9
              for a, b in zip(SCALES, SCALES[1:])),
          'entropy is monotone in the scale across the sweep',
          ', '.join(f'{s}:{scale_entropy[s]:.4f}' for s in SCALES))

    print()
    if FAILURES:
        print(f'FAILURES ({len(FAILURES)}):')
        for name in FAILURES:
            print('  -', name)
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
