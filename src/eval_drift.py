#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate semantic drift of round-trip paraphrases with a confidence interval.

Reads the *_review_<LANG>.tsv files written by augment_roundtrip.py and scores
how much meaning survived the round trip. Two signals:

  * cosine similarity between multilingual sentence embeddings of the source
    comment and its round-trip (automatic, scales to any sample size), and
  * a binary drift label -- either the --drift-label column from a hand-annotated
    sample, or `cosine < --threshold` when no hand label is present.

A row counts as drifted when its hand label says so, otherwise when its cosine
falls below --threshold. The drift rate is reported with a 95% Wilson score
interval overall and broken down by language and label, so a small random
sample is enough to bound the true rate instead of inspecting every pair.

Usage:
    python src/eval_drift.py aug_rt_review_EN.tsv aug_rt_review_IT.tsv
    python src/eval_drift.py aug_rt_review_*.tsv --threshold 0.8
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Z = 1.96  # 95% interval
TRUTHY = ('1', 'yes', 'true', 'y', 't')
FALSY = ('0', 'no', 'false', 'n', 'f')


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def mean_ci(values, z: float = Z):
    """Mean with a normal-approximation 95% interval."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = z * (var / n) ** 0.5
    return mean, mean - half, mean + half


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='review TSVs from augment_roundtrip.py')
    parser.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Cosine below this counts as drifted when no hand label is present')
    parser.add_argument('--drift-label', default=None,
                        help='Optional TSV column holding a 1/0 (yes/no) hand drift label')
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    return parser.parse_args(argv)


def read_reviews(paths, drift_label):
    rows = []
    for path in paths:
        with open(path, encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_ALL):
                hand = None
                if drift_label is not None:
                    raw = str(r.get(drift_label, '')).strip().lower()
                    if raw in TRUTHY:
                        hand = 1
                    elif raw in FALSY:
                        hand = 0
                rows.append({
                    'src': r.get('yt_comment', ''), 'rt': r.get('roundtrip', ''),
                    'lang': r.get('lang', ''), 'stereotype': r.get('stereotype', ''),
                    'token_overlap': _to_float(r.get('token_overlap')),
                    'len_ratio': r.get('len_ratio', ''), 'hand_drift': hand,
                })
    return rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_pct(x):
    return f'{100.0 * x:.1f}%'


def report(title, subset):
    n = len(subset)
    if not n:
        print(f'  {title}: n=0')
        return
    k = sum(r['drifted'] for r in subset)
    lo, hi = wilson(k, n)
    cmean, clo, chi = mean_ci([r['cosine'] for r in subset])
    overlaps = [r['token_overlap'] for r in subset if r['token_overlap'] is not None]
    ov = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(f'  {title}: n={n} | drift {k}/{n} = {fmt_pct(k / n)} '
          f'95% CI [{fmt_pct(lo)}, {fmt_pct(hi)}] | cosine {cmean:.3f} [{clo:.3f}, {chi:.3f}] '
          f'| token_overlap {ov:.3f}')


def main():
    args = parse_args()
    rows = read_reviews(args.paths, args.drift_label)
    if not rows:
        raise SystemExit('error: no rows read from ' + ', '.join(args.paths))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    src = model.encode([r['src'] for r in rows], normalize_embeddings=True, show_progress_bar=False)
    rt = model.encode([r['rt'] for r in rows], normalize_embeddings=True, show_progress_bar=False)
    for r, a, b in zip(rows, src, rt):
        r['cosine'] = float((a * b).sum())
        r['drifted'] = r['hand_drift'] if r['hand_drift'] is not None \
            else int(r['cosine'] < args.threshold)

    hand = sum(1 for r in rows if r['hand_drift'] is not None)
    print(f'Pairs: {len(rows)} | model: {args.model} | cosine threshold: {args.threshold} '
          f'| hand-labelled: {hand}')
    print('Drift rate (Wilson 95% CI):')
    report('ALL', rows)
    for lang in sorted({r['lang'] for r in rows}):
        report(f'lang {lang}', [r for r in rows if r['lang'] == lang])
    for lab in sorted({r['stereotype'] for r in rows}):
        report(f'label {lab}', [r for r in rows if r['stereotype'] == lab])


if __name__ == '__main__':
    main()
