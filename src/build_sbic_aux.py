#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds Task-A schema TSVs from SBIC.v2 aggregate files.

Outputs (written next to the input files, EN language tag in the filename so
load_frame_from_file infers the right prompts):
  sbic_lgbtq_EN_training.tsv : posts targeting LGBTQ groups (gay/lesbian/trans/
                               bisexual/queer) labelled stereotype=yes, balanced
                               with non-biased posts (no). High-precision Task-A
                               positives outside StereoQueerEval.
  sbic_bias_EN_training.tsv  : stage-1 skill pool: hasBiasedImplication=1 posts
                               (yes) balanced with hasBiasedImplication=0 (no);
                               the biased-implicature concept is the closest
                               existing cousin of 'stereotype'.

Both share the 7-column schema and feed train.py via --train-extras / --input.
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys

import pandas as pd

REQUIRED_COLUMNS = [
    'StereoQueerEval_id',
    'yt_title',
    'yt_description',
    'yt_comment',
    'stereotype',
    'hate_speech',
    'target',
]

LGBTQ_KEY = re.compile(r'lgbt|gay|lesbian|trans|queer|homosexual|bisexual', re.I)


def parse_list(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        try:
            return [str(v) for v in ast.literal_eval(x)]
        except Exception:
            return []
    return []


def clean_post(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip()


def write_tsv(path: str, rows: list) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS,
                                delimiter='\t', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agg-dir', default=r'C:\CODE_SOMETHING\LGBT\SBIC.v2',
                        help='Folder holding SBIC.v2.agg.{trn,dev,tst}.csv')
    parser.add_argument('--out-dir', default=r'C:\CODE_SOMETHING\LGBT\LGBT',
                        help='Where the built TSVs are written')
    parser.add_argument('--neg-ratio', type=float, default=1.0,
                        help='Non-biased negatives per positive for the separate files')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    frames = []
    for split in ('trn', 'dev', 'tst'):
        path = os.path.join(args.agg_dir, f'SBIC.v2.agg.{split}.csv')
        if not os.path.exists(path):
            print(f'skip missing {path}', file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df['_split'] = split
        frames.append(df)
    if not frames:
        raise SystemExit('No SBIC.v2.agg.*.csv files found.')
    df = pd.concat(frames, ignore_index=True)
    print(f'Loaded {len(df)} SBIC rows ({args.agg_dir})')

    df['_tm'] = df['targetMinority'].apply(parse_list)
    df['_ts'] = df['targetStereotype'].apply(parse_list)
    df['_lgbtq'] = df['_tm'].apply(lambda xs: any(LGBTQ_KEY.search(x) for x in xs))

    def base_row(row, prefix, i):
        return {
            'StereoQueerEval_id': f'{prefix}-{i:06d}',
            'yt_title': '',
            'yt_description': '',
            'yt_comment': clean_post(row['post']),
            'stereotype': 'yes',
            'hate_speech': '',
            'target': ' | '.join(row['_ts']) if row['_ts'] else ' | '.join(row['_tm']),
        }

    negs = df[df['hasBiasedImplication'] == 0].reset_index(drop=True)

    # 1) LGBTQ-targeted precision set (all splits), balanced with non-biased rows.
    lgbtq = df[df['_lgbtq']].reset_index(drop=True)
    n_neg = int(len(lgbtq) * args.neg_ratio)
    neg_sample = negs.sample(n=min(n_neg, len(negs)), random_state=args.seed).reset_index(drop=True)
    rows = [base_row(row, 'SBIC-LGBTQ', i) for i, (_, row) in enumerate(lgbtq.iterrows())]
    for i, row in enumerate(neg_sample.itertuples(), start=len(lgbtq)):
        rows.append({
            'StereoQueerEval_id': f'SBIC-LGBTQ-{i:06d}',
            'yt_title': '',
            'yt_description': '',
            'yt_comment': clean_post(row.post),
            'stereotype': 'no',
            'hate_speech': '',
            'target': '',
        })
    write_tsv(os.path.join(args.out_dir, 'sbic_lgbtq_EN_training.tsv'), rows)

    # 2) Bias-implicature stage-1 pool (all splits), balanced.
    bias = df[df['hasBiasedImplication'] == 1].reset_index(drop=True)
    n_neg = int(len(bias) * args.neg_ratio)
    neg_sample2 = negs.sample(n=min(n_neg, len(negs)), random_state=args.seed).reset_index(drop=True)
    rows2 = [base_row(row, 'SBIC-BIAS', i) for i, (_, row) in enumerate(bias.iterrows())]
    for i, row in enumerate(neg_sample2.itertuples(), start=len(bias)):
        rows2.append({
            'StereoQueerEval_id': f'SBIC-BIAS-{i:06d}',
            'yt_title': '',
            'yt_description': '',
            'yt_comment': clean_post(row.post),
            'stereotype': 'no',
            'hate_speech': '',
            'target': '',
        })
    write_tsv(os.path.join(args.out_dir, 'sbic_bias_EN_training.tsv'), rows2)

    print(f'LGBTQ positives: {len(lgbtq)} | bias positives: {len(bias)}')


if __name__ == '__main__':
    main()