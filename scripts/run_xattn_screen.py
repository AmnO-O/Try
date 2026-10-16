"""Run the section-6 xattn screen: encoder sharing, block LR and pooled_v.

This is the notebook cell from ``train_xattn.ipynb`` section 6, lifted out so the
grid can be launched as a plain script. The cell itself is left alone.

What is being compared
----------------------
Each tower gets its **own** encoder by default, so the two writing styles
(terse ``title | description`` vs. informal, often abusive comments) can
specialise in the blocks that training re-opens. The head input is
``pooled_c`` **only**, so cross-attention is the single channel from the video
tower.

===========================  =============================================
config                       what it is
===========================  =============================================
``sep_u2_l3``                the default: separate encoders, top 2 encoder
                             blocks unfrozen per tower, no pooled_v
``sep_u2_l3_pv``             default + ``--pooled-v``: appends
                             ``mean-pool(H_v)`` to the head (late fusion)
``sep_u2_l3_x1e3``           default + ``--xattn-lr 1e-3``: decoder blocks at
                             the head LR, i.e. the pre-fix behaviour. The
                             control for Risk 2.
``shared_u2_l3``             default + ``--shared-encoder``: one encoder for
                             both towers (the architecture behind every
                             earlier result)
``sep_u0_l3``                default + ``--unfreeze-layers 0``: both encoders
                             frozen
===========================  =============================================

Backbone rows
-------------
Same control otherwise (``--unfreeze-layers 2 --layers 3 --no-pooled-v``), only
the encoder changes, so a difference in a row is a difference in the encoder.
``sep_u2_l3`` **is** the MiniLM row (the default backbone is MiniLM), so the
comparison is three-way without a duplicate row:

===========================  =============================================
``e5_u2_l3``                 ``--backbone intfloat/multilingual-e5-small``
                             (12x384, XLM-R tokenizer, same shape as MiniLM)
``nano_u2_l3``               ``--backbone jinaai/jina-embeddings-v5-text-nano``
                             (12x768 EuroBERT, bidirectional, no dropout;
                             loaded natively - see ``model_xattn.load_encoder``
                             and ``src/probe_backbone.py``)
===========================  =============================================

The two new encoders are not interchangeable with MiniLM at the same settings in
one respect: unfreezing 2 of 12 blocks trains ~19M/66M parameters instead of
~14M (measured in ``probe_backbone.py``), on the same ~7.9k rows. The recipe
(``--backbone-lr 2e-5``, ``--head-lr 1e-3``, ``--epochs 15``) was tuned for the
14M case, so a loss in these rows is evidence about the recipe as much as about
the encoder.

Every row spells out its flags, including the pooled_v choice, so a later
default change cannot silently alter what an already-stored row means.

Reading the result
------------------
Defaults to **3 seeds (40, 41, 42)**, not the notebook's ``SEEDS = (42,)``.
A single seed is not evidence: the measured within-config sd is ~2.7pt while
the whole config spread is ~1.4pt, so a one-seed ranking is decided by noise.
``--seeds 42`` is there for a cheap smoke run only, and this script labels its
output as non-evidence when n=1.

Usage
-----
    python scripts/run_xattn_screen.py                          # full grid, 3 seeds
    python scripts/run_xattn_screen.py --configs shared_u2_l3,sep_u2_l3
    python scripts/run_xattn_screen.py --seeds 42               # smoke, not evidence
    python scripts/run_xattn_screen.py --dry-run                # print commands only
    python scripts/run_xattn_screen.py --configs sep_u0_l3      # the stale row, re-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

SEEDS_DEFAULT = (40, 41, 42)

BASE = ('--languages EN IT NL --epochs 15 --patience 5 --batch-size 32 '
        '--max-len 160 --video-max-len 256 --dropout 0.1 --weight-decay 0.01 --head-lr 1e-3')

CONFIGS = {
    'sep_u2_l3':      '--unfreeze-layers 2 --layers 3 --no-pooled-v',
    'sep_u2_l3_pv':   '--unfreeze-layers 2 --layers 3 --pooled-v',
    'sep_u2_l3_x1e3': '--unfreeze-layers 2 --layers 3 --no-pooled-v --xattn-lr 1e-3',
    'shared_u2_l3':   '--unfreeze-layers 2 --layers 3 --no-pooled-v --shared-encoder',
    'sep_u0_l3':      '--unfreeze-layers 0 --layers 3 --no-pooled-v',
    # Backbone rows: the control above with only the encoder swapped, so the
    # config name is also the output name and the two can never collide with the
    # MiniLM rows (the backbone is not in the output path).
    'e5_u2_l3':       ('--unfreeze-layers 2 --layers 3 --no-pooled-v '
                       '--backbone intfloat/multilingual-e5-small'),
    'nano_u2_l3':     ('--unfreeze-layers 2 --layers 3 --no-pooled-v '
                       '--backbone jinaai/jina-embeddings-v5-text-nano'),
}

# The comparison the screen exists to answer.
CONTROL = 'sep_u2_l3'
TEST = 'shared_u2_l3'

CSV_NAME = 'xattn_confirm.csv'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', default=','.join(str(s) for s in SEEDS_DEFAULT),
                   help='comma-separated split/train seeds (default: 40,41,42)')
    p.add_argument('--configs', default=None,
                   help='comma-separated subset of CONFIGS (default: all)')
    p.add_argument('--weights-dir', default='weights', help='output directory')
    p.add_argument('--dry-run', action='store_true', help='print the commands and exit')
    p.add_argument('--rerun', action='store_true',
                   help='re-run a config even if its history file already exists')
    p.add_argument('--extra', default=None,
                   help='flags appended to every command, e.g. "--epochs 1" for a smoke run. '
                        'Results produced this way are for plumbing only.')
    return p.parse_args(argv)


def build_jobs(args):
    seeds = tuple(int(s) for s in args.seeds.split(',') if s.strip())
    names = list(CONFIGS) if not args.configs else [
        n.strip() for n in args.configs.split(',') if n.strip()]

    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f'unknown config(s): {unknown}; known: {list(CONFIGS)}')

    jobs = []
    for name in names:
        for s in seeds:
            out = os.path.join(args.weights_dir, f'xattn_{name}_s{s}.pt')
            extra = f' {args.extra}' if args.extra else ''
            cmd = (f'python src/train_xattn.py --split-seed {s} --seed {s} '
                   f'{BASE} {CONFIGS[name]}{extra} --out {out}')
            jobs.append({'config': name, 'seed': s, 'out': out, 'cmd': cmd})
    return seeds, names, jobs


def best_score(out):
    """Peak val_macro_f1 for one run, read from its history file."""
    history = os.path.splitext(out)[0] + '_history.csv'
    if not os.path.exists(history):
        raise SystemExit(f'no history file after the run: {history}')
    h = pd.read_csv(history)
    if 'val_macro_f1' not in h.columns:
        raise SystemExit(f'{history} has no val_macro_f1 column; columns={list(h.columns)}')
    return round(float(h['val_macro_f1'].max()), 4)


def report(df, seeds, names):
    print('\n' + '=' * 72)
    summary = (df.groupby('config')['val_macro_f1']
                 .agg(mean='mean', std='std', n='count')
                 .round(4)
                 .sort_values('mean', ascending=False))
    print(summary.to_string())
    print('\nper seed:')
    print(df.pivot(index='config', columns='seed', values='val_macro_f1').to_string())

    if len(seeds) < 2:
        print('\n!! n=1 - NOT EVIDENCE. The within-config spread is ~2.7pt and the')
        print('!! config spread is ~1.4pt, so this ranking is decided by noise.')
        print('!! Re-run with the default seeds before quoting any number.')
        return

    floor = float(df.groupby('config')['val_macro_f1'].std().max() * 100)
    print(f'\nnoise floor (largest within-config sd): {floor:.2f}pt')

    pivot = df.pivot(index='seed', columns='config', values='val_macro_f1')
    if CONTROL in pivot.columns and TEST in pivot.columns:
        paired = (pivot[TEST] - pivot[CONTROL]) * 100
        print(f'{TEST} - {CONTROL} (paired, pt): '
              f'{", ".join(f"s{int(s)} {d:+.2f}" for s, d in paired.items())}')
        delta = float(paired.mean())
        verdict = ('INSIDE THE NOISE FLOOR - not a result' if abs(delta) < floor
                   else f'above the noise floor - worth quoting, but still n={len(paired)}')
        print(f'mean delta {delta:+.2f}pt over n={len(paired)} -> {verdict}')

    best = summary.index[0]
    print(f'\nbest config by mean: {best} ({summary.loc[best, "mean"]:.4f})')
    if abs(float(summary["mean"].max()) - float(summary["mean"].min())) * 100 < floor:
        print('BUT the config spread is smaller than the noise floor: the screen is flat,')
        print('and no config in it can be called the winner.')


def main(argv=None):
    args = parse_args(argv)
    seeds, names, jobs = build_jobs(args)

    print(f'seeds={list(seeds)}  configs={names}  runs={len(jobs)}')
    print(f'BASE: {BASE}\n')
    if args.dry_run:
        for j in jobs:
            print('RUN:', j['cmd'])
        return 0

    os.makedirs(args.weights_dir, exist_ok=True)
    rows = []
    for j in jobs:
        history = os.path.splitext(j['out'])[0] + '_history.csv'
        if os.path.exists(history) and not args.rerun:
            print(f'-- exists, skipping: {history}')
            rows.append({'config': j['config'], 'seed': j['seed'],
                         'val_macro_f1': best_score(j['out'])})
            continue
        print('RUN:', j['cmd'])
        try:
            subprocess.run(j['cmd'], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f'{j["config"]} seed {j["seed"]} failed '
                             f'(exit {e.returncode}); results before this point are in '
                             f'{os.path.join(args.weights_dir, CSV_NAME)}')
        rows.append({'config': j['config'], 'seed': j['seed'],
                     'val_macro_f1': best_score(j['out'])})

    df = pd.DataFrame(rows)
    path = os.path.join(args.weights_dir, CSV_NAME)
    df.to_csv(path, index=False)
    print(f'\nwrote {path}')
    report(df, seeds, names)
    return 0


if __name__ == '__main__':
    sys.exit(main())
