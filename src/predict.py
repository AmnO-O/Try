#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load a Task A checkpoint and write stereotype predictions to a CSV.

Usage:
    python src/predict.py --eval-only
    python src/predict.py --input LGBT/StereoQueerEval_NL_training.tsv --out nl_preds.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from config import (
    MAX_LEN,
    MMBERT_MODEL_NAME,
    READOUT_INIT,
    READOUT_INITS,
    READOUT_LAYERS,
    READOUT_HIDDEN,
    USE_MLM_HEAD,
    best_weights_path,
)
from model import load_model, load_model_weights, load_tokenizer
from train import collect, format_confusion_matrix, metrics_from, print_metrics, resolve_device

OUTPUT_COLUMNS = ['StereoQueerEval_id', 'lang', 'stereotype_pred', 'p_yes']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default=None,
                        help='Checkpoint (default: weights/best_mmbert_stereotype_prompt_<readout>.pt)')
    parser.add_argument('--model-name', default=MMBERT_MODEL_NAME)
    parser.add_argument('--input', default=None, help='TSV to score; defaults to all training languages')
    parser.add_argument('--label-column', default='stereotype',
                        help="Label column used for metrics: 'stereotype' or 'hate_speech'")
    parser.add_argument('--out', default='predictions.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--no-mlm-head', action='store_true')
    parser.add_argument('--readout-init', choices=READOUT_INITS, default=READOUT_INIT,
                        help="Must match the value used at training time")
    parser.add_argument('--readout-layers', type=int, default=READOUT_LAYERS,
                        help="Must match the value used at training time")
    parser.add_argument('--readout-hidden', type=int, default=READOUT_HIDDEN,
                        help="Must match the value used at training time")
    parser.add_argument('--device', default=None)
    parser.add_argument('--eval-only', action='store_true', help='Metrics only, do not write a CSV')

    args = parser.parse_args(argv)
    if args.weights is None:
        args.weights = best_weights_path(args.readout_init)
    return args


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.input:
        df = data_mod.load_frame_from_file(args.input, max_len=args.max_len,
                                           label_column=args.label_column)
    else:
        df = data_mod.load_frame(max_len=args.max_len, label_column=args.label_column)

    tokenizer = load_tokenizer(args.model_name)
    loader = data_mod.build_inference_loader(df, tokenizer, args.batch_size, args.max_len)

    model = load_model(
        args.model_name,
        unfreeze_layers=0,
        readout_init=args.readout_init,
        readout_layers=args.readout_layers,
        readout_hidden=args.readout_hidden,
        use_mlm_head=USE_MLM_HEAD and not args.no_mlm_head,
        tokenizer=tokenizer,
    )
    load_model_weights(model, args.weights, device=device)
    model.to(device)
    # predict() also calls this, but set it here so dropout is off before any
    # forward pass regardless of what runs next.
    model.eval()

    # A single pass gives both the predictions for the CSV and the metrics; the
    # criterion is None so metrics contains loss=nan.
    out = collect(model, loader, device)
    preds, probs, labels = out['preds'], out['probs'], out['labels']
    print_metrics(metrics_from(preds, labels, out['loss']), f'INPUT ({len(df)} rows)')
    print('--- CONFUSION MATRIX ---')
    print(format_confusion_matrix(labels, preds))

    if args.eval_only:
        return

    out = pd.DataFrame({
        'StereoQueerEval_id': df['StereoQueerEval_id'],
        'lang': df['lang'],
        'stereotype_pred': [data_mod.decode_stereotype(p) for p in preds],
        'p_yes': np.round(probs, 6),
    })[OUTPUT_COLUMNS]
    out.to_csv(args.out, index=False, encoding='utf-8')
    print(f'Wrote {len(out)} predictions to {args.out}')


if __name__ == '__main__':
    main()