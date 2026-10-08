#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline translation augmentation with NLLB-200 distilled 1.3B.

Translates comments per (src, tgt) pair driven by --plan. Gold labels
(stereotype/hate_speech/target) travel with the row; only yt_comment is
rewritten, same convention as gen_llm_data.py --mode translate. Outputs are
schema-compatible TSVs for train.py --train-extras, with the target language
in the filename.

Needs the model's HF license accepted (gated repo); `huggingface-cli login`.

Usage:
    python src/augment_nllb.py --n 500
    python src/augment_nllb.py --plan '{"EN>IT": {"label_filter": "yes", "n_samples": 600}, "EN>NL": {"label_filter": "yes", "n_samples": 250}}'
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data as data_mod
from src.gen_llm_data import infer_lang, read_tsv, write_tsv

MODEL_NAME = 'facebook/nllb-200-distilled-1.3B'
LANG_CODES = {'EN': 'eng_Latn', 'IT': 'ita_Latn', 'NL': 'nld_Latn'}
MAX_WORDS = 512  # NLLB caps at 1024 positions; ~1.35 tok/word keeps this safe
QC_SAMPLES = 5
QC_CHARS = 500


def out_name(prefix: str, src: str, tgt: str) -> str:
    return f'{prefix}_{src}_to_{tgt}_training.tsv'


def translate_texts(tok, mdl, device, src, tgt, texts, batch_size: int):
    """Batched NLLB translate; returns stripped strings aligned with `texts`."""
    tok.src_lang = LANG_CODES[src]
    bos = tok.convert_tokens_to_ids(LANG_CODES[tgt])
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors='pt', padding=True,
                      truncation=True, max_length=MAX_WORDS).to(device)
            gen = mdl.generate(**enc, forced_bos_token_id=bos, max_new_tokens=MAX_WORDS)
            out.extend(t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True))
    return out


def _parse_plan(plan_str, src_langs, default_n):
    """--plan JSON ('SRC>TGT' -> {label_filter, n_samples}) or the flat default."""
    if not plan_str:
        return {(s, t): {'label_filter': None, 'n_samples': default_n}
                for s in src_langs for t in LANG_CODES if t != s}
    try:
        raw = json.loads(plan_str)
    except json.JSONDecodeError as err:
        raise SystemExit(f'error: --plan is not valid JSON: {err}')
    plan = {}
    for key, spec in raw.items():
        src, sep, tgt = str(key).partition('>')
        filt = spec.get('label_filter')
        n = spec.get('n_samples', default_n)
        if not sep or src not in LANG_CODES or tgt not in LANG_CODES or src == tgt:
            raise SystemExit(f'error: bad plan pair {key!r} (want "SRC>TGT" from EN/IT/NL)')
        if filt not in (None, 'yes', 'no'):
            raise SystemExit(f'error: bad label_filter {filt!r} for {key} (want yes/no/null)')
        if not isinstance(n, int) or n < 0:
            raise SystemExit(f'error: bad n_samples {n!r} for {key} (want int >= 0)')
        plan[(src, tgt)] = {'label_filter': filt, 'n_samples': n}
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', nargs='*', default=None,
                        help='Training TSVs (default: all StereoQueerEval_*_training.tsv)')
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--n', type=int, default=500, help='Samples per pair when --plan is omitted')
    parser.add_argument('--plan', default=None,
                        help='JSON plan: {"EN>IT": {"label_filter": "yes", "n_samples": 600}, ...}; '
                             'label_filter null samples all rows')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default=None, help='cuda | cuda:0 | cpu (default: auto)')
    parser.add_argument('--out-prefix', default='aug_nllb')
    return parser.parse_args(argv)


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    rows_by_lang = {}
    for path in args.inputs or data_mod.find_data_files():
        rows_by_lang[infer_lang(path)] = [
            r for r in read_tsv(path) if str(r.get('yt_comment', '')).strip()]
    plan = _parse_plan(args.plan, rows_by_lang, args.n)
    print(f'Model: {args.model} | device: {device} | pairs: {len(plan)}')

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    ).to(device).eval()

    total = 0
    for (src, tgt), spec in plan.items():
        filt = spec['label_filter']
        elig = [r for r in rows_by_lang.get(src, [])
                if filt is None or str(r.get('stereotype', '')).strip().lower() == filt]
        kept = rng.sample(elig, min(spec['n_samples'], len(elig)))
        texts = [' '.join(str(r['yt_comment']).split()[:MAX_WORDS]) for r in kept]
        out_rows, pairs, dropped = [], [], 0
        for r, text, hyp in zip(kept, texts, translate_texts(tok, mdl, device, src, tgt, texts, args.batch_size)):
            if not hyp:
                dropped += 1
                continue
            new = dict(r)
            new['yt_comment'] = hyp
            new['StereoQueerEval_id'] = f"{args.out_prefix}-{src}{tgt}-{r.get('StereoQueerEval_id', '')}"
            out_rows.append(new)
            pairs.append((text, hyp))
        for o, h in rng.sample(pairs, min(QC_SAMPLES, len(pairs))):
            print(f'  QC SRC [{src}]: {o[:QC_CHARS]}')
            print(f'  QC TGT [{tgt}]: {h[:QC_CHARS]}')
        out_path = out_name(args.out_prefix, src, tgt)
        write_tsv(out_path, out_rows)
        print(f'{src}->{tgt}: {len(out_rows)}/{len(elig)} eligible'
              f'{" (+%d empty dropped)" % dropped if dropped else ""} -> {out_path}')
        total += len(out_rows)
    print(f'Done: {total} augmented rows')


if __name__ == '__main__':
    main()
