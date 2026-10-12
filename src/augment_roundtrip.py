#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-trip (back-translation) augmentation for Task A.

Translates each training row into a pivot language and back into its own
language, producing a paraphrase that keeps the gold labels. Same language,
labels preserved -> the safest form of MT augmentation. Generation is
restricted to the *train* split (same val_fraction/split_seed as train.py),
so a paraphrase of a held-out video can never leak into training when the
output is fed via `train.py --train-extras`.

Fields: yt_comment + yt_title + yt_description (the three fields that
build_context / render_prompt consume). Pivots: IT->EN->IT, NL->EN->NL,
EN->DE->EN (DE is a neutral high-resource pivot, not a task language).

Caveat: round-tripping label=1 rows can smooth away sarcasm/slurs, leaving a
neutral paraphrase still tagged 'yes' (label noise, not drift). Use
--qc-label yes and --review-n to eyeball label-1 pairs before trusting a run,
and/or --min-overlap-yes / --skip-label yes to protect that class.

Needs the NLLB HF license accepted (gated repo; `huggingface-cli login`) and
`sentencepiece` (NLLB tokenizer).

Usage:
    python src/augment_roundtrip.py --limit 200 --device cpu --dry-run
    python src/augment_roundtrip.py --languages EN IT NL --split-seed 42 \\
        --min-token-overlap 0.2 --max-len-ratio 2.0 --min-overlap-yes 0.4 \\
        --keep-emoji --num-beams 4 --qc-label yes --review-n 20
"""

import argparse
import csv
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data as data_mod
from src.config import SPLIT_SEED, VAL_FRACTION
from src.tsv_io import write_tsv

MODEL_NAME = 'facebook/nllb-200-distilled-1.3B'
DATA_CODES = {'EN': 'eng_Latn', 'IT': 'ita_Latn', 'NL': 'nld_Latn'}
PIVOT = {'IT': 'EN', 'NL': 'EN', 'EN': 'DE'}
PIVOT_CODES = {'EN': 'eng_Latn', 'DE': 'deu_Latn'}
FIELDS = ('yt_comment', 'yt_title', 'yt_description')
MAX_WORDS = 512  # NLLB caps at 1024 positions; ~1.35 tok/word keeps this safe
QC_CHARS = 500
REVIEW_COLUMNS = ['StereoQueerEval_id', 'lang', 'stereotype', 'yt_comment',
                  'roundtrip', 'token_overlap', 'len_ratio']


def out_name(prefix: str, lang: str) -> str:
    """Filename ends in _<LANG>_training.tsv so train.py infers the language."""
    return f'{prefix}_{lang}_to_{lang}_training.tsv'


def batch_translate(tok, mdl, device, src_code, tgt_code, texts, batch_size, max_new_tokens,
                    num_beams=1, min_new_tokens=48):
    """Batched NLLB translate; returns stripped strings aligned with `texts`.

    The generation cap is adaptive: a translation is about as long as its
    source, so only ~1.6x the longest input in the batch is allowed (floored at
    --min-new-tokens, ceiling --max-new-tokens). A fixed 512 cap made every short
    comment decode 512 tokens, which is the bulk of the CPU wall time.

    num_beams>1 switches from greedy to beam search, which cuts the worst
    short-input hallucinations at ~num_beams x decode cost.
    """
    tok.src_lang = src_code
    bos = tok.convert_tokens_to_ids(tgt_code)
    out = []
    gen_kwargs = {}
    if num_beams and num_beams > 1:
        gen_kwargs = {'num_beams': num_beams, 'early_stopping': True}
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors='pt', padding=True,
                      truncation=True, max_length=MAX_WORDS).to(device)
            cap = min(max_new_tokens,
                      max(min_new_tokens, int(1.6 * enc['input_ids'].shape[1])))
            gen = mdl.generate(**enc, forced_bos_token_id=bos, max_new_tokens=cap,
                               **gen_kwargs)
            out.extend(t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True))
    return out


def fake_translate(texts):
    """Stand-in for --dry-run: drops every other token so the QC filters bite."""
    out = []
    for text in texts:
        toks = text.split()
        out.append(text if len(toks) <= 2 else ' '.join(toks[::2]))
    return out


def token_overlap(src: str, out: str) -> float:
    """Fraction of source tokens that survive the round trip (1.0 = untouched)."""
    a, b = set(str(src).lower().split()), set(str(out).lower().split())
    return len(a & b) / len(a) if a else 1.0


def length_ratio(src: str, out: str) -> float:
    a = len(str(src).split())
    return len(str(out).split()) / a if a else 1.0


# Emoji + their modifiers joiners (ZWJ, VS16, skin tones, regional flags). NLLB
# drops these, which both loses signal and deflates token_overlap (each emoji is
# a whitespace token, so a preserved-meaning row looks eroded). --keep-emoji
# scrapes them from the source and re-attaches them to the paraphrase.
EMOJI_RE = re.compile(
    '['
    '\U0001F000-\U0001FAFF'  # pictographs, symbols, supplemental, extended
    '\U00002600-\U000027BF'  # misc symbols + dingbats
    '\U0001F1E6-\U0001F1FF'  # regional indicators (flags)
    '\U0001F3FB-\U0001F3FF'  # skin-tone modifiers
    '\u200d\uFE0E\uFE0F'     # ZWJ, VS15, VS16
    ']+'
)


def extract_emoji(text: str) -> str:
    return ''.join(EMOJI_RE.findall(str(text)))


def restore_emoji(src: str, out: str) -> str:
    """Re-attach the source emoji to a paraphrase that lost them."""
    emojis = extract_emoji(src)
    return (str(out).rstrip() + ' ' + emojis).strip() if emojis else str(out)


def roundtrip(translate, src_code, pivot_code, texts):
    """src -> pivot -> src; only non-empty strings are sent to the model."""
    out = [''] * len(texts)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    if not idx:
        return out
    fwd = translate(src_code, pivot_code, [texts[i] for i in idx])
    bwd = translate(pivot_code, src_code, fwd)
    for j, i in enumerate(idx):
        out[i] = bwd[j]
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--languages', nargs='*', default=None,
                        help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--label-column', default='stereotype')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Ceiling on generated tokens per field; the effective cap is '
                             '~1.6x the longest input in the batch, never above this')
    parser.add_argument('--min-new-tokens', type=int, default=128,
                        help='Floor on the adaptive cap so short comments whose translation '
                             'expands are not clipped (was a hard-coded 48)')
    parser.add_argument('--num-beams', type=int, default=1,
                        help='Beam width for NLLB decoding (>1 cuts short-input hallucinations '
                             'at ~num_beams x decode cost; 1 = greedy)')
    parser.add_argument('--device', default=None, help='cuda | cuda:0 | cpu (default: auto)')
    parser.add_argument('--fields', choices=('comment', 'all'), default='all',
                        help='Round-trip all three fields or yt_comment only')
    parser.add_argument('--keep-emoji', action='store_true',
                        help='Re-attach source emoji (dropped by the MT) to each paraphrase; '
                             'keeps the expressive signal and stops emoji loss from deflating '
                             'token_overlap')
    parser.add_argument('--min-token-overlap', type=float, default=0.0,
                        help='Drop a row if the round-tripped comment keeps fewer than this '
                             'fraction of the source tokens (0 = keep all)')
    parser.add_argument('--min-overlap-yes', type=float, default=None,
                        help='Override --min-token-overlap for label=yes rows (guards '
                             'against smoothing the stereotype away)')
    parser.add_argument('--max-len-ratio', type=float, default=0.0,
                        help='Drop if round-trip/source token ratio falls outside [1/r, r] '
                             '(0 = keep all)')
    parser.add_argument('--skip-label', choices=('yes', 'no'), default=None,
                        help='Do not augment rows carrying this label')
    parser.add_argument('--limit', type=int, default=0, help='Cap rows per language (0 = all)')
    parser.add_argument('--review-n', type=int, default=0,
                        help='Write up to N pairs per language to aug_rt_review_<LANG>.tsv '
                             '(for manual/eval drift checks)')
    parser.add_argument('--review-label', choices=('all', 'yes', 'no'), default='all',
                        help='Restrict the review sample to one label (default: random over all)')
    parser.add_argument('--qc-label', choices=('all', 'yes', 'no'), default='all',
                        help='Which label to show in the printed QC samples')
    parser.add_argument('--qc-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out-prefix', default='aug_rt')
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip NLLB; use a token-dropping stand-in translator to exercise '
                             'the split/QC/write pipeline without the model.')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    df = data_mod.load_frame(languages=args.languages, label_column=args.label_column)
    df_train, df_val = data_mod.split_by_video(df, args.val_fraction, args.split_seed)
    print(f'Train {len(df_train)} | val {len(df_val)} (val-fraction {args.val_fraction}, '
          f'split-seed {args.split_seed})')
    for lang, n in sorted(df_train['lang'].value_counts().items()):
        print(f'  train {lang}: {n}')

    if args.dry_run:
        print('DRY RUN: NLLB skipped; using the token-dropping stand-in translator')

        def translate(src_code, tgt_code, texts):
            return fake_translate(texts)
    else:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f'Model: {args.model} | device: {device}')
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        ).to(device).eval()

        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        ).to(device).eval()
        if mdl.generation_config is not None:
            mdl.generation_config.max_length = None

        def translate(src_code, tgt_code, texts):
            return batch_translate(tok, mdl, device, src_code, tgt_code, texts,
                                   args.batch_size, args.max_new_tokens, args.num_beams,
                                   args.min_new_tokens)

    fields = FIELDS if args.fields == 'all' else ('yt_comment',)
    total_kept = 0
    for lang in sorted(df_train['lang'].unique()):
        if lang not in PIVOT:
            print(f'  skip {lang}: no pivot configured')
            continue
        pivot = PIVOT[lang]
        src_code, pivot_code = DATA_CODES[lang], PIVOT_CODES[pivot]

        sub = df_train[df_train['lang'] == lang]
        if args.skip_label:
            sub = sub[sub[args.label_column].str.strip().str.lower() != args.skip_label]
        rows = sub.to_dict('records')
        if args.limit and len(rows) > args.limit:
            rows = rng.sample(rows, args.limit)

        translated = {}
        for field in fields:
            texts = [' '.join(str(r.get(field, '')).split()[:MAX_WORDS]) for r in rows]
            out = roundtrip(translate, src_code, pivot_code, texts)
            if args.keep_emoji:
                out = [restore_emoji(s, h) for s, h in zip(texts, out)]
            translated[field] = out

        kept, review = [], []
        counts = {'yes': 0, 'no': 0, 'empty': 0, 'overlap': 0, 'ratio': 0}
        for i, r in enumerate(rows):
            label = str(r.get(args.label_column, '')).strip().lower()
            src_comment = str(r.get('yt_comment', ''))
            hyp_comment = translated['yt_comment'][i]
            if src_comment.strip() and not hyp_comment.strip():
                counts['empty'] += 1
                continue
            ov = token_overlap(src_comment, hyp_comment)
            lr = length_ratio(src_comment, hyp_comment)
            thr = args.min_overlap_yes if (label == 'yes' and args.min_overlap_yes is not None) \
                else args.min_token_overlap
            if ov < thr:
                counts['overlap'] += 1
                continue
            if args.max_len_ratio > 0 and not (1 / args.max_len_ratio <= lr <= args.max_len_ratio):
                counts['ratio'] += 1
                continue

            new = dict(r)
            for field in fields:
                new[field] = translated[field][i]
            new['StereoQueerEval_id'] = f"{args.out_prefix}-{lang}-{r.get('StereoQueerEval_id', '')}"
            kept.append(new)
            counts[label] = counts.get(label, 0) + 1
            review.append({'StereoQueerEval_id': r.get('StereoQueerEval_id', ''), 'lang': lang,
                           'stereotype': label, 'yt_comment': src_comment,
                           'roundtrip': hyp_comment, 'token_overlap': round(ov, 3),
                           'len_ratio': round(lr, 3)})

        out_path = out_name(args.out_prefix, lang)
        write_tsv(out_path, kept)
        print(f'{lang}->{pivot}->{lang}: kept {len(kept)}/{len(rows)} '
              f'(yes {counts["yes"]} / no {counts["no"]}) | dropped empty {counts["empty"]}, '
              f'overlap {counts["overlap"]}, ratio {counts["ratio"]} -> {out_path}')

        qc = [p for p in review if args.qc_label == 'all' or p['stereotype'] == args.qc_label]
        for p in rng.sample(qc, min(args.qc_samples, len(qc))):
            print(f'  QC [{p["stereotype"]}] overlap={p["token_overlap"]} len_ratio={p["len_ratio"]}')
            print(f'    src: {p["yt_comment"][:QC_CHARS]}')
            print(f'    rt : {p["roundtrip"][:QC_CHARS]}')

        if args.review_n and review:
            pool = [p for p in review
                    if args.review_label == 'all' or p['stereotype'] == args.review_label]
            picks = rng.sample(pool, min(args.review_n, len(pool)))
            rev_path = f'{args.out_prefix}_review_{lang}.tsv'
            with open(rev_path, 'w', encoding='utf-8', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, delimiter='\t',
                                        quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(picks)
            n_yes = sum(1 for p in picks if p['stereotype'] == 'yes')
            print(f'  review -> {rev_path} ({len(picks)} pairs, {n_yes} label=yes)')
        total_kept += len(kept)

    print(f'Done: {total_kept} augmented rows')


if __name__ == '__main__':
    main()
