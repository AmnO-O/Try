#!/usr/bin/env python3
"""Round-trip (back-translation) augmentation for Task A.

Translates each training row into a pivot language and back into its own
language, producing a paraphrase that keeps the gold labels. Same language,
labels preserved -> the safest form of MT augmentation. Generation is
restricted to the *train* split (same val_fraction/split_seed as train.py),
so a paraphrase of a held-out video can never leak into training when the
output is fed via `train.py --train-extras`.

Fields: yt_comment by default (--fields all also round-trips yt_title and
yt_description, gating on the worst-case overlap/ratio/cosine across them).
Pivots: IT->EN->IT, NL->EN->NL, EN->DE->EN (DE is a neutral high-resource
pivot, not a task language).

Caveat: round-tripping label=1 rows can smooth away sarcasm/slurs, leaving a
neutral paraphrase still tagged 'yes' (label noise, not drift). Use
--qc-label yes and --review-n to eyeball label-1 pairs before trusting a run,
and/or --min-overlap-yes / --skip-label yes to protect that class.

Needs `sentencepiece` (NLLB tokenizer) and `transformers`; the NLLB-200
distilled checkpoint downloads without a gated license.

Usage:
    python src/augment_roundtrip.py --limit 200 --device cpu --dry-run
    python src/augment_roundtrip.py --languages EN IT NL --split-seed 42 \\
        --fields comment --min-cosine 0.8 --max-len-ratio 2.0 --keep-emoji \\
        --num-beams 4 --qc-label yes --review-n 20
"""

import argparse
import csv
import os
import random
import re
import sys

import pandas as pd
import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data as data_mod
from src.config import SPLIT_SEED, VAL_FRACTION
from src.tsv_io import write_tsv

MODEL_NAME = 'facebook/nllb-200-distilled-1.3B'
DATA_CODES = {'EN': 'eng_Latn', 'IT': 'ita_Latn', 'NL': 'nld_Latn'}
PIVOT = {'IT': 'EN', 'NL': 'EN', 'EN': 'DE'}
PIVOT_CODES = {'EN': 'eng_Latn', 'DE': 'deu_Latn'}
FIELDS = ('yt_comment', 'yt_title', 'yt_description')
TOKEN_LIMIT = 512    # NLLB input limit, in tokens (not words)
MAX_SEG_WORDS = 200  # words per translation chunk (~270 tokens, safely under TOKEN_LIMIT)
QC_CHARS = 500
REVIEW_COLUMNS = ['StereoQueerEval_id', 'lang', 'stereotype', 'yt_comment',
                  'roundtrip', 'token_overlap', 'len_ratio', 'cosine']


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
                      truncation=True, max_length=TOKEN_LIMIT).to(device)
            if enc['input_ids'].shape[1] >= TOKEN_LIMIT:
                print(f'  warning: an input hit the {TOKEN_LIMIT}-token limit; '
                      f'increase MAX_SEG_WORDS chunking if this is common')
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


def clean_field(value) -> str:
    """Normalize a raw TSV cell to a stripped string.

    Missing values (None / NaN / pd.NA) become '' rather than the literal
    'nan' string that `str(value)` would produce; this keeps NaN comments out
    of both the translation inputs and the QC evaluation.
    """
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value).strip()


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


# Sentence boundary = . ! ? followed by whitespace (so URLs/decimals survive).
_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def segment_sentences(text: str):
    """Split a comment into sentences. NLLB drops trailing clauses on multi-
    clause inputs, so translating each sentence separately keeps them all."""
    parts = [p.strip() for p in _SENT_RE.split(str(text))]
    return [p for p in parts if p]


def chunk_words(text: str, max_words: int = MAX_SEG_WORDS):
    """Split an over-long segment into word chunks that fit the model.

    Keeps *all* words (unlike slicing), so a field is never silently shortened
    and then presented as a paraphrase of the whole source.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def roundtrip(translate, src_code, pivot_code, texts, split_sentences=True):
    """src -> pivot -> src; only non-empty strings are sent to the model.

    With split_sentences, each sentence is round-tripped on its own and the
    results are rejoined, which prevents the model from dropping later clauses
    of a long comment (measured: overlap 0.214 -> 0.5 on a 3-clause example).
    Segments longer than MAX_SEG_WORDS are chunked first so nothing is truncated.
    """
    out = [''] * len(texts)
    flat, per_row = [], [[] for _ in texts]
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        segs = segment_sentences(t) if split_sentences else [t.strip()]
        for s in segs:
            for chunk in chunk_words(s):
                per_row[i].append(len(flat))
                flat.append(chunk)
    if not flat:
        return out
    fwd = translate(src_code, pivot_code, flat)
    bwd = translate(pivot_code, src_code, fwd)
    for i, idxs in enumerate(per_row):
        out[i] = ' '.join(bwd[j] for j in idxs).strip()
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
    parser.add_argument('--fields', choices=('comment', 'all'), default='comment',
                        help='Round-trip yt_comment only (default; QC and review focus on the '
                             'comment) or all three fields. With "all" the keep/drop gate uses '
                             'the worst overlap/ratio/cosine across the fields, but only the '
                             'comment is shown in QC/review, so inspect title/description too')
    parser.add_argument('--keep-emoji', action='store_true',
                        help='Re-attach source emoji (dropped by the MT) to each paraphrase; '
                             'keeps the expressive signal and stops emoji loss from deflating '
                             'token_overlap')
    parser.add_argument('--no-split-sentences', action='store_true',
                        help='Round-trip each whole field at once instead of sentence by '
                             'sentence (splitting is on by default; it stops NLLB dropping '
                             'trailing clauses of multi-clause comments)')
    parser.add_argument('--min-token-overlap', type=float, default=0.0,
                        help='Drop a row if the round-tripped comment keeps fewer than this '
                             'fraction of the source tokens (0 = keep all)')
    parser.add_argument('--min-overlap-yes', type=float, default=None,
                        help='Override --min-token-overlap for label=yes rows (guards '
                             'against smoothing the stereotype away)')
    parser.add_argument('--min-cosine', type=float, default=0.0,
                        help='Semantic gate: drop a row when the embedding cosine between the '
                             'source comment and its round-trip is below this (0 = off). When '
                             'set it REPLACES the token-overlap thresholds: keeps synonym / '
                             'slur-swap paraphrases (low overlap, high cosine) and drops subtle '
                             'smoothing (high overlap, low cosine)')
    parser.add_argument('--cosine-model', default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Sentence-embedding model used by the --min-cosine gate')
    parser.add_argument('--max-len-ratio', type=float, default=0.0,
                        help='Drop if round-trip/source token ratio falls outside [1/r, r] '
                             '(0 = keep all; otherwise must be >= 1)')
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
    args = parser.parse_args(argv)

    if args.max_len_ratio != 0 and args.max_len_ratio < 1:
        parser.error('--max-len-ratio must be 0 (off) or >= 1 '
                     '(values in (0, 1) make the [1/r, r] window unsatisfiable)')
    if args.batch_size <= 0:
        parser.error('--batch-size must be > 0')
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    if args.review_n < 0 or args.qc_samples < 0:
        parser.error('--review-n and --qc-samples must be >= 0')
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be > 0')
    if args.num_beams < 1:
        parser.error('--num-beams must be >= 1')
    for name in ('min_token_overlap', 'min_overlap_yes', 'min_cosine'):
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            parser.error(f'--{name.replace("_", "-")} must be in [0, 1]')
    return args


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    df = data_mod.load_frame(languages=args.languages, label_column=args.label_column)
    labels = df[args.label_column].astype('string').str.strip().str.lower()
    bad = labels[~labels.isin(('yes', 'no'))]
    if len(bad):
        raise SystemExit(
            f'error: {len(bad)}/{len(df)} rows have a missing or non-binary value in '
            f'{args.label_column!r} (e.g. {bad.dropna().unique()[:5].tolist() or ["<missing>"]}); '
            f'expected yes/no. A missing label must not survive into the augmented output.')
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
        if mdl.generation_config is not None:
            mdl.generation_config.max_length = None

        def translate(src_code, tgt_code, texts):
            return batch_translate(tok, mdl, device, src_code, tgt_code, texts,
                                   args.batch_size, args.max_new_tokens, args.num_beams,
                                   args.min_new_tokens)

    cos_model = None
    if args.min_cosine > 0:
        from sentence_transformers import SentenceTransformer
        cos_model = SentenceTransformer(args.cosine_model, device=str(device))
        print(f'Semantic gate: cosine >= {args.min_cosine} ({args.cosine_model}); '
              f'token-overlap thresholds ignored for the keep/drop decision')

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

        n = len(rows)
        src_field = {f: [clean_field(r.get(f, '')) for r in rows] for f in fields}

        translated = {}
        for field in fields:
            out = roundtrip(translate, src_code, pivot_code, src_field[field],
                            split_sentences=not args.no_split_sentences)
            if args.keep_emoji:
                out = [restore_emoji(s, h) for s, h in zip(src_field[field], out)]
            translated[field] = out

        cos_field = {f: [None] * n for f in fields}
        if cos_model is not None:
            for field in fields:
                idx = [i for i in range(n) if src_field[field][i] and translated[field][i]]
                if idx:
                    ea = cos_model.encode([src_field[field][i] for i in idx],
                                          normalize_embeddings=True)
                    eb = cos_model.encode([translated[field][i] for i in idx],
                                          normalize_embeddings=True)
                    for j, i in enumerate(idx):
                        cos_field[field][i] = float((ea[j] * eb[j]).sum())

        kept, review = [], []
        counts = {'yes': 0, 'no': 0, 'empty': 0, 'overlap': 0, 'cosine': 0, 'ratio': 0}
        for i, r in enumerate(rows):
            label = clean_field(r.get(args.label_column, '')).lower()
            src_comment = src_field['yt_comment'][i]
            hyp_comment = translated['yt_comment'][i]
            if not src_comment:
                counts['empty'] += 1
                continue

            # Gate on the worst case across every augmented field so an empty
            # or drifted title/description cannot ride along with --fields all.
            ov, lr, cos = 1.0, 1.0, None
            for field in fields:
                src, hyp = src_field[field][i], translated[field][i]
                if not src:
                    continue
                ov = min(ov, token_overlap(src, hyp))
                fr = length_ratio(src, hyp)
                if abs(fr - 1) > abs(lr - 1):
                    lr = fr
                cf = cos_field[field][i]
                if cf is not None:
                    cos = cf if cos is None else min(cos, cf)

            if cos is not None:
                if cos < args.min_cosine:
                    counts['cosine'] += 1
                    continue
            else:
                thr = args.min_overlap_yes if (
                    label == 'yes' and args.min_overlap_yes is not None) else args.min_token_overlap
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
                           'len_ratio': round(lr, 3),
                           'cosine': round(cos, 3) if cos is not None else ''})

        out_path = out_name(args.out_prefix, lang)
        write_tsv(out_path, kept)
        print(f'{lang}->{pivot}->{lang}: kept {len(kept)}/{len(rows)} '
              f'(yes {counts["yes"]} / no {counts["no"]}) | dropped empty {counts["empty"]}, '
              f'cosine {counts["cosine"]}, overlap {counts["overlap"]}, ratio {counts["ratio"]} '
              f'-> {out_path}')

        qc = [p for p in review if args.qc_label == 'all' or p['stereotype'] == args.qc_label]
        for p in rng.sample(qc, min(args.qc_samples, len(qc))):
            print(f'  QC [{p["stereotype"]}] overlap={p["token_overlap"]} '
                  f'cosine={p.get("cosine", "")} len_ratio={p["len_ratio"]}')
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
