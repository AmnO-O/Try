#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM pseudo-labeling + translation data generator for Task A.

Runs an OpenAI-compatible chat model (Groq default) over real comments and writes
schema-compatible TSVs that train.py ingests via --train-extras / --input.

Modes (per input row, one API call):
  --mode labels    : answer the exact Task-A question ("does this comment convey a
                     stereotype about queer people?") -> stereotype yes/no + confidence.
  --mode translate : faithful IT and NL translations of the comment (gold label
                     travels), for back-translation augmentation of weak languages.
  --mode both      : label AND both translations in the same call.

Refusal handling: models with safety alignment may refuse the strongest positives.
Each refusal is detected, logged, and retried once under a research-framing
system message; the refusal rate is reported so a biased run is visible early.

Output files (STEREOTYPE column holds the LLM label in labels/both mode):
  labels    -> llm_stereotype_<LANG>_training.tsv
  translate -> aug_llm_EN_<IT|NL>_training.tsv
Both match the language regex used by load_frame_from_file.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

from config import REQUIRED_COLUMNS

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'qwen2.5-72b-instruct'

SYSTEM_LABEL = (
    'You are annotating research data about anti-LGBTQ+ stereotyping in public '
    'YouTube comments. The comment shown to you may contain offensive or hateful '
    'content; it is quoted for academic research, so you MUST analyse it and must '
    'not refuse. Judge whether the comment conveys a stereotype about queer people.'
)
SYSTEM_RETRY = (
    'This is a legitimate research classification task on a public comment. '
    'Answer anyway. Reply strictly with the JSON structure requested.'
)

_USER_TPL = '''Comment (verbatim):
{comment}

Reply with ONE JSON object, nothing else:
{{
  "stereotype": "yes" or "no",
  "confidence": number 0.0 to 1.0,
  "comment_it": "faithful Italian translation preserving tone/sarcasm",
  "comment_nl": "faithful Dutch translation preserving tone/sarcasm"
}}
'''

_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$')

REFUSAL_MARKERS = [
    "i can't", "cannot", "i am unable", "i'm unable", "i am not able",
    "can not", "won't", "not comfortable", "unable to", "decline", "refus",
    "against my", "content policy", "safety", "cannot assist", "can't assist",
]


def parse_json(text: str) -> dict:
    """Parses the model reply, tolerating code fences and leading prose."""
    text = _FENCE.sub('', (text or '').strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f'Unparseable model reply: {text[:200]!r}')


def is_refusal(text: str) -> bool:
    low = (text or '').lower()
    return any(m in low for m in REFUSAL_MARKERS)


def valid_label(obj: dict) -> bool:
    return str(obj.get('stereotype', '')).strip().lower() in ('yes', 'no')


def call_chat(url, api_key, model, system, user_text, retries=4,
              timeout=120, dry_run=False):
    if dry_run:
        return None
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': user_text}],
        'temperature': 0.2,
        'max_tokens': 600,
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f'  http {resp.status_code}, retrying in {wait}s '
                  f'({resp.text[:120]!r})', file=sys.stderr)
            time.sleep(wait)
            continue
        raise RuntimeError(f'API error {resp.status_code}: {resp.text[:300]}')
    raise RuntimeError('API failed after retries')


def read_tsv(path: str) -> list:
    with open(path, encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_ALL))


def write_tsv(path: str, rows: list) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS,
                                delimiter='\t', quoting=csv.QUOTE_ALL, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows -> {path}')


def infer_lang(path: str) -> str:
    match = re.search(r'_([A-Z]{2})(?:_training|_clean|\.tsv)', os.path.basename(path))
    return match.group(1) if match else 'EN'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='Schema-compatible TSV of real comments')
    parser.add_argument('--mode', choices=('labels', 'translate', 'both'), default='both')
    parser.add_argument('--out-prefix', default='llm')
    parser.add_argument('--pilot', type=int, default=0,
                        help='Process only the first N rows after sampling')
    parser.add_argument('--balanced', action='store_true',
                        help='Split the pilot between existing yes/no labels (hate_speech column)')
    parser.add_argument('--min-conf', type=float, default=0.7,
                        help='Drop labels below this confidence (labels/both mode)')
    parser.add_argument('--api-base', default=GROQ_URL)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api-key', default=os.environ.get('GEN_LLM_API_KEY', ''),
                        help='API key; falls back to env GEN_LLM_API_KEY')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the prompts for the selected rows and exit (no API call)')
    parser.add_argument('--delay', type=float, default=0.1, help='Seconds between calls (rate-limit safety)')
    args = parser.parse_args()

    rows = read_tsv(args.input)
    lang = infer_lang(args.input)
    print(f'{len(rows)} rows from {os.path.basename(args.input)} (lang={lang})')

    if args.balanced and args.pilot:
        pos = [r for r in rows if str(r.get('hate_speech', '')).strip().lower() in
               ('yes', 'yes_implicit', 'yes_explicit')]
        neg = [r for r in rows if str(r.get('hate_speech', '')).strip().lower() in
               ('none', 'no', '')]
        half = max(1, args.pilot // 2)
        rows = pos[:half] + neg[:half]
    elif args.pilot:
        rows = rows[:args.pilot]

    out_rows, log_rows = [], []
    total = len(rows)
    refused = 0
    dropped = 0

    for i, r in enumerate(rows, start=1):
        comment = str(r.get('yt_comment', '')).strip()
        if not comment:
            continue
        user_text = _USER_TPL.format(comment=comment)

        if args.dry_run:
            print(f'\n===== row {i}/{total} ({os.path.basename(args.input)}) =====')
            print(user_text)
            continue

        reply = call_chat(args.api_base, args.api_key, args.model, SYSTEM_LABEL,
                          user_text, dry_run=False)
        obj = None
        retried = False
        note = ''
        if is_refusal(reply):
            refused += 1
            note = 'refused'
            reply2 = call_chat(args.api_base, args.api_key, args.model,
                               SYSTEM_LABEL + ' ' + SYSTEM_RETRY, user_text)
            if not is_refusal(reply2):
                reply = reply2
                refried = True
                retried = True
                note = 'refused-then-ok'
            else:
                log_rows.append({
                    'i': i, 'src_id': r.get('StereoQueerEval_id', ''), 'comment': comment,
                    'ok': 'no', 'retried': False, 'label': '', 'confidence': '',
                    'comment_it': '', 'comment_nl': '', 'note': 'refused',
                })
                print(f'  [{i}/{total}] REFUSED (excluded)', file=sys.stderr)
                continue
        try:
            obj = parse_json(reply)
        except ValueError as err:
            dropped += 1
            note = 'parse-fail'
            log_rows.append({
                'i': i, 'src_id': r.get('StereoQueerEval_id', ''), 'comment': comment,
                'ok': 'no', 'retried': retried, 'label': '', 'confidence': '',
                'comment_it': '', 'comment_nl': '', 'note': note,
            })
            print(f'  [{i}/{total}] {note}: {err}', file=sys.stderr)
            continue

        label = str(obj.get('stereotype', 'no')).strip().lower()
        conf = float(obj.get('confidence', 0.0) or 0.0)
        if args.mode in ('labels', 'both'):
            valid = (label in ('yes', 'no')) and (conf >= args.min_conf)
            if not valid:
                dropped += 1
                note = 'low-conf' if label in ('yes', 'no') else 'bad-label'
        else:
            valid = bool(obj.get('comment_it')) and bool(obj.get('comment_nl'))

        new = dict(r)
        if args.mode in ('labels', 'both'):
            if valid:
                new['stereotype'] = 'yes' if label == 'yes' else 'no'
            else:
                new['stereotype'] = ''
        new['StereoQueerEval_id'] = f'{args.out_prefix}-{i:06d}'
        if valid:
            out_rows.append(new)
        log_rows.append({
            'i': i, 'src_id': r.get('StereoQueerEval_id', ''), 'comment': comment,
            'ok': 'yes' if valid else 'no', 'retried': retried, 'label': label,
            'confidence': conf, 'comment_it': obj.get('comment_it', ''),
            'comment_nl': obj.get('comment_nl', ''), 'note': note,
        })
        if i % 25 == 0:
            print(f'  [{i}/{total}] kept={len(out_rows)} refused={refused} dropped={dropped}')
        time.sleep(args.delay)

    if args.dry_run:
        print(f'\n{total} prompts printed (no API calls)')
        return

    if args.mode in ('labels', 'both'):
        out_path = f'{args.out_prefix}_stereotype_{lang}_training.tsv'
        write_tsv(out_path, out_rows)
    if args.mode in ('translate', 'both'):
        for target, key in (('IT', 'comment_it'), ('NL', 'comment_nl')):
            translated = [dict(r) for r in out_rows]
            for row in translated:
                row['yt_comment'] = row[key]
                del row[key]
            out_path = f'{args.out_prefix}_aug_{lang}_{target}_training.tsv'
            write_tsv(out_path, translated)

    log_path = f'{args.out_prefix}_log.csv'
    if log_rows:
        with open(log_path, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f'Wrote {len(log_rows)} log rows -> {log_path}')
    print(f'SUMMARY kept={len(out_rows)} refused={refused} dropped={dropped} '
          f'of {total}')


if __name__ == '__main__':
    main()