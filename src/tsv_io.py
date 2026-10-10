#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared TSV readers/writers and filename language inference for Task A.

Schema-compatible with `train.py`: tab-separated, QUOTE_ALL quoting
(embedded newlines are quoted), columns REQUIRED_COLUMNS from config.
"""

import csv
import os
import re

from src.config import REQUIRED_COLUMNS


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