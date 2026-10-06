#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data loading and torch Dataset for Task A (stereotype detection).

Every input is rendered into an MLM prompt that ends in a [MASK] slot, so the
model predicts the label word ('yes' / 'no') rather than reading a classifier
logit off an arbitrary projection.
"""

import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import (
    CLASS_TO_ID,
    DATA_DIR,
    DEFAULT_PROMPT,
    LABEL_CLASSES,
    MASK_LABEL,
    MAX_LEN,
    PROMPT_TEMPLATES,
    PromptBudget,
    REQUIRED_COLUMNS,
    SEP_TOKEN,
    TRAIN_GLOB,
)

# -----------------------------------------------------------------------------
# LABELS
# -----------------------------------------------------------------------------

def encode_stereotype(label: str) -> int:
    """'yes' -> 1, 'no' -> 0."""
    value = str(label).strip().lower()
    if value not in CLASS_TO_ID:
        raise ValueError(f"Unknown stereotype label: {label!r} (expected 'yes' or 'no')")
    return CLASS_TO_ID[value]


def decode_stereotype(class_id: int) -> str:
    """Class index -> 'yes' / 'no'."""
    return LABEL_CLASSES[int(class_id)]


# -----------------------------------------------------------------------------
# TEXT CLEANING
# -----------------------------------------------------------------------------

# Zero-width / bidi junk present in the corpus. U+200D (ZWJ) and U+FE0F (VS16)
# are preserved because emoji such as the rainbow flag depend on them.
_INVISIBLE_JUNK = re.compile('[\u200b\u2060\u00ad\ufeff\u200c\u202a-\u202e\u2066-\u2069]')

# Any spelling of the mask marker, in either case. A comment containing one of
# these would be tokenised as an extra mask token, and the label slot is located
# by searching for the first one, so the model would read the wrong position.
_MASK_LIKE = re.compile(r'<\s*mask\s*>|\[\s*mask\s*\]|__mask__', re.IGNORECASE)


def clean_text(text: str) -> str:
    """Lowercase, flatten newlines and strip invisible junk. Keeps diacritics.

    Mask markers are removed as well: the [MASK] slot is located by token id, so
    a literal marker inside a comment would be mistaken for the slot.
    """
    text = str(text).lower()
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = _INVISIBLE_JUNK.sub(' ', text)
    text = _MASK_LIKE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_context(comment: str, title: str, description: str) -> str:
    """Comment first (it carries the label), then the video title and description."""
    fields = [clean_text(comment), clean_text(title), clean_text(description)]
    return f' {SEP_TOKEN} '.join(f for f in fields if f)


def build_prompt(context: str, title: str, description: str, lang: str) -> str:
    """Renders the MLM prompt for one row.

    The video title and description go into {video} while {context} holds the
    comment alone, so the question always refers to the comment and not to the
    surrounding video metadata.
    """
    template = PROMPT_TEMPLATES.get(lang, DEFAULT_PROMPT)[0]
    video = f'{SEP_TOKEN} '.join(f for f in (clean_text(title), clean_text(description)) if f)
    return template.format(context=context, video=video, mask=MASK_LABEL)


def truncate_words(text: str, max_words: int) -> str:
    """Cuts a field to its first max_words words."""
    words = text.split()
    return ' '.join(words[:max_words]) if len(words) > max_words else text


def render_prompt(row, lang: str, budget: PromptBudget) -> str:
    """Builds a prompt that is guaranteed to fit the word budget.

    The [MASK] slot is the last thing in every prompt, so a prompt that overflows
    max_len loses it to truncation and the sample becomes unusable. Instead of
    measuring an 'overhead' and hoping the arithmetic works out, the fixed part of
    the prompt (template plus question) is measured directly and the remaining
    budget is handed out to the fields in priority order: comment first, then
    title, then description.
    """
    comment = clean_text(row['yt_comment'])
    title = clean_text(row['yt_title'])
    description = clean_text(row['yt_description'])

    fixed_words = len(build_prompt('', '', '', lang).split())
    allowance = max(1, budget.total_words - fixed_words)

    comment = truncate_words(comment, min(budget.comment_words, allowance))
    allowance -= len(comment.split())

    title = truncate_words(title, allowance)
    allowance -= len(title.split())

    description = truncate_words(description, allowance)
    return build_prompt(comment, title, description, lang)


# -----------------------------------------------------------------------------
# LOADING
# -----------------------------------------------------------------------------

def _read_tsv(path: str) -> pd.DataFrame:
    # QUOTE_ALL matches how the corpus quotes fields that contain newlines;
    # without it the embedded line breaks shift the columns.
    df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_ALL, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[REQUIRED_COLUMNS].copy()


def _add_features(df: pd.DataFrame, lang: str, max_len: int = MAX_LEN) -> pd.DataFrame:
    budget = PromptBudget(max_len=max_len)
    df['lang'] = lang
    df['context'] = [
        build_context(c, t, d)
        for c, t, d in zip(df['yt_comment'], df['yt_title'], df['yt_description'])
    ]
    df['prompt'] = [render_prompt(row, lang, budget) for _, row in df.iterrows()]
    df['st_y'] = [encode_stereotype(s) for s in df['stereotype']]
    return df


def find_data_files(data_dir: str = DATA_DIR) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, TRAIN_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"No training files matching '{TRAIN_GLOB}' under {data_dir}. "
            "Set LGBT_DATA_DIR to the folder holding the StereoQueerEval TSVs."
        )
    return paths


def load_frame(
    data_dir: str = DATA_DIR,
    languages: Optional[List[str]] = None,
    max_len: int = MAX_LEN,
) -> pd.DataFrame:
    """Reads every language TSV into one frame with prompts and encoded labels."""
    frames = []
    for path in find_data_files(data_dir):
        match = re.search(r'_([A-Z]{2})_training\.tsv$', os.path.basename(path))
        if not match:
            continue
        lang = match.group(1)
        if languages and lang not in languages:
            continue
        frames.append(_add_features(_read_tsv(path), lang, max_len))

    if not frames:
        raise ValueError('No language files matched the requested languages.')
    return pd.concat(frames, ignore_index=True)


def load_frame_from_file(path: str, max_len: int = MAX_LEN) -> pd.DataFrame:
    """Same encoding as load_frame but for a single explicit TSV path."""
    match = re.search(r'_([A-Z]{2})(?:_training|_clean)?\.tsv$', os.path.basename(path))
    return _add_features(_read_tsv(path), match.group(1) if match else 'EN', max_len)


def split_by_video(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Video-level split: all comments under one video stay on the same side."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df['yt_title']))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(df_train['yt_title']) & set(df_val['yt_title'])
    if overlap:
        raise RuntimeError(f"Video leakage: {len(overlap)} videos in both splits")
    return df_train, df_val


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class StereoQueerDataset(Dataset):
    """Tokenizes once at init and caches ids, mask and the [MASK] slot position.

    Sequences are stored un-padded and padded per batch, so a batch only pays for
    its own longest prompt. On this corpus that roughly halves the token count
    versus padding everything to max_len.
    """

    def __init__(self, prompts, labels, tokenizer, max_len: int = MAX_LEN):
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError(f"Tokenizer for this model has no mask token (needed for {MASK_LABEL})")

        prompts = [
            str(p).replace(SEP_TOKEN, tokenizer.sep_token).replace(MASK_LABEL, tokenizer.mask_token)
            for p in prompts
        ]

        # Tokenize without truncation first. The word-based budget is only an
        # estimate: the real tokenizer can still produce more tokens than max_len
        # for a small minority of prompts. Right-truncation would cut the tail of
        # the sequence, which is exactly where the slot lives, so such prompts are
        # left-truncated instead — a prompt over budget keeps its last max_len
        # tokens, i.e. the question, the slot, and as much of the comment as fits.
        full = tokenizer(prompts, truncation=False, padding=False)
        self.input_ids = [
            ids[-max_len:] if len(ids) > max_len else ids
            for ids in full['input_ids']
        ]
        self.attention_mask = [
            [1] * len(ids)
            for ids in self.input_ids
        ]

        # The prompt must contain exactly one slot. Left-truncation keeps the
        # mask (it sits at the end), so a wrong count here means a stray marker
        # slipped through cleaning, or a tokenizer that moved the mask — fail
        # loudly rather than read the wrong position.
        self.mask_positions = []
        for i, ids in enumerate(self.input_ids):
            occurrences = [j for j, token in enumerate(ids) if token == mask_token_id]
            if len(occurrences) != 1:
                raise ValueError(
                    f'Prompt {i} has {len(occurrences)} {MASK_LABEL} tokens; expected exactly 1 '
                    f'(max_len={max_len}). A literal mask marker inside the comment, or an '
                    'unusual tokenizer, can break the single-slot invariant.'
                )
            self.mask_positions.append(occurrences[0])
        self.labels = torch.as_tensor(np.array(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
            self.mask_positions[idx],
        )


def collate(batch, pad_token_id: int):
    """Pads a batch to its own longest sequence."""
    longest = max(len(ids) for ids, _, _, _ in batch)
    input_ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, (ids, mask, _, _) in enumerate(batch):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = torch.tensor(mask, dtype=torch.long)
    labels = torch.stack([label for _, _, label, _ in batch])
    mask_positions = torch.tensor([pos for _, _, _, pos in batch], dtype=torch.long)
    return input_ids, attention_mask, labels, mask_positions


class LengthGroupedSampler(Sampler):
    """Shuffles, then sorts inside mega-batches so similar lengths sit together.

    Keeps the ordering random across epochs while cutting padding waste; the
    shuffle comes from a generator seeded per epoch, so runs stay reproducible.
    """

    def __init__(self, lengths, batch_size: int, mega_batch: int = 50, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.mega_batch = mega_batch * batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        indices = generator.permutation(len(self.lengths))
        ordered = []
        for start in range(0, len(indices), self.mega_batch):
            block = indices[start:start + self.mega_batch]
            block = block[np.argsort(self.lengths[block], kind='stable')]
            ordered.extend(block)
        return iter(ordered)


def build_inference_loader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    max_len: int = MAX_LEN,
) -> DataLoader:
    """Single sequential loader for scoring.

    build_loaders(df, df, ...) would tokenize the frame twice and build a
    training sampler that is then thrown away, so inference uses its own path:
    one dataset, no shuffle, no length grouping.
    """
    dataset = StereoQueerDataset(
        df['prompt'].values, df['st_y'].values, tokenizer, max_len=max_len
    )
    pad_token_id = tokenizer.pad_token_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_token_id),
    )


def build_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    max_len: int = MAX_LEN,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = StereoQueerDataset(
        df_train['prompt'].values, df_train['st_y'].values, tokenizer, max_len=max_len
    )
    val_dataset = StereoQueerDataset(
        df_val['prompt'].values, df_val['st_y'].values, tokenizer, max_len=max_len
    )

    pad_token_id = tokenizer.pad_token_id
    sampler = LengthGroupedSampler(
        [len(ids) for ids in train_dataset.input_ids], batch_size, seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate(batch, pad_token_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_token_id),
    )
    return train_loader, val_loader


def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Count of 'yes' vs 'no', handy as a sanity check before training."""
    return df['stereotype'].value_counts().to_dict()