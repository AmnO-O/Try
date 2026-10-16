#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the Seq2Seq cross-attention classifier for Task A (stereotype yes/no).

Feeds the video context (title + description) through a pretrained encoder and
the YouTube comment through the pretrained decoder, which cross-attends to the
encoder memory.  The hidden state at the comment's ``</s>`` token (or its
mean-pooled alternative) drives a small classification head.

See ``model_seq2seq.py`` for the full architecture description.

Conventions shared with the existing trainers
---------------------------------------------
- Data loading, video-level split, and label encoding re-use ``data.py``
  utilities exactly as ``train_xattn.py`` does.
- Metric names, confusion matrix formatting and ``print_metrics`` are imported
  from ``train.py``.
- ``--val-fraction`` / ``--split-seed`` carry the same meaning; identical seeds
  produce identical train / val splits across all three trainers.

Typical usage
-------------
    # Quick smoke-test on English with facebook/bart-base (English only, fast):
    python src/train_seq2seq.py --backbone facebook/bart-base --languages EN --epochs 3 --batch-size 4

    # mBART-50, all three languages, freeze encoder entirely, fine-tune top 2 decoder layers:
    python src/train_seq2seq.py --backbone facebook/mbart-large-50 --languages EN IT NL \
        --unfreeze-enc 0 --unfreeze-dec 2 --batch-size 4 --grad-accum 4 --epochs 20

    # NLLB-200 distilled 600M, EOS pooling:
    python src/train_seq2seq.py --backbone facebook/nllb-200-distilled-600M \
        --unfreeze-dec 2 --pooling eos --batch-size 4 --grad-accum 4

    # Mean pooling instead of EOS:
    python src/train_seq2seq.py --pooling mean --unfreeze-dec 2

    # Evaluate an existing checkpoint:
    python src/train_seq2seq.py --evaluate-only --out weights/best_seq2seq.pt
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
import model_seq2seq
from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    SPLIT_SEED,
    VAL_FRACTION,
    WEIGHTS_DIR,
)
from train import (
    autocast_ctx,
    format_confusion_matrix,
    make_scaler,
    metrics_from,
    print_metrics,
    resolve_device,
    set_seed,
)

# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------

class Seq2SeqDataset(Dataset):
    """Tokenize the context and comment into two separate sequences.

    Language tags
    -------------
    mBART-50 and NLLB-200 use language-identification tokens prepended to every
    sequence.  The tokenizer handles this automatically when ``src_lang`` and
    ``tgt_lang`` are set *before* calling it.  For each row we look up the
    dataset language code (``'EN'``, ``'IT'``, ``'NL'``) and map it to the
    backbone-specific string (``'en_XX'`` for mBART-50, ``'eng_Latn'`` for
    NLLB).

    When the backbone does not need language tags (e.g., ``facebook/bart-base``)
    the tokenizer is called without src/tgt configuration.

    Lengths
    -------
    Sequences are stored un-padded and padded per batch in ``collate_seq2seq``
    to avoid wasting memory on padding tokens in batches where all sequences
    are short.

    All-padding validation
    ----------------------
    An all-padding context or comment row would silently produce a NaN loss
    several epochs in; we reject such rows at load time with a clear error.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        backbone: str,
        context_max_len: int = 256,
        comment_max_len: int = 128,
    ):
        needs_lang = model_seq2seq.backbone_requires_lang_tag(backbone)

        contexts, comments, langs = [], [], []
        for _, row in df.iterrows():
            title = data_mod.clean_text(row['yt_title'])
            desc  = data_mod.clean_text(row['yt_description'])
            sep   = tokenizer.sep_token or tokenizer.eos_token or '</s>'
            ctx   = f' {sep} '.join(t for t in [title, desc] if t)
            com   = data_mod.clean_text(row['yt_comment'])
            contexts.append(ctx)
            comments.append(com)
            langs.append(str(row.get('lang', 'EN')))

        # Tokenize context (encoder input) and comment (decoder input).
        # For mBART / NLLB we must set src_lang / tgt_lang per-call; the
        # dataset mixes EN, IT and NL rows, so we tokenize in language groups
        # to avoid one giant src_lang overwriting another.
        if needs_lang:
            self.context_ids, self.context_mask = self._tokenize_by_lang(
                tokenizer, contexts, langs, backbone,
                max_len=context_max_len, side='src',
            )
            self.comment_ids, self.comment_mask = self._tokenize_by_lang(
                tokenizer, comments, langs, backbone,
                max_len=comment_max_len, side='tgt',
            )
        else:
            enc_ctx = tokenizer(contexts, truncation=True, max_length=context_max_len, padding=False)
            enc_com = tokenizer(comments, truncation=True, max_length=comment_max_len, padding=False)
            self.context_ids  = enc_ctx['input_ids']
            self.context_mask = enc_ctx['attention_mask']
            self.comment_ids  = enc_com['input_ids']
            self.comment_mask = enc_com['attention_mask']

        # Validate: no all-padding rows.
        _assert_nonempty('context', self.context_mask, contexts)
        _assert_nonempty('comment', self.comment_mask, comments)

        self.labels = df['st_y'].astype(int).tolist()

    @staticmethod
    def _tokenize_by_lang(tokenizer, texts, langs, backbone, max_len, side):
        """Tokenize texts in language-homogeneous groups, then reassemble.

        ``side`` is ``'src'`` for the encoder (sets ``src_lang``) or ``'tgt'``
        for the decoder (sets ``tgt_lang``).
        """
        n = len(texts)
        all_ids   = [None] * n
        all_masks = [None] * n

        # Group indices by language so we set the tokenizer's lang once per group.
        from collections import defaultdict
        lang_groups: dict = defaultdict(list)
        for i, lang in enumerate(langs):
            lang_groups[lang].append(i)

        for lang, indices in lang_groups.items():
            lang_code = model_seq2seq.lang_code_for(backbone, lang)
            if side == 'src':
                tokenizer.src_lang = lang_code
            else:
                tokenizer.tgt_lang = lang_code

            batch_texts = [texts[i] for i in indices]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_len,
                padding=False,
            )
            for pos, idx in enumerate(indices):
                all_ids[idx]   = enc['input_ids'][pos]
                all_masks[idx] = enc['attention_mask'][pos]

        return all_ids, all_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.context_ids[idx],
            self.context_mask[idx],
            self.comment_ids[idx],
            self.comment_mask[idx],
            self.labels[idx],
        )


def _assert_nonempty(field, masks, texts):
    """Fail at load time if any row tokenised to zero tokens."""
    empty = [i for i, m in enumerate(masks) if not any(m)]
    if not empty:
        return
    first = empty[0]
    raise ValueError(
        f'{len(empty)}/{len(masks)} {field} row(s) tokenised to zero tokens '
        f'(first: row {first}, text {texts[first]!r}).  '
        'Every row needs at least one token: an all-padding row has no finite '
        'attention score and its forward pass is undefined.'
    )


def collate_seq2seq(batch, pad_ctx: int, pad_com: int):
    """Pad a batch of (context, comment) pairs to their own longest sequence."""
    ctx_len = max(len(b[0]) for b in batch)
    com_len = max(len(b[2]) for b in batch)
    n = len(batch)

    ctx_ids  = torch.full((n, ctx_len), pad_ctx, dtype=torch.long)
    ctx_mask = torch.zeros((n, ctx_len), dtype=torch.long)
    com_ids  = torch.full((n, com_len), pad_com, dtype=torch.long)
    com_mask = torch.zeros((n, com_len), dtype=torch.long)
    labels   = torch.tensor([b[4] for b in batch], dtype=torch.long)

    for i, (cids, cmask, dids, dmask, _) in enumerate(batch):
        ctx_ids [i, :len(cids)]  = torch.tensor(cids,  dtype=torch.long)
        ctx_mask[i, :len(cmask)] = torch.tensor(cmask, dtype=torch.long)
        com_ids [i, :len(dids)]  = torch.tensor(dids,  dtype=torch.long)
        com_mask[i, :len(dmask)] = torch.tensor(dmask, dtype=torch.long)

    return ctx_ids, ctx_mask, com_ids, com_mask, labels


def make_loader(
    df, tokenizer, backbone, batch_size, context_max_len, comment_max_len,
    shuffle_seed=None,
):
    """Build a DataLoader for one split."""
    dataset = Seq2SeqDataset(df, tokenizer, backbone, context_max_len, comment_max_len)
    pad_ctx = tokenizer.pad_token_id
    pad_com = tokenizer.pad_token_id

    if shuffle_seed is not None:
        lengths = [len(c) + len(d)
                   for c, d in zip(dataset.context_ids, dataset.comment_ids)]
        sampler = data_mod.LengthGroupedSampler(lengths, batch_size, seed=shuffle_seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: collate_seq2seq(b, pad_ctx, pad_com),
            drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_seq2seq(b, pad_ctx, pad_com),
    )


# ---------------------------------------------------------------------------
# Gradient accumulation training helpers
# ---------------------------------------------------------------------------

def _effective_batch(batch_size: int, grad_accum: int) -> int:
    return batch_size * grad_accum


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(net, loader, device, criterion=None, use_amp: bool = False):
    """One full pass over a loader; returns preds, probs, labels and mean loss."""
    net.eval()
    preds, probs, labels, losses = [], [], [], []
    for ctx_ids, ctx_mask, com_ids, com_mask, label in loader:
        ctx_ids  = ctx_ids .to(device)
        ctx_mask = ctx_mask.to(device)
        com_ids  = com_ids .to(device)
        com_mask = com_mask.to(device)
        with autocast_ctx(device, use_amp):
            logits = net(ctx_ids, ctx_mask, com_ids, com_mask)
            if criterion is not None:
                losses.append(float(criterion(logits, label.to(device))))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(label.numpy())
    return {
        'preds':  np.concatenate(preds),
        'probs':  np.concatenate(probs),
        'labels': np.concatenate(labels),
        'loss':   float(np.mean(losses)) if losses else float('nan'),
    }


def evaluate(net, loader, device, criterion=None, use_amp: bool = False):
    out = collect(net, loader, device, criterion, use_amp=use_amp)
    return metrics_from(out['preds'], out['labels'], out['loss'])


def group_grad_norms(net):
    """Pre-clip gradient L2 norms for (encoder, decoder, head) groups."""
    groups = model_seq2seq.trainable_groups(net)
    return [
        math.sqrt(sum(float((p.grad ** 2).sum()) for p in g if p.grad is not None))
        if g else float('nan')
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- backbone ----
    parser.add_argument('--backbone', default=model_seq2seq.DEFAULT_BACKBONE,
                        help='HuggingFace encoder-decoder model ID '
                             '(default: facebook/mbart-large-50)')
    parser.add_argument('--pooling', choices=['eos', 'mean'], default='eos',
                        help="Representation to classify: 'eos' = last </s> hidden "
                             "state (default), 'mean' = mean of non-padding positions")
    # ---- data ----
    parser.add_argument('--languages', nargs='*', default=None,
                        help='Subset of EN/IT/NL (default: all available)')
    parser.add_argument('--label-column', default='stereotype',
                        help="Label column: 'stereotype' (Task A) or 'hate_speech'")
    parser.add_argument('--context-max-len', type=int, default=256,
                        help='Max token length for the encoder context (title + description)')
    parser.add_argument('--comment-max-len', type=int, default=128,
                        help='Max token length for the decoder comment')
    parser.add_argument('--train-extras', default=None,
                        help='Comma-separated TSVs appended to TRAIN split only '
                             '(val is unchanged)')
    # ---- training ----
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Physical batch size per step (default 4; '
                             'use --grad-accum to scale effective batch)')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Gradient accumulation steps (default 4 → '
                             'effective batch = batch-size × grad-accum = 16)')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    # ---- learning rates ----
    parser.add_argument('--enc-lr', type=float, default=1e-5,
                        help='LR for unfrozen encoder layers (default 1e-5)')
    parser.add_argument('--dec-lr', type=float, default=1e-5,
                        help='LR for unfrozen decoder layers (default 1e-5)')
    parser.add_argument('--head-lr', type=float, default=3e-4,
                        help='LR for the classification head (default 3e-4)')
    # ---- unfreezing ----
    parser.add_argument('--unfreeze-enc', type=int, default=0,
                        help='Top N encoder layers to unfreeze (0 = frozen)')
    parser.add_argument('--unfreeze-dec', type=int, default=2,
                        help='Top N decoder layers to unfreeze (default 2)')
    # ---- model misc ----
    parser.add_argument('--dropout', type=float, default=0.1)
    # ---- split / reproducibility ----
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    # ---- I/O ----
    parser.add_argument('--out', default=os.path.join(WEIGHTS_DIR, 'best_seq2seq_stereotype.pt'))
    parser.add_argument('--history', default=None,
                        help='CSV for per-epoch metrics (default: <out>_history.csv)')
    # ---- device / precision ----
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable fp16 mixed precision (auto-enabled on CUDA)')
    # ---- misc ----
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Load --out and report val metrics, then exit')

    args = parser.parse_args(argv)
    if args.history is None:
        args.history = os.path.splitext(args.out)[0] + '_history.csv'
    if args.grad_accum < 1:
        parser.error('--grad-accum must be >= 1')
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device    = resolve_device(args.device)
    use_amp   = not args.no_amp and device.type == 'cuda'
    eff_batch = _effective_batch(args.batch_size, args.grad_accum)

    print(f'Device: {device} | backbone: {args.backbone} | '
          f'mixed precision: {"ON (fp16)" if use_amp else "off"}')
    print(f'Batch: {args.batch_size} × {args.grad_accum} accum steps '
          f'= {eff_batch} effective')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    df_all = data_mod.load_frame(languages=args.languages, label_column=args.label_column)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)

    if args.train_extras:
        extras = [
            data_mod.load_frame_from_file(p.strip(), label_column=args.label_column)
            for p in args.train_extras.split(',') if p.strip()
        ]
        df_train = pd.concat([df_train, *extras], ignore_index=True)
        print(f'Train extras: +{sum(len(e) for e in extras)} rows '
              f'(total train {len(df_train)})')

    print(f'Train {len(df_train)} | val {len(df_val)} '
          f'(val-fraction {args.val_fraction}, split-seed {args.split_seed})')
    for lang, n in sorted(df_train['lang'].value_counts().items()):
        print(f'  train {lang}: {n}')

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Tokenizer: {tokenizer.__class__.__name__} | '
          f'vocab: {tokenizer.vocab_size} | '
          f'pad_id: {tokenizer.pad_token_id}')

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = make_loader(
        df_train, tokenizer, args.backbone,
        args.batch_size, args.context_max_len, args.comment_max_len,
        shuffle_seed=args.seed,
    )
    val_loader = make_loader(
        df_val, tokenizer, args.backbone,
        args.batch_size, args.context_max_len, args.comment_max_len,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    net = model_seq2seq.build_model(
        backbone=args.backbone,
        pooling=args.pooling,
        dropout=args.dropout,
        unfreeze_enc=args.unfreeze_enc,
        unfreeze_dec=args.unfreeze_dec,
    ).to(device)

    total     = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    enc_params, dec_params, head_params = model_seq2seq.trainable_groups(net)

    print(f'  d_model {net.d_model} | pooling {args.pooling} | '
          f'unfreeze enc {args.unfreeze_enc} / dec {args.unfreeze_dec}')
    print(f'  trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params')
    print(f'  lr: enc {args.enc_lr:g} ({sum(p.numel() for p in enc_params)/1e6:.2f}M) | '
          f'dec {args.dec_lr:g} ({sum(p.numel() for p in dec_params)/1e6:.2f}M) | '
          f'head {args.head_lr:g} ({sum(p.numel() for p in head_params)/1e6:.2f}M)')

    # ------------------------------------------------------------------
    # Evaluate-only path
    # ------------------------------------------------------------------
    if args.evaluate_only:
        net.load_state_dict(torch.load(args.out, map_location=device))
        out = collect(net, val_loader, device, use_amp=use_amp)
        print_metrics(metrics_from(out['preds'], out['labels']),
                      f'VALIDATION (loaded {args.out})')
        print(format_confusion_matrix(out['labels'], out['preds']))
        return

    # ------------------------------------------------------------------
    # Optimizer / loss / scaler
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    param_groups = [{'params': head_params, 'lr': args.head_lr}]
    if enc_params:
        param_groups.append({'params': enc_params, 'lr': args.enc_lr})
    if dec_params:
        param_groups.append({'params': dec_params, 'lr': args.dec_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler    = make_scaler(use_amp)

    best_score      = -1.0
    epochs_no_improve = 0
    history         = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loader.sampler.set_epoch(epoch)
        net.train()

        running_loss = 0.0
        seen         = 0
        gnorm_sum    = np.zeros(3)   # (enc, dec, head)
        n_steps      = 0

        optimizer.zero_grad()

        for step, (ctx_ids, ctx_mask, com_ids, com_mask, label) in \
                enumerate(train_loader, 1):

            ctx_ids  = ctx_ids .to(device)
            ctx_mask = ctx_mask.to(device)
            com_ids  = com_ids .to(device)
            com_mask = com_mask.to(device)
            label    = label   .to(device)

            with autocast_ctx(device, use_amp):
                logits = net(ctx_ids, ctx_mask, com_ids, com_mask)
                # Divide loss by accumulation steps so the gradient magnitude is
                # independent of grad_accum (mirrors HuggingFace Trainer).
                loss   = criterion(logits, label) / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum   # track the true loss
            seen         += label.size(0)

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                gnorm_sum += np.array(group_grad_norms(net))
                n_steps   += 1
                nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = running_loss / max(len(train_loader), 1)
        gnorms     = gnorm_sum / max(n_steps, 1)
        metrics    = evaluate(net, val_loader, device, criterion, use_amp=use_amp)

        history.append({
            'epoch':         epoch,
            'train_loss':    train_loss,
            'val_loss':      metrics['loss'],
            'val_macro_f1':  metrics['macro_f1'],
            'val_acc':       metrics['acc'],
            'seconds':       round(time.time() - started, 1),
            'gnorm_enc':     round(float(gnorms[0]), 4),
            'gnorm_dec':     round(float(gnorms[1]), 4),
            'gnorm_head':    round(float(gnorms[2]), 4),
        })
        print(
            f"Epoch {epoch:>3}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | val_macro_f1 {metrics['macro_f1']:.4f} | "
            f"val_acc {metrics['acc']:.4f} | {time.time() - started:.0f}s | "
            f"gnorm enc {gnorms[0]:.2f} dec {gnorms[1]:.2f} head {gnorms[2]:.2f} (clip 1.0)"
        )

        if metrics['macro_f1'] > best_score + 1e-4:
            best_score        = metrics['macro_f1']
            epochs_no_improve = 0
            torch.save(net.state_dict(), args.out)
            print(f'  saved new best to {args.out}  (macro_f1={best_score:.4f})')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'Early stopping after {epoch} epochs '
                      f'(no improvement for {args.patience})')
                break

    # ------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # ------------------------------------------------------------------
    pd.DataFrame(history).to_csv(args.history, index=False, encoding='utf-8')
    print(f'\nWrote {len(history)} epoch(s) of metrics to {args.history}')

    net.load_state_dict(torch.load(args.out, map_location=device))
    best_out = collect(net, val_loader, device, criterion, use_amp=use_amp)
    print_metrics(metrics_from(best_out['preds'], best_out['labels'], best_out['loss']),
                  'VALIDATION (best checkpoint)')
    print('--- CONFUSION MATRIX (best checkpoint) ---')
    print(format_confusion_matrix(best_out['labels'], best_out['preds']))

    print('--- PER LANGUAGE ---')
    for lang in sorted(df_val['lang'].unique()):
        subset = df_val[df_val['lang'] == lang].reset_index(drop=True)
        lang_loader = make_loader(
            subset, tokenizer, args.backbone,
            args.batch_size, args.context_max_len, args.comment_max_len,
        )
        out_lang = collect(net, lang_loader, device, use_amp=use_amp)
        m = metrics_from(out_lang['preds'], out_lang['labels'])
        print(f"  [{lang}] acc {m['acc']:.4f} | macro_f1 {m['macro_f1']:.4f}")
        print(format_confusion_matrix(out_lang['labels'], out_lang['preds']))


if __name__ == '__main__':
    main()
