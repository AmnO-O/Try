#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the two-tower + cross-attention classifier (see model_xattn.py).

Same data pipeline, split, labels, metrics and early-stopping conventions as
train.py, so runs are comparable: identical --split-seed / --val-fraction keep
the same video-level val split, and --train-extras appends augmented files to
the TRAIN split only.

Usage:
    python src/train_xattn.py --languages EN IT NL --epochs 10
    python src/train_xattn.py --backbone jhu-clsp/mmbert-base --heads 4 --layers 3
    python src/train_xattn.py --unfreeze-layers 2 --head-lr 5e-4 --backbone-lr 2e-5
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
import model_xattn
from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    SEP_TOKEN,
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


class PairedDataset(Dataset):
    """Tokenizes the comment and the video context into two separate sequences."""

    def __init__(self, df, tokenizer, comment_max_len: int = 128, video_max_len: int = 192):
        comments, videos = [], []
        for _, row in df.iterrows():
            comments.append(data_mod.clean_text(row['yt_comment']))
            title = data_mod.clean_text(row['yt_title'])
            desc = data_mod.clean_text(row['yt_description'])
            videos.append(f' {SEP_TOKEN} '.join(f for f in (title, desc) if f))

        c = tokenizer(comments, truncation=True, max_length=comment_max_len, padding=False)
        v = tokenizer(videos, truncation=True, max_length=video_max_len, padding=False)
        self.comment_ids = c['input_ids']
        self.comment_mask = c['attention_mask']
        self.video_ids = v['input_ids']
        self.video_mask = v['attention_mask']
        self.labels = df['st_y'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.comment_ids[idx], self.comment_mask[idx],
                self.video_ids[idx], self.video_mask[idx], self.labels[idx])


def collate_paired(batch, pad_token_id: int):
    c_len = max(len(b[0]) for b in batch)
    v_len = max(len(b[2]) for b in batch)
    n = len(batch)
    comment_ids = torch.full((n, c_len), pad_token_id, dtype=torch.long)
    comment_mask = torch.zeros((n, c_len), dtype=torch.long)
    video_ids = torch.full((n, v_len), pad_token_id, dtype=torch.long)
    video_mask = torch.zeros((n, v_len), dtype=torch.long)
    labels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    for i, (cids, cmask, vids, vmask, _) in enumerate(batch):
        comment_ids[i, :len(cids)] = torch.tensor(cids)
        comment_mask[i, :len(cmask)] = torch.tensor(cmask)
        video_ids[i, :len(vids)] = torch.tensor(vids)
        video_mask[i, :len(vmask)] = torch.tensor(vmask)
    return comment_ids, comment_mask, video_ids, video_mask, labels


@torch.no_grad()
def collect(net, loader, device, criterion=None, use_amp: bool = False):
    net.eval()
    preds, probs, labels, losses = [], [], [], []
    for cids, cmask, vids, vmask, label in loader:
        with autocast_ctx(device, use_amp):
            logits = net(cids.to(device), cmask.to(device), vids.to(device), vmask.to(device))
            if criterion is not None:
                losses.append(float(criterion(logits, label.to(device))))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(label.numpy())
    return {
        'preds': np.concatenate(preds),
        'probs': np.concatenate(probs),
        'labels': np.concatenate(labels),
        'loss': float(np.mean(losses)) if losses else float('nan'),
    }


def evaluate(net, loader, device, criterion=None, use_amp: bool = False):
    out = collect(net, loader, device, criterion, use_amp=use_amp)
    return metrics_from(out['preds'], out['labels'], out['loss'])


def make_loader(df, tokenizer, batch_size, comment_max_len, video_max_len, shuffle_sampler=None):
    dataset = PairedDataset(df, tokenizer, comment_max_len, video_max_len)
    pad = tokenizer.pad_token_id
    if shuffle_sampler is not None:
        lengths = [len(c) + len(v) for c, v in zip(dataset.comment_ids, dataset.video_ids)]
        sampler = data_mod.LengthGroupedSampler(lengths, batch_size, seed=shuffle_sampler)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=lambda b: collate_paired(b, pad), drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=lambda b: collate_paired(b, pad))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backbone', default=model_xattn.DEFAULT_BACKBONE)
    parser.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--label-column', default='stereotype')
    parser.add_argument('--max-len', type=int, default=160, help='Comment max tokens')
    parser.add_argument('--video-max-len', type=int, default=256, help='Title+description max tokens (all videos fit)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--unfreeze-layers', type=int, default=0, help='Unfreeze last N encoder blocks (0 = frozen)')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3, help='Number of cross-attention blocks')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ffn-mult', type=int, default=4)
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--out', default=os.path.join(WEIGHTS_DIR, 'best_xattn_stereotype.pt'))
    parser.add_argument('--history', default=None)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--train-extras', default=None,
                        help='Comma-separated TSVs appended to the TRAIN split only (val unchanged)')
    args = parser.parse_args(argv)
    if args.history is None:
        args.history = os.path.splitext(args.out)[0] + '_history.csv'
    if args.heads <= 0 or args.layers <= 0:
        parser.error('--heads and --layers must be >= 1')
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f'Device: {device} | backbone: {args.backbone} | '
          f'mixed precision: {"ON (fp16)" if use_amp else "off"}')

    df_all = data_mod.load_frame(languages=args.languages, label_column=args.label_column)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    if args.train_extras:
        extras = [data_mod.load_frame_from_file(f.strip(), label_column=args.label_column)
                  for f in args.train_extras.split(',') if f.strip()]
        df_train = pd.concat([df_train, *extras], ignore_index=True)
        print(f'Train extras: +{sum(len(e) for e in extras)} rows (total train {len(df_train)})')

    print(f'Train {len(df_train)} | val {len(df_val)} (val-fraction {args.val_fraction}, '
          f'split-seed {args.split_seed})')
    for lang, n in sorted(df_train['lang'].value_counts().items()):
        print(f'  train {lang}: {n}')

    from model import load_tokenizer
    tokenizer = load_tokenizer(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    net = model_xattn.build_model(
        args.backbone, unfreeze_layers=args.unfreeze_layers,
        heads=args.heads, layers=args.layers, dropout=args.dropout, ffn_mult=args.ffn_mult,
    ).to(device)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f'  d_model {net.d_model} | heads {args.heads} | cross-attn layers {args.layers} | '
          f'trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params')

    train_loader = make_loader(df_train, tokenizer, args.batch_size,
                               args.max_len, args.video_max_len, shuffle_sampler=args.seed)
    val_loader = make_loader(df_val, tokenizer, args.batch_size, args.max_len, args.video_max_len)

    if args.evaluate_only:
        net.load_state_dict(torch.load(args.out, map_location=device))
        print_metrics(evaluate(net, val_loader, device, use_amp=use_amp),
                      f'VALIDATION (loaded {args.out})')
        return

    backbone_params, head_params = model_xattn.trainable_groups(net)
    groups = [{'params': head_params, 'lr': args.head_lr}]
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = make_scaler(use_amp)

    best_score, epochs_no_improve, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loader.sampler.set_epoch(epoch)
        net.train()
        running_loss, seen = 0.0, 0
        for cids, cmask, vids, vmask, label in train_loader:
            optimizer.zero_grad()
            with autocast_ctx(device, use_amp):
                logits = net(cids.to(device), cmask.to(device), vids.to(device), vmask.to(device))
                loss = criterion(logits, label.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            seen += label.size(0)

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(net, val_loader, device, criterion, use_amp=use_amp)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': metrics['loss'], 'val_macro_f1': metrics['macro_f1'],
                        'val_acc': metrics['acc'], 'seconds': round(time.time() - started, 1)})
        print(f"Epoch {epoch:>3}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {metrics['loss']:.4f} | val_macro_f1 {metrics['macro_f1']:.4f} | "
              f"val_acc {metrics['acc']:.4f} | {time.time() - started:.0f}s")

        if metrics['macro_f1'] > best_score + 1e-4:
            best_score = metrics['macro_f1']
            epochs_no_improve = 0
            torch.save(net.state_dict(), args.out)
            print(f'  saved new best to {args.out}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'Early stopping after {epoch} epochs (no improvement for {args.patience})')
                break

    pd.DataFrame(history).to_csv(args.history, index=False, encoding='utf-8')
    print(f'\nWrote {len(history)} epoch(s) of metrics to {args.history}')

    net.load_state_dict(torch.load(args.out, map_location=device))
    print_metrics(evaluate(net, val_loader, device, criterion, use_amp=use_amp),
                  'VALIDATION (best checkpoint)')
    best = collect(net, val_loader, device, criterion, use_amp=use_amp)
    print('--- CONFUSION MATRIX (best checkpoint) ---')
    print(format_confusion_matrix(best['labels'], best['preds']))

    print('--- PER LANGUAGE ---')
    for lang in sorted(df_val['lang'].unique()):
        subset = df_val[df_val['lang'] == lang].reset_index(drop=True)
        loader = make_loader(subset, tokenizer, args.batch_size, args.max_len, args.video_max_len)
        out = collect(net, loader, device, use_amp=use_amp)
        m = metrics_from(out['preds'], out['labels'])
        print(f"  [{lang}] acc {m['acc']:.4f} | macro_f1 {m['macro_f1']:.4f}")
        print(format_confusion_matrix(out['labels'], out['preds']))


if __name__ == '__main__':
    main()
