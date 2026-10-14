#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the two-tower + cross-attention classifier (see model_xattn.py).

Same data pipeline, split, labels, metrics and early-stopping conventions as
train.py, so runs are comparable: identical --split-seed / --val-fraction keep
the same video-level val split, and --train-extras appends augmented files to
the TRAIN split only.

Fields are joined with the tokenizer's own separator token (XLM-R `</s>`), not
the literal `[SEP]` text from config: these tokenizers have no `[SEP]` token and
would split it into ordinary subwords, losing the pretrained boundary signal.

Usage:
    python src/train_xattn.py --languages EN IT NL --epochs 10
    python src/train_xattn.py --backbone jhu-clsp/mmbert-base --heads 4 --layers 3
    python src/train_xattn.py --unfreeze-layers 2 --head-lr 5e-4 --backbone-lr 2e-5
    python src/train_xattn.py --freeze-epochs 5 --unfreeze-layers 2 --layers 3
    python src/train_xattn.py --query-fields comment,title --unfreeze-layers 2 --layers 3
    python src/train_xattn.py --query-fields title,comment --keep-kv-title --layers 3
    python src/train_xattn.py --no-pooled-v --layers 3      # cross-attention only, no late fusion
    python src/train_xattn.py --shared-encoder --layers 3   # one shared encoder (pre-2026 default)
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


FIELD_COLUMNS = {'comment': 'yt_comment', 'title': 'yt_title', 'description': 'yt_description'}
# Only the short fields may move into the query stream; description is long
# (mean 91 tokens, 27% over 128) and would eat the comment budget.
QUERY_CHOICES = ('comment', 'title')


def sep_for(tokenizer) -> str:
    """The tokenizer's own segment separator, as literal text.

    `config.SEP_TOKEN` is the BERT-style `[SEP]`, which *neither* of our
    tokenizers has as a special token: XLM-R splits it into four ordinary
    subwords (`[`, `S`, `EP`, `]`) and mmBERT into three. So joining fields with
    literal `[SEP]` text hands the pretrained encoder a boundary marker it has
    never seen, instead of the single separator token it was trained on. The
    prompt baseline substitutes the real token (data.py:255); the xattn path
    was not doing so.
    """
    return tokenizer.sep_token or tokenizer.eos_token or SEP_TOKEN


def split_fields(query_fields, keep_kv_title: bool = False):
    """Returns (query_fields, video_fields).

    A field placed in the query is normally removed from the K/V stream. With
    `keep_kv_title` the title stays in *both*, so adding it in-sequence does not
    also take away its cross-attention / pooled-video channels.
    """
    q = tuple(dict.fromkeys(query_fields))
    v = tuple(f for f in ('title', 'description')
              if f not in q or (f == 'title' and keep_kv_title))
    return q, v


class PairedDataset(Dataset):
    """Tokenizes the comment and the video context into two separate sequences.

    `query_fields` selects which fields make up the query (comment) stream; a
    field listed there is *removed* from the video (K/V) stream unless
    `keep_kv_title` pins the title to both.
    """

    def __init__(self, df, tokenizer, comment_max_len: int = 128, video_max_len: int = 192,
                 query_fields=('comment',), keep_kv_title: bool = False):
        query_fields, video_fields = split_fields(query_fields, keep_kv_title)
        # Join with the tokenizer's real separator token, not config.SEP_TOKEN:
        # a literal "[SEP]" is 3-4 ordinary subwords to these tokenizers, so the
        # tower boundary the encoder was pretrained to see would be missing.
        sep = sep_for(tokenizer)
        comments, videos = [], []
        for _, row in df.iterrows():
            # Query fields keep the order given on the command line, so
            # '--query-fields comment,title' puts the comment first and a
            # right-truncated query never drops the labelled text.
            q = [data_mod.clean_text(row[FIELD_COLUMNS[f]]) for f in query_fields]
            comments.append(f' {sep} '.join(t for t in q if t))

            v = [data_mod.clean_text(row[FIELD_COLUMNS[f]]) for f in video_fields]
            videos.append(f' {sep} '.join(t for t in v if t))

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


def make_loader(df, tokenizer, batch_size, comment_max_len, video_max_len, shuffle_sampler=None,
                query_fields=('comment',), keep_kv_title: bool = False):
    dataset = PairedDataset(df, tokenizer, comment_max_len, video_max_len, query_fields, keep_kv_title)
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
    parser.add_argument('--query-fields', default='comment',
                        help='Comma-separated fields placed in the query/comment stream '
                             f'(choices: {", ".join(QUERY_CHOICES)}). A listed field is removed from '
                             'the K/V stream, so "comment,title" moves the title in-sequence and '
                             'leaves only the description as attention memory. Order is preserved; '
                             'comment must be included.')
    parser.add_argument('--keep-kv-title', action='store_true',
                        help='Keep the title in the K/V stream even when it is also a query field, '
                             'so the title is added in-sequence without losing its attention and '
                             'pooled-video channels (a "duplicate title" format).')
    parser.add_argument('--no-pooled-v', action='store_true',
                        help='Drop the mean-pooled video vector from the head, leaving the '
                             'cross-attention stack as the only path from the video tower '
                             '(ablation: is the cross-attention earning its keep?)')
    parser.add_argument('--shared-encoder', action='store_true',
                        help='Share one encoder between the comment and video towers instead of '
                             'giving each tower its own copy (the default). Halves the parameter '
                             'count and makes the video K/V cacheable, but the two towers then '
                             'share weights and their gradients interfere.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--unfreeze-layers', type=int, default=2,
                        help='Unfreeze last N encoder blocks (0 = frozen; use a large N to unfreeze all)')
    parser.add_argument('--freeze-epochs', type=int, default=0,
                        help='Two-phase: keep the encoder frozen for K epoch(s), then unfreeze the '
                             'last --unfreeze-layers blocks at --backbone-lr (0 = single phase)')
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
    fields = tuple(dict.fromkeys(f.strip().lower() for f in args.query_fields.split(',') if f.strip()))
    bad = [f for f in fields if f not in QUERY_CHOICES]
    if bad:
        parser.error(f'--query-fields must be a subset of {QUERY_CHOICES}; got {bad}')
    if 'comment' not in fields:
        parser.error('--query-fields must include comment (it is the labelled text)')
    args.query_fields = fields
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
    query_fields, video_fields = split_fields(args.query_fields, args.keep_kv_title)
    for lang, n in sorted(df_train['lang'].value_counts().items()):
        print(f'  train {lang}: {n}')

    from model import load_tokenizer, unfreeze_last_n
    tokenizer = load_tokenizer(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    sep = sep_for(tokenizer)
    print(f'  query stream: {f" {sep} ".join(query_fields)} | '
          f'K/V stream: {f" {sep} ".join(video_fields) or "(none)"}')

    # Two-phase: phase 1 keeps the encoder frozen so the cross-attention stack and
    # head settle against stationary features; phase 2 (epoch freeze_epochs+1)
    # re-opens the last --unfreeze-layers blocks at the backbone LR.
    unfreeze_epoch = args.freeze_epochs + 1 if args.freeze_epochs > 0 else None
    build_unfreeze = 0 if unfreeze_epoch is not None else args.unfreeze_layers

    net = model_xattn.build_model(
        args.backbone, unfreeze_layers=build_unfreeze,
        heads=args.heads, layers=args.layers, dropout=args.dropout, ffn_mult=args.ffn_mult,
        use_pooled_v=not args.no_pooled_v, separate_encoders=not args.shared_encoder,
    ).to(device)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    encoders = 'separate' if net.separate_encoders else 'shared'
    print(f'  d_model {net.d_model} | heads {args.heads} | cross-attn layers {args.layers} | '
          f'encoders {encoders} | head in {"pooled_c+pooled_v" if net.use_pooled_v else "pooled_c only"} | '
          f'trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params')
    if unfreeze_epoch is not None:
        if args.unfreeze_layers <= 0:
            print('  NOTE: --freeze-epochs has no effect with --unfreeze-layers 0')
        elif unfreeze_epoch > args.epochs:
            print(f'  WARNING: --freeze-epochs {args.freeze_epochs} >= --epochs {args.epochs}; '
                  f'phase 2 never starts')
        else:
            print(f'  two-phase: encoder frozen for {args.freeze_epochs} epoch(s), then unfreeze '
                  f'last {args.unfreeze_layers} block(s) at lr {args.backbone_lr}')

    train_loader = make_loader(df_train, tokenizer, args.batch_size,
                               args.max_len, args.video_max_len, shuffle_sampler=args.seed,
                               query_fields=args.query_fields, keep_kv_title=args.keep_kv_title)
    val_loader = make_loader(df_val, tokenizer, args.batch_size, args.max_len, args.video_max_len,
                             query_fields=args.query_fields, keep_kv_title=args.keep_kv_title)

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
        if unfreeze_epoch is not None and epoch == unfreeze_epoch and args.unfreeze_layers > 0:
            for enc in net.encoders():
                unfreeze_last_n(enc, args.unfreeze_layers)
            net.encoder_trainable = True
            new_params = [p for enc in net.encoders() for p in enc.parameters() if p.requires_grad]
            if new_params:
                optimizer.add_param_group({'params': new_params, 'lr': args.backbone_lr})
            # Phase 1 is a warm-up, not a contest: the patience clock must start
            # at the unfreeze epoch or early stopping can kill the run before the
            # fine-tune phase ever gets a chance.
            epochs_no_improve = 0
            from model import layer_index_in_name
            unfrozen = sorted({
                i for n, p in net.named_parameters()
                if p.requires_grad and n.startswith(model_xattn.ENCODER_PREFIXES)
                for i in [layer_index_in_name(n)] if i is not None
            })
            print(f'  [phase 2] epoch {epoch}: unfroze encoder blocks {unfrozen} '
                  f'({sum(p.numel() for p in new_params)/1e6:.2f}M params now trainable)')
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
        loader = make_loader(subset, tokenizer, args.batch_size, args.max_len, args.video_max_len,
                             query_fields=args.query_fields, keep_kv_title=args.keep_kv_title)
        out = collect(net, loader, device, use_amp=use_amp)
        m = metrics_from(out['preds'], out['labels'])
        print(f"  [{lang}] acc {m['acc']:.4f} | macro_f1 {m['macro_f1']:.4f}")
        print(format_confusion_matrix(out['labels'], out['preds']))


if __name__ == '__main__':
    main()
