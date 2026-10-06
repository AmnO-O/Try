#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune mmBERT for Task A: stereotype detection (yes / no) via MLM prompting.

Usage:
    python src/train.py
    python src/train.py --unfreeze-layers 4 --max-len 384 --epochs 30
    python src/train.py --languages EN NL
    python src/train.py --no-mlm-head
"""

import argparse
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from config import (
    BACKBONE_LR,
    BATCH_SIZE,
    HEAD_LR,
    LABEL_CLASSES,
    MAX_EPOCHS,
    MAX_LEN,
    MMBERT_MODEL_NAME,
    PATIENCE,
    READOUT_INIT,
    READOUT_INITS,
    SPLIT_SEED,
    UNFREEZE_LAYERS,
    USE_MLM_HEAD,
    VAL_FRACTION,
    WEIGHT_DECAY,
    best_weights_path,
)
from model import (
    label_token_ids,
    load_model,
    load_model_weights,
    load_tokenizer,
    save_model,
    trainable_parameter_groups,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-name', default=MMBERT_MODEL_NAME)
    parser.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--head-lr', type=float, default=HEAD_LR)
    parser.add_argument('--backbone-lr', type=float, default=BACKBONE_LR)
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--unfreeze-layers', type=int, default=UNFREEZE_LAYERS, help='0 freezes all of mmBERT')
    parser.add_argument('--no-mlm-head', action='store_true', help='Skip the pretrained MLM head')
    parser.add_argument('--readout-init', choices=READOUT_INITS, default=READOUT_INIT,
                        help="Initialisation of the 2-way readout: 'random' or 'verbalizer' "
                             '(pretrained decoder rows for the label words)')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out', default=None,
                        help='Checkpoint path (default: weights/best_mmbert_stereotype_prompt_<readout>.pt)')
    parser.add_argument('--history', default=None,
                        help='CSV for per-epoch metrics (default: <out>_history.csv)')
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (default: fp16 autocast on CUDA)')
    parser.add_argument('--evaluate-only', action='store_true', help='Load --out and report val metrics only')

    args = parser.parse_args(argv)
    # Resolved here rather than as argparse defaults because both depend on
    # --readout-init, which argparse cannot see when setting defaults.
    if args.out is None:
        args.out = best_weights_path(args.readout_init)
    if args.history is None:
        args.history = os.path.splitext(args.out)[0] + '_history.csv'
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def autocast_ctx(device, enabled: bool):
    """CUDA mixed-precision context, or a no-op when disabled or on CPU.

    mmBERT is 300M+ params; fp32 on a T4 runs on plain CUDA cores (~8 TFLOPS)
    while fp16 uses tensor cores (~65 TFLOPS), so AMP typically cuts epoch time
    by 3-5x. Only the forward+loss runs under autocast; weights stay fp32.
    """
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)  # torch >= 2.0
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)     # fallback


def build_optimizer(model, head_lr, backbone_lr, weight_decay):
    """Encoder params (if unfrozen) get a smaller LR than the fresh-ish head."""
    encoder_params, head_params = trainable_parameter_groups(model)
    groups = [{'params': head_params, 'lr': head_lr}]
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def collect(model, loader, device, criterion=None, use_amp: bool = False):
    """One pass over a loader. Evaluates even while the model is in train mode.

    Returns preds, probs, labels and the mean loss (nan when no criterion is
    given). Used by both predict() and evaluate() so computation happens once
    per call rather than once per metric.
    """
    model.eval()
    preds, probs, labels, losses = [], [], [], []
    for input_ids, mask, label, mask_pos in loader:
        with autocast_ctx(device, use_amp):
            logits = model(input_ids.to(device), mask.to(device), mask_pos.to(device))
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


def metrics_from(preds, labels, loss=float('nan')):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        'loss': loss,
        'acc': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_yes': f1_score(labels, preds, pos_label=1, zero_division=0),
        'precision_yes': precision_score(labels, preds, pos_label=1, zero_division=0),
        'recall_yes': recall_score(labels, preds, pos_label=1, zero_division=0),
    }


def evaluate(model, loader, device, criterion=None, use_amp: bool = False):
    out = collect(model, loader, device, criterion, use_amp=use_amp)
    return metrics_from(out['preds'], out['labels'], out['loss'])


def format_confusion_matrix(labels, preds):
    """2x2 matrix as text, rows = truth, columns = prediction. yes is positive."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return '\n'.join([
        '              pred no  pred yes',
        f'  true no    {tn:>8} {fp:>9}',
        f'  true yes   {fn:>8} {tp:>9}',
        f'  (tn={tn} fp={fp} fn={fn} tp={tp})',
    ])


def print_metrics(metrics, name='VALIDATION'):
    print(f'--- {name} ---')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


def evaluate_per_language(model, df, tokenizer, args, device, use_amp: bool = False):
    print('--- PER LANGUAGE ---')
    for lang in sorted(df['lang'].unique()):
        subset = df[df['lang'] == lang].reset_index(drop=True)
        loader = data_mod.build_inference_loader(subset, tokenizer, args.batch_size, args.max_len)
        out = collect(model, loader, device, use_amp=use_amp)
        metrics = metrics_from(out['preds'], out['labels'], out['loss'])
        print(f"  [{lang}] acc {metrics['acc']:.4f} | macro_f1 {metrics['macro_f1']:.4f}")
        print(format_confusion_matrix(out['labels'], out['preds']))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f'Device: {device} | mixed precision: {"ON (fp16)" if use_amp else "off"}')

    df_all = data_mod.load_frame(languages=args.languages, max_len=args.max_len)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | labels: {data_mod.label_distribution(df_train)}")

    tokenizer = load_tokenizer(args.model_name)
    if args.readout_init == 'verbalizer':
        print(f"Label words {LABEL_CLASSES} -> vocab ids {label_token_ids(tokenizer)}")

    train_loader, val_loader = data_mod.build_loaders(
        df_train, df_val, tokenizer, args.batch_size, args.max_len, args.seed
    )
    sampler = train_loader.sampler

    model = load_model(
        args.model_name,
        unfreeze_layers=args.unfreeze_layers,
        readout_init=args.readout_init,
        use_mlm_head=USE_MLM_HEAD and not args.no_mlm_head,
        tokenizer=tokenizer,
        device=device,
    )

    if args.evaluate_only:
        load_model_weights(model, args.out, device=device)
        print_metrics(evaluate(model, val_loader, device, use_amp=use_amp),
                      f'VALIDATION (loaded {args.out})')
        evaluate_per_language(model, df_val, tokenizer, args, device, use_amp=use_amp)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    scaler = make_scaler(use_amp)

    best_score = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for input_ids, mask, label, mask_pos in train_loader:
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = mask.to(device)
            label = label.to(device)
            mask_pos = mask_pos.to(device)

            with autocast_ctx(device, use_amp):
                logits = model(input_ids, attention_mask, mask_pos)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # Training-set accuracy comes from the same forward pass, so it costs
            # nothing extra. It is computed with dropout still on, hence slightly
            # pessimistic; use it only to spot underfitting, not to rank runs.
            running_correct += int((logits.argmax(dim=1) == label).sum())
            seen += label.size(0)

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = running_correct / max(seen, 1)
        metrics = evaluate(model, val_loader, device, criterion, use_amp=use_amp)
        val_score = metrics['macro_f1']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': metrics['loss'],
            'val_macro_f1': metrics['macro_f1'],
            'val_acc': metrics['acc'],
            'seconds': round(time.time() - started, 1),
        })

        print(
            f"Epoch {epoch:>3}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | val_macro_f1 {metrics['macro_f1']:.4f} | "
            f"val_acc {metrics['acc']:.4f} | {time.time() - started:.0f}s"
        )

        if val_score > best_score + 1e-4:
            best_score = val_score
            epochs_without_improvement = 0
            save_model(model, args.out)
            print(f'  saved new best to {args.out}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f'Early stopping after {epoch} epochs (no improvement for {args.patience})')
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.history, index=False, encoding='utf-8')
    print(f'\nWrote {len(history_df)} epoch(s) of metrics to {args.history}')

    load_model_weights(model, args.out, device=device)
    print_metrics(evaluate(model, val_loader, device, criterion, use_amp=use_amp),
                  'VALIDATION (best checkpoint)')
    best = collect(model, val_loader, device, criterion, use_amp=use_amp)
    print('--- CONFUSION MATRIX (best checkpoint) ---')
    print(format_confusion_matrix(best['labels'], best['preds']))
    evaluate_per_language(model, df_val, tokenizer, args, device, use_amp=use_amp)


if __name__ == '__main__':
    main()