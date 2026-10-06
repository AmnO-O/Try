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

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from config import (
    BACKBONE_LR,
    BATCH_SIZE,
    BEST_WEIGHTS,
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
)
from model import (
    label_token_ids,
    load_model,
    load_model_weights,
    load_tokenizer,
    save_model,
    trainable_parameter_groups,
)


def parse_args():
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
    parser.add_argument('--out', default=BEST_WEIGHTS)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--evaluate-only', action='store_true', help='Load --out and report val metrics only')
    return parser.parse_args()


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


def build_optimizer(model, head_lr, backbone_lr, weight_decay):
    """Encoder params (if unfrozen) get a smaller LR than the fresh-ish head."""
    encoder_params, head_params = trainable_parameter_groups(model)
    groups = [{'params': head_params, 'lr': head_lr}]
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (class indices, yes-probabilities, true labels)."""
    model.eval()
    preds, probs, labels = [], [], []
    for input_ids, mask, label, mask_pos in loader:
        logits = model(input_ids.to(device), mask.to(device), mask_pos.to(device))
        yes_index = LABEL_CLASSES.index('yes')
        probs.append(torch.softmax(logits, dim=1)[:, yes_index].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(labels)


def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds, _, labels = predict(model, loader, device)
    return {
        'acc': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_yes': f1_score(labels, preds, pos_label=1, zero_division=0),
        'precision_yes': precision_score(labels, preds, pos_label=1, zero_division=0),
        'recall_yes': recall_score(labels, preds, pos_label=1, zero_division=0),
    }


def print_metrics(metrics, name='VALIDATION'):
    print(f'--- {name} ---')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


def evaluate_per_language(model, df, tokenizer, args, device):
    print('--- PER LANGUAGE ---')
    for lang in sorted(df['lang'].unique()):
        subset = df[df['lang'] == lang].reset_index(drop=True)
        _, loader = data_mod.build_loaders(subset, subset, tokenizer, args.batch_size, args.max_len)
        metrics = evaluate(model, loader, device)
        print(f"  [{lang}] acc {metrics['acc']:.4f} | macro_f1 {metrics['macro_f1']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f'Device: {device}')

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
        print_metrics(evaluate(model, val_loader, device), f'VALIDATION (loaded {args.out})')
        evaluate_per_language(model, df_val, tokenizer, args, device)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)

    best_score = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for input_ids, mask, label, mask_pos in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids.to(device), mask.to(device), mask_pos.to(device))
            loss = criterion(logits, label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device)
        val_score = metrics['macro_f1']

        print(
            f"Epoch {epoch:>3}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_macro_f1 {metrics['macro_f1']:.4f} | val_acc {metrics['acc']:.4f} | "
            f"{time.time() - started:.0f}s"
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

    load_model_weights(model, args.out, device=device)
    print_metrics(evaluate(model, val_loader, device), 'VALIDATION (best checkpoint)')
    evaluate_per_language(model, df_val, tokenizer, args, device)


if __name__ == '__main__':
    main()