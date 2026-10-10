#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plain [CLS]-classification baseline for Task A via HF Trainer.

Same corpus, split and prompt text as train.py, but the model is a standard
AutoModelForSequenceClassification over mmBERT: the classifier head reads the
lead token (<s>) instead of the [MASK]-slot vector that the prompt head reads.
Running both on the same recipe isolates the value of the prompt vehicle:

    python src/train_cls.py                                  # best-recipe baseline
    python src/train_cls.py --languages EN NL --split-seed 7
    python src/train_cls.py --train-extras llm_aug_EN_to_IT_training.tsv,...

Inputs mirror train.py --input / --train-extras / --label-column, so a stage-1
transfer run or an augmentation condition is expressed identically. Outputs use
the same <out>_history.csv convention; the metric tracked for best-model is
validation macro-F1 (Task A metric).
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from config import (
    BACKBONE_LR,
    BATCH_SIZE,
    MASK_LABEL,
    MAX_EPOCHS,
    MAX_LEN,
    MMBERT_MODEL_NAME,
    PATIENCE,
    SEP_TOKEN,
    SPLIT_SEED,
    UNFREEZE_LAYERS,
    VAL_FRACTION,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-name', default=MMBERT_MODEL_NAME)
    parser.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--input', default=None,
                        help='Single TSV to train on instead of the --languages glob')
    parser.add_argument('--label-column', default='stereotype',
                        help="Label column: 'stereotype' (Task A) or 'hate_speech'")
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--lr', type=float, default=BACKBONE_LR, help='Learning rate (all trainable params)')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--unfreeze-layers', type=int, default=UNFREEZE_LAYERS, help='0 freezes all of mmBERT')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out', default=None, help='Checkpoint path (state dict)')
    parser.add_argument('--history', default=None,
                        help='CSV for per-epoch metrics (default: <out>_history.csv)')
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--train-extras', default=None,
                        help="Comma-separated schema-compatible TSVs appended to the TRAIN split only, "
                             'after the video-level split (val stays identical to baseline)')
    parser.add_argument('--from-checkpoint', default=None,
                        help='Continue from a saved state dict (same architecture)')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Load --out and report val metrics only')
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(WEIGHTS_DIR, 'best_mmbert_stereotype_cls.pt')
    if args.history is None:
        args.history = os.path.splitext(args.out)[0] + '_history.csv'
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    if args.input:
        df_all = data_mod.load_frame_from_file(args.input, max_len=args.max_len,
                                               label_column=args.label_column)
    else:
        df_all = data_mod.load_frame(languages=args.languages, max_len=args.max_len,
                                     label_column=args.label_column)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    if args.train_extras:
        extra_frames = [
            data_mod.load_frame_from_file(path.strip(), max_len=args.max_len,
                                          label_column=args.label_column)
            for path in args.train_extras.split(',') if path.strip()
        ]
        df_train = pd.concat([df_train] + extra_frames, ignore_index=True)
        print(f'Train extras: +{sum(len(f) for f in extra_frames)} rows '
              f'(total train {len(df_train)}) | extra langs: '
              f"{sorted({l for f in extra_frames for l in f['lang'].unique()})}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | labels: {data_mod.label_distribution(df_train)}")
    return df_train, df_val


class CLSDataset(torch.utils.data.Dataset):
    """Left-truncated tokenization of the prompt text minus its [MASK] slot.

    The prompt model keeps the last max_len tokens so the question+slot survive
    right-truncation; the same policy is applied here so both heads read exactly
    the same span, differing only in where they pool from.
    """

    def __init__(self, prompts, labels, tokenizer, max_len: int = MAX_LEN):
        texts = [
            str(p).replace(MASK_LABEL, '').replace(SEP_TOKEN, tokenizer.sep_token)
            for p in prompts
        ]
        full = tokenizer(texts, truncation=False, padding=False)
        self.input_ids = [
            ids[-max_len:] if len(ids) > max_len else ids
            for ids in full['input_ids']
        ]
        self.attention_mask = [[1] * len(ids) for ids in self.input_ids]
        self.labels = torch.as_tensor(np.array(labels, dtype=np.int64))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


def collate(batch, pad_token_id: int):
    longest = max(len(x['input_ids']) for x in batch)
    input_ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, x in enumerate(batch):
        input_ids[row, :len(x['input_ids'])] = torch.tensor(x['input_ids'], dtype=torch.long)
        attention_mask[row, :len(x['attention_mask'])] = torch.tensor(x['attention_mask'], dtype=torch.long)
    labels = torch.stack([x['labels'] for x in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {f'{k}': v for k, v in metrics_from(
        np.argmax(logits, axis=1), np.asarray(labels)).items()}


@torch.no_grad()
def evaluate(model, df, tokenizer, args, device) -> dict:
    dataset = CLSDataset(df['prompt'].values, df['st_y'].values, tokenizer, max_len=args.max_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = model(input_ids=batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device)).logits
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(batch['labels'].numpy())
    out = metrics_from(np.concatenate(preds), np.concatenate(labels))
    print('--- VALIDATION ---')
    for key, value in out.items():
        print(f'  {key}: {value:.4f}')
    return out


def print_per_language(model, df, tokenizer, args, device):
    print('--- PER LANGUAGE ---')
    for lang in sorted(df['lang'].unique()):
        subset = df[df['lang'] == lang].reset_index(drop=True)
        dataset = CLSDataset(subset['prompt'].values, subset['st_y'].values, tokenizer,
                             max_len=args.max_len)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(input_ids=batch['input_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device)).logits
                preds.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(batch['labels'].numpy())
        m = metrics_from(np.concatenate(preds), np.concatenate(labels))
        print(f"  [{lang}] acc {m['acc']:.4f} | macro_f1 {m['macro_f1']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                             Trainer, TrainingArguments)
    from model import unfreeze_last_n
    try:
        from transformers.integrations import EarlyStoppingCallback
        can_early_stop = True
    except Exception:
        can_early_stop = False

    df_train, df_val = load_data(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    if args.from_checkpoint:
        model.load_state_dict(torch.load(args.from_checkpoint, map_location='cpu'))
        print(f'  resumed from checkpoint: {args.from_checkpoint}')

    n_blocks = unfreeze_last_n(model.base_model, args.unfreeze_layers)
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  unfrozen: last {args.unfreeze_layers} blocks (of {n_blocks}) | '
          f'trainable {trainable / 1e6:.2f}M / {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params')
    model.to(device)

    if args.evaluate_only:
        model.load_state_dict(torch.load(args.out, map_location='cpu'))
        model.to(device)
        evaluate(model, df_val, tokenizer, args, device)
        print_per_language(model, df_val, tokenizer, args, device)
        return

    train_ds = CLSDataset(df_train['prompt'].values, df_train['st_y'].values, tokenizer,
                          max_len=args.max_len)
    val_ds = CLSDataset(df_val['prompt'].values, df_val['st_y'].values, tokenizer,
                        max_len=args.max_len)
    pad_id = tokenizer.pad_token_id
    train_collate = lambda batch: collate(batch, pad_id)

    kwargs = dict(
        output_dir=os.path.join(os.path.dirname(args.out), 'cls_runs'),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=(device.type == 'cuda'),
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=25,
        disable_tqdm=False,
        seed=args.seed,
        report_to=[],
    )
    try:
        training_args = TrainingArguments(eval_strategy='epoch', **kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy='epoch', **kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=train_collate,
        compute_metrics=compute_metrics,
    )
    if can_early_stop and args.patience:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))

    started = time.time()
    trainer.train()
    print(f'Training done in {time.time() - started:.0f}s')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f'Saved best (validated) state dict -> {args.out}')

    rows = []
    for entry in trainer.state.log_history:
        if 'eval_macro_f1' in entry:
            row = {'epoch': entry.get('epoch', float('nan')),
                   'val_loss': entry.get('eval_loss', float('nan')),
                   'val_macro_f1': entry['eval_macro_f1'],
                   'val_acc': entry.get('eval_accuracy', float('nan'))}
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(args.history, index=False, encoding='utf-8')
        print(f'Wrote {len(rows)} evaluation rows -> {args.history}')

    evaluate(model, df_val, tokenizer, args, device)
    print_per_language(model, df_val, tokenizer, args, device)


if __name__ == '__main__':
    main()