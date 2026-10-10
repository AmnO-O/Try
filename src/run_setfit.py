#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SetFit contrastive baseline for Task A (stereotype detection).

Non-prompt control for the prompt-MLM pipeline (train.py). Same mmBERT family,
same video-grouped split, same seeds, same macro-F1, but the model is SetFit:
contrastive fine-tuning of a sentence-embedding backbone followed by a small
differentiable head trained on the frozen embeddings. The text is fed WITHOUT
the question/answer [MASK] framing, isolating whether the prompt vehicle adds
value.

The modern SetFit v2 API is used:
  * `SetFitModel(model_body=<SentenceTransformer>, model_head=<head>)`
  * `datasets.Dataset` (not pandas) fed to `Trainer` + `TrainingArguments`
  * `trainer.train()` (embedding contrastive phase, then classifier phase)
  * `--head-only` calls `model.fit(...)` directly: no contrastive phase, so the
    pretrained-embeddings + head control stays separable from contrastive + head
  * `model.save_pretrained(<dir>)` / `SetFitModel.from_pretrained(<dir>)`

The classifier head is a torch `SetFitHead` (CrossEntropyLoss, `--head-epochs`
epochs) by default, or sklearn `LogisticRegression` (`--head lr`, tuned via
`--lr-c` / `--lr-max-iter`; epochs and batch size are ignored for LR).
Training is deterministic: fixed seed, and SetFit's canonical classifier phase
is a fixed-epoch run -- there is no per-epoch early stopping on val macro-F1,
metrics are reported once after training on the video-grouped validation split.

Backbone: `vllm-sr/mmbert-embed-32k-2d-matryoshka` (mmBERT-Embed, a fine-tune of
`jhu-clsp/mmbert-base`), which loads directly as a SentenceTransformer.
`--backbone jhu-clsp/mmbert-base` wraps the raw base with mean pooling;
load failures auto-fall-back to multilingual MiniLM.

Requires setfit >= 1.0 (2.x supported: v2 keeps `Trainer`, `TrainingArguments`
and `SetFitHead`; only the deprecated `SetFitTrainer` was removed).

Usage:
    python src/run_setfit.py --split-seed 40 --out weights/setfit_s40
    python src/run_setfit.py --dry-run            # config only, no torch/setfit
    python src/run_setfit.py --smoke 256 --num-iterations 1 --head-epochs 2
    python src/run_setfit.py --head lr --head-only --out weights/setfit_s40_lr
"""

import argparse
import csv
import os
import random
import sys
import time

from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    MAX_LEN,
    SPLIT_SEED,
    VAL_FRACTION,
)

DEFAULT_BACKBONE = 'vllm-sr/mmbert-embed-32k-2d-matryoshka'
BASE_BACKBONE = 'jhu-clsp/mmbert-base'
FALLBACK_BACKBONE = 'paraphrase-multilingual-MiniLM-L12-v2'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backbone', default=DEFAULT_BACKBONE,
                        help=f"Sentence-Transformer checkpoint or {BASE_BACKBONE!r} (wrapped raw base); "
                             f"falls back to {FALLBACK_BACKBONE!r} on load failure")
    parser.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--label-column', default='stereotype',
                        help="Label column: 'stereotype' (Task A) or 'hate_speech' (aux files)")
    parser.add_argument('--max-len', type=int, default=MAX_LEN,
                        help='Word budget used by the same PromptBudget as train.py')
    parser.add_argument('--max-seq-length', type=int, default=384,
                        help='Tokenizer truncation cap passed to the SentenceTransformer body')
    parser.add_argument('--head', choices=('torch', 'lr'), default='torch',
                        help="Classifier head: 'torch' SetFitHead (epoch-loop trained with CrossEntropyLoss) "
                             "or 'lr' sklearn LogisticRegression (single fit; epochs/batch ignored)")
    parser.add_argument('--contrastive-epochs', type=int, default=1,
                        help='Embedding-phase epochs (contrastive fine-tuning of the body)')
    parser.add_argument('--num-iterations', type=int, default=5,
                        help='SetFit contrastive pair-generating passes for CosineSimilarityLoss (>= 1); '
                             'ignored with --head-only')
    parser.add_argument('--head-epochs', type=int, default=MAX_EPOCHS,
                        help='Epochs for the torch classifier head (ignored with --head lr)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Classifier-phase batch size (torch head only)')
    parser.add_argument('--contrastive-batch-size', type=int, default=16,
                        help='Embedding-phase batch size (contrastive pairs)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Body learning rate (embedding contrastive phase)')
    parser.add_argument('--head-learning-rate', type=float, default=1e-2,
                        help='Learning rate for the torch classifier head')
    parser.add_argument('--lr-c', type=float, default=1.0,
                        help='Inverse regularization strength for the logistic-regression head (--head lr)')
    parser.add_argument('--lr-max-iter', type=int, default=1000,
                        help='Max solver iterations for the logistic-regression head (--head lr)')
    parser.add_argument('--head-only', action='store_true',
                        help='Skip the contrastive embedding phase; train only the classifier head on frozen '
                             '(pretrained) embeddings -> the non-contrastive control')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out', default=None,
                        help='Directory written by save_pretrained (default: weights/setfit_s<seed>)')
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (default: fp16 on CUDA)')
    parser.add_argument('--smoke', type=int, default=0,
                        help='Train on a random subset of N rows (quick sanity run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration and split sizes, then exit without torch/setfit')
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = f'weights/setfit_s{args.split_seed}'
    if args.num_iterations < 1 and not args.head_only:
        parser.error('--num-iterations must be >= 1')
    if args.head_epochs < 1:
        parser.error('--head-epochs must be >= 1')
    if args.smoke and args.smoke < 8:
        parser.error('--smoke must be >= 8 rows')
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)


def resolve_device(name):
    if name:
        return name
    return 'cuda' if _torch_available() else 'cpu'


def _torch_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def load_deps():
    """Imports torch, setfit/datasets and the local modules.

    Everything that needs a GPU environment is deferred so --dry-run and --help
    work on a machine without torch installed.
    """
    import torch
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as data_mod
    return torch, np, pd, data_mod


def build_sentence_model(args):
    from sentence_transformers import SentenceTransformer, models

    backbone = args.backbone
    if backbone == BASE_BACKBONE:
        st = SentenceTransformer(modules=[
            models.Transformer(BASE_BACKBONE),
            models.Pooling(768, pooling_mode='mean'),
        ])
    else:
        try:
            st = SentenceTransformer(backbone)
        except Exception as err:
            if backbone == FALLBACK_BACKBONE:
                raise
            print(f'WARNING: loading {backbone!r} failed ({err}); '
                  f'falling back to {FALLBACK_BACKBONE!r}')
            backbone = FALLBACK_BACKBONE
            st = SentenceTransformer(backbone)
    st.max_seq_length = args.max_seq_length

    import torch
    for module in st:
        try:
            module.to(torch.float32)  # hub may ship bf16; T4 cannot compute bf16
        except Exception:
            pass
    print(f'Backbone: {backbone} | max_seq_length: {st.max_seq_length} | '
          f'embedding dim: {st.get_sentence_embedding_dimension()} | dtype: fp32')
    return st, backbone


def build_setfit_model(args, st, device):
    if args.head == 'torch':
        from setfit import SetFitHead, SetFitModel

        head = SetFitHead(
            in_features=st.get_sentence_embedding_dimension(),
            out_features=2,
            device=device,
        )
        return SetFitModel(model_body=st, model_head=head)

    from sklearn.linear_model import LogisticRegression
    from setfit import SetFitModel

    head = LogisticRegression(C=args.lr_c, max_iter=args.lr_max_iter)
    return SetFitModel(model_body=st, model_head=head)


def build_texts(df, data_mod, max_len: int) -> list:
    from config import PromptBudget

    budget = PromptBudget(max_len=max_len)
    texts = []
    for _, row in df.iterrows():
        context = data_mod.build_context(row['yt_comment'], row['yt_title'], row['yt_description'])
        texts.append(data_mod.truncate_words(context, budget.total_words))
    return texts


def f1_metric(y_pred, y_test):
    from sklearn.metrics import f1_score
    return {'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0)}


def make_training_args(args, amp: bool):
    from setfit import TrainingArguments

    return TrainingArguments(
        output_dir=os.path.join(os.path.dirname(args.out) or '.', 'setfit_checkpoints'),
        batch_size=(args.contrastive_batch_size, args.batch_size),
        num_epochs=(args.contrastive_epochs, args.head_epochs),
        num_iterations=args.num_iterations,
        body_learning_rate=(args.learning_rate, args.learning_rate),
        head_learning_rate=args.head_learning_rate,
        seed=args.seed,
        use_amp=amp,
        eval_strategy='no',
        logging_strategy='no',
        save_strategy='no',
        report_to='none',
        show_progress_bar=True,
    )


def make_trainer(args, model, train_texts, train_labels, amp):
    from datasets import Dataset
    from setfit import Trainer

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': list(train_labels)})
    return Trainer(
        model=model,
        args=make_training_args(args, amp),
        train_dataset=train_dataset,
    )


def evaluate(model, texts, labels):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    proba = np.asarray(model.predict_proba(texts))
    if proba.ndim == 1 or proba.shape[1] == 1:
        preds = (proba[:, 0] >= 0.5).astype(int)
    elif proba.shape[1] == 2:
        preds = proba.argmax(axis=1)
    else:
        raise RuntimeError(
            f'Unexpected predict_proba shape {proba.shape}; expected binary (n, 2). '
            'Run the smoke test first.'
        )
    return {
        'acc': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_yes': f1_score(labels, preds, pos_label=1, zero_division=0),
        'precision_yes': precision_score(labels, preds, pos_label=1, zero_division=0),
        'recall_yes': recall_score(labels, preds, pos_label=1, zero_division=0),
        'preds': preds,
        'labels': labels,
    }


def main():
    args = parse_args()

    if args.dry_run:
        print('DRY RUN: no torch/setfit loaded.')
        print(f'  backbone: {args.backbone} | max_seq_length: {args.max_seq_length} | '
              f'max_len (word budget): {args.max_len}')
        print(f'  languages: {args.languages or "EN IT NL"} | split_seed: {args.split_seed} | '
              f'head: {args.head} | head_only: {args.head_only}')
        print(f'  contrastive: {args.contrastive_epochs} epoch(s) x {args.num_iterations} iters '
              f'(batch {args.contrastive_batch_size}) | head: {args.head_epochs} epoch(s) '
              f'(batch {args.batch_size}, lr {args.head_learning_rate})')
        print('  Run --smoke on the GPU host to verify the full path.')
        return

    device = resolve_device(args.device)
    if not _torch_available():
        raise SystemExit('No CUDA torch found (run from the Kaggle/Colab notebook).')
    amp = not args.no_amp and device == 'cuda'
    print(f'SetFit baseline | device: {device} | mixed precision: {"ON (fp16)" if amp else "off"}')
    print(f'languages: {args.languages if args.languages else "EN IT NL"} | split_seed: {args.split_seed} | '
          f'out: {args.out} | head: {args.head} | head_only: {args.head_only}')

    torch, np, pd, data_mod = load_deps()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = data_mod.load_frame(languages=args.languages, max_len=args.max_len,
                                 label_column=args.label_column)
    if args.smoke:
        df_all = df_all.sample(n=min(args.smoke, len(df_all)), random_state=args.seed).reset_index(drop=True)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    print(f'Train: {len(df_train)} | Val: {len(df_val)} | '
          f'labels: {data_mod.label_distribution(df_train)}')

    train_texts = build_texts(df_train, data_mod, args.max_len)
    val_texts = build_texts(df_val, data_mod, args.max_len)
    val_labels = df_val['st_y'].values

    st, used_backbone = build_sentence_model(args)
    model = build_setfit_model(args, st, device)
    model.to(device)

    train_labels = df_train['st_y'].values
    started = time.time()
    if args.head_only:
        print('HEAD-ONLY: skipping the contrastive phase; training the classifier head on frozen '
              'pretrained embeddings (the non-contrastive control).')
        model.fit(train_texts, list(train_labels),
                  num_epochs=args.head_epochs, batch_size=args.batch_size,
                  head_learning_rate=args.head_learning_rate,
                  show_progress_bar=True, end_to_end=False)
    else:
        trainer = make_trainer(args, model, train_texts, train_labels, amp)
        trainer.train()
    print(f'fit finished in {time.time() - started:.0f}s')

    metrics = evaluate(model, val_texts, val_labels)
    print("--- VALIDATION (after fit) ---")
    for key, value in metrics.items():
        if key not in ('preds', 'labels'):
            print(f'  {key}: {value:.4f}')

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    history = [{
        'epoch': args.head_epochs,
        'train_loss': '',
        'val_macro_f1': f"{metrics['macro_f1']:.6f}",
        'val_acc': f"{metrics['acc']:.6f}",
    }]
    history_path = f'{args.out}_history.csv'
    with open(history_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f'\nSaved model -> {args.out} (save_pretrained)')
    print(f'Wrote metrics -> {history_path}')

    print('--- PER LANGUAGE (val) ---')
    for lang in sorted(df_val['lang'].unique()):
        rows = df_val['lang'] == lang
        lm = evaluate(model, [t for t, keep in zip(val_texts, rows) if keep],
                      df_val.loc[rows, 'st_y'].values)
        print(f"  [{lang}] acc {lm['acc']:.4f} | macro_f1 {lm['macro_f1']:.4f}")

    if args.smoke:
        from setfit import SetFitModel
        print('Reload check: SetFitModel.from_pretrained ->')
        reloaded = SetFitModel.from_pretrained(args.out)
        reloaded.to('cpu')
        proba = np.asarray(reloaded.predict_proba(val_texts[: min(8, len(val_texts))]))
        print(f'  predict_proba on {proba.shape[0]} val rows -> shape {proba.shape}')


if __name__ == '__main__':
    main()