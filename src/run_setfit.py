#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SetFit contrastive baseline for Task A (stereotype detection).

Non-prompt control for the prompt-MLM pipeline (train.py). Same mmBERT family,
same video-grouped split, same seeds, same macro-F1, but the model is SetFit:
contrastive fine-tuning of a sentence-embedding backbone followed by a small
head. The text is fed WITHOUT the question/answer [MASK] framing, isolating
whether the prompt vehicle adds value.

The modern SetFit v2 API is used:
  * `SetFitModel(model_body=<SentenceTransformer>, model_head=<head>)`
  * `Trainer` + `TrainingArguments` + `datasets.Dataset`
  * contrastive embedding phase via `Trainer.train_embeddings(...)`
  * classifier head trained separately so each epoch logs validation macro-F1:
      - `--head torch`: a `SetFitHead` (CrossEntropyLoss) trained for up to
        `--head-epochs` epochs with `--patience` early stopping; the best-epoch
        head is restored and saved. This is implemented in this script (not by
        the library), so the per-epoch history is real.
      - `--head lr`: sklearn `LogisticRegression` (single fit; epochs, batch
        size and patience are N/A, tune `--lr-c` / `--lr-max-iter` instead).
  * `--head-only` calls the head phase directly on frozen (pretrained) body
    embeddings with NO contrastive phase -- the non-contrastive control.
  * `model.save_pretrained(<dir>)` / `SetFitModel.from_pretrained(<dir>)`.

`--label-column` is honoured by `data.load_frame`: it encodes the chosen column
into the binary `st_y` (stereotype yes/no, or hate_speech none/no/yes_implicit/
yes_explicit -> 0/1). The head width tracks the number of classes in `st_y`,
and the evaluator is multi-class, so the same script can score a 3-class label
if such an encoder is added.

Outputs:
  * `<out>/`            - SetFit model via save_pretrained
  * `<out>_history.csv` - per-epoch val macro-F1 / acc (real epochs for the
                          torch head; one row for the LR head)
  * `<out>_run.json`    - full experiment metadata (versions, backbone, seeds,
                          hyper-parameters, label mapping, token audit,
                          final + per-language metrics)

Backbone: `vllm-sr/mmbert-embed-32k-2d-matryoshka` (mmBERT-Embed, a fine-tune
of `jhu-clsp/mmbert-base`), which loads directly as a SentenceTransformer.
`--backbone jhu-clsp/mmbert-base` wraps the raw base with mean pooling;
load failures auto-fall-back to multilingual MiniLM. The word budget
(`--max-len`) is only an estimate: a tokenizer can still produce more tokens
than `--max-seq-length`, so the script audits post-tokenizer lengths per
language and warns about rows that hit the tokenizer cap.

Usage:
    python src/run_setfit.py --split-seed 40 --seed 40 --out weights/setfit_s40_seed40
    python src/run_setfit.py --dry-run            # config only, no torch/setfit
    python src/run_setfit.py --smoke 256 --num-iterations 1 --head-epochs 2
    python src/run_setfit.py --head lr --head-only --out weights/setfit_s40_lr
"""

import argparse
import csv
import json
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
PATIENCE = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backbone', default=DEFAULT_BACKBONE,
                        help=f"Sentence-Transformer checkpoint or {BASE_BACKBONE!r} (wrapped raw base); "
                             f"falls back to {FALLBACK_BACKBONE!r} on load failure")
    parser.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    parser.add_argument('--label-column', default='stereotype',
                        help="Label column encoded into st_y by data.load_frame: "
                             "'stereotype' (Task A) or 'hate_speech' (aux files)")
    parser.add_argument('--max-len', type=int, default=MAX_LEN,
                        help='Word budget used by the same PromptBudget as train.py')
    parser.add_argument('--max-seq-length', type=int, default=384,
                        help='Tokenizer truncation cap passed to the SentenceTransformer body')
    parser.add_argument('--head', choices=('torch', 'lr'), default='torch',
                        help="Classifier head: 'torch' SetFitHead (per-epoch training, --head-epochs + "
                             "--patience early stopping) or 'lr' sklearn LogisticRegression "
                             "(single fit, tuned via --lr-c / --lr-max-iter)")
    parser.add_argument('--contrastive-epochs', type=int, default=1,
                        help='Embedding-phase epochs (contrastive fine-tuning of the body)')
    parser.add_argument('--num-iterations', type=int, default=5,
                        help='SetFit contrastive pair-generating passes for CosineSimilarityLoss (>= 1); '
                             'ignored with --head-only')
    parser.add_argument('--head-epochs', type=int, default=MAX_EPOCHS,
                        help='Maximum epochs for the torch classifier head (ignored with --head lr)')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early-stopping patience on validation macro-F1 for the torch head '
                             '(0 or -1 disables early stopping)')
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Training seed (defaults to --split-seed)')
    parser.add_argument('--out', default=None,
                        help='Directory written by save_pretrained '
                             '(default: weights/setfit_s<split_seed>_seed<seed>)')
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (default: fp16 on CUDA)')
    parser.add_argument('--smoke', type=int, default=0,
                        help='Train on a random subset of N rows (quick sanity run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration and split sizes, then exit without torch/setfit')
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = args.split_seed
    if args.out is None:
        args.out = f'weights/setfit_s{args.split_seed}_seed{args.seed}'
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


def build_setfit_model(args, st, device, n_classes: int):
    if args.head == 'torch':
        from setfit import SetFitHead, SetFitModel

        head = SetFitHead(
            in_features=st.get_sentence_embedding_dimension(),
            out_features=n_classes,
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


def make_trainer(args, model, training_args):
    from setfit import Trainer

    return Trainer(model=model, args=training_args)


def predict_labels(model, texts):
    import numpy as np

    preds = model.predict(list(texts), use_labels=False)
    if hasattr(preds, 'tolist'):
        preds = preds.tolist()
    return np.asarray(preds, dtype=int)


def evaluate(model, texts, labels):
    """Multi-class evaluation. Handles (n,), (n,1) and (n,k>=2) proba shapes."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    proba = np.asarray(model.predict_proba(list(texts)), dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    elif proba.ndim == 2 and proba.shape[1] == 1:
        proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError(
            f'Unexpected predict_proba shape {proba.shape}; expected >= 2 class columns. '
            'Run the smoke test first.'
        )
    preds = proba.argmax(axis=1)

    metrics = {
        'acc': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'preds': preds,
        'labels': labels,
    }
    if proba.shape[1] == 2:
        metrics['f1_yes'] = float(
            f1_score(labels, preds, labels=[1], average=None, zero_division=0)[0])
    for cls in sorted(set(int(c) for c in labels)):
        metrics[f'f1_class{cls}'] = float(
            f1_score(labels, preds, labels=[cls], average=None, zero_division=0)[0])
    return metrics


def train_classifier_phase(args, model, train_texts, train_labels,
                           val_texts, val_labels, device):
    """Trains only the classifier head on frozen body embeddings.

    torch head: per-epoch loop logging validation macro-F1/acc, with
    `--patience` early stopping and best-epoch head restore. Returns the real
    per-epoch history and the best epoch.
    lr head: single sklearn fit; returns an empty history (one metrics row is
    appended by the caller).
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    if args.head == 'lr':
        print(f'Training {args.head} head: single LogisticRegression fit '
              '(epochs/patience/batch are N/A; tune C and max_iter).')
        model.fit(list(train_texts), list(train_labels), num_epochs=1,
                  show_progress_bar=True, end_to_end=False)
        return [], 0

    print(f'Training torch head: up to {args.head_epochs} epoch(s), '
          f'patience {args.patience}, batch {args.batch_size}, '
          f'lr {args.head_learning_rate}.')
    model.freeze('body')
    dataloader = model._prepare_dataloader(
        list(train_texts), list(train_labels), args.batch_size, args.max_seq_length)
    criterion = model.model_head.get_loss_fn()
    optimizer = model._prepare_optimizer(args.head_learning_rate, None, 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history, best_macro, best_state, best_epoch, wait = [], -1.0, None, 0, 0
    for epoch in range(1, args.head_epochs + 1):
        model.model_body.train()
        model.model_head.train()
        total_loss, n_batches = 0.0, 0
        for features, labels in dataloader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.model_body(features)
            if model.normalize_embeddings:
                outputs['sentence_embedding'] = torch.nn.functional.normalize(
                    outputs['sentence_embedding'], p=2, dim=1)
            outputs = model.model_head(outputs)
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        model.model_body.eval()
        model.model_head.eval()
        preds = predict_labels(model, val_texts)
        macro = float(f1_score(val_labels, preds, average='macro', zero_division=0))
        acc = float(accuracy_score(val_labels, preds))
        row = {
            'epoch': epoch,
            'train_loss': f'{total_loss / max(n_batches, 1):.4f}',
            'val_macro_f1': f'{macro:.6f}',
            'val_acc': f'{acc:.6f}',
        }
        history.append(row)
        print(f"  epoch {epoch:02d}: train_loss {row['train_loss']} | "
              f"val_macro_f1 {macro:.4f} | val_acc {acc:.4f}")
        if macro > best_macro:
            best_macro, best_epoch, wait = macro, epoch, 0
            best_state = {name: p.detach().clone()
                          for name, p in model.model_head.named_parameters()}
        else:
            wait += 1
            if args.patience > 0 and wait >= args.patience:
                print(f'  early stop at epoch {epoch} (patience {args.patience}); '
                      f'best = epoch {best_epoch} ({best_macro:.4f})')
                break

    if best_state is not None:
        with torch.no_grad():
            for name, p in model.model_head.named_parameters():
                p.copy_(best_state[name])
        print(f'Restored best head from epoch {best_epoch} '
              f'(val_macro_f1 {best_macro:.4f})')
    model.unfreeze('body')
    return history, best_epoch


def audit_token_lengths(val_texts, langs, st, max_seq_length: int):
    """Post-tokenizer length check: the word budget is only an estimate."""
    import numpy as np

    tokenizer = st.tokenizer
    rows = {}
    total_over = 0
    for text, lang in zip(val_texts, langs):
        n = len(tokenizer(text, add_special_tokens=True)['input_ids'])
        rows.setdefault(lang, []).append(n)
        if n > max_seq_length:
            total_over += 1

    stats = {}
    print(f'Token-length audit on {len(val_texts)} val texts '
          f'(tokenizer cap {max_seq_length}):')
    for lang, lens in sorted(rows.items()):
        a = np.asarray(lens, dtype=float)
        over = int((a > max_seq_length).sum())
        stats[lang] = {
            'n': int(len(a)),
            'mean': round(float(a.mean()), 1),
            'p95': round(float(np.percentile(a, 95)), 1),
            'max': int(a.max()),
            'n_over_cap': over,
        }
        print(f"  [{lang}] n={len(a)}: mean {a.mean():.0f} | "
              f"p95 {np.percentile(a, 95):.0f} | max {a.max():.0f} | "
              f'>{max_seq_length}: {over}')
    if total_over:
        print(f'WARNING: {total_over}/{len(val_texts)} val texts exceed the tokenizer cap '
              f'{max_seq_length} and are right-truncated by the tokenizer. Consider lowering '
              '--max-len for this backbone.')
    return stats


def _dump_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)


def write_run_metadata(args, meta_extra: dict, path: str) -> None:
    import sklearn
    import sentence_transformers
    import setfit
    import torch
    import transformers

    meta = {
        'script': 'run_setfit.py',
        'setfit_version': setfit.__version__,
        'sentence_transformers_version': sentence_transformers.__version__,
        'transformers_version': transformers.__version__,
        'torch_version': torch.__version__,
        'sklearn_version': sklearn.__version__,
        'languages': args.languages or ['EN', 'IT', 'NL'],
        'label_column': args.label_column,
        'max_len': args.max_len,
        'max_seq_length': args.max_seq_length,
        'split_seed': args.split_seed,
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'head': args.head,
        'head_only': args.head_only,
        'backbone': args.backbone,
        'contrastive_epochs': args.contrastive_epochs,
        'num_iterations': args.num_iterations if not args.head_only else None,
        'head_epochs': args.head_epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'contrastive_batch_size': args.contrastive_batch_size,
        'learning_rate': args.learning_rate,
        'head_learning_rate': args.head_learning_rate,
        'lr_c': args.lr_c,
        'lr_max_iter': args.lr_max_iter,
        'device': args.device or 'auto',
        'amp': not args.no_amp,
    }
    meta.update(meta_extra)
    _dump_json(path, meta)


def main():
    args = parse_args()

    if args.dry_run:
        print('DRY RUN: no torch/setfit loaded.')
        print(f'  backbone: {args.backbone} | max_seq_length: {args.max_seq_length} | '
              f'max_len (word budget): {args.max_len} | label_column: {args.label_column}')
        print(f'  languages: {args.languages or "EN IT NL"} | split_seed: {args.split_seed} | '
              f'seed: {args.seed} | head: {args.head} | head_only: {args.head_only} | '
              f'out: {args.out}')
        print(f'  contrastive: {args.contrastive_epochs} epoch(s) x {args.num_iterations} iters '
              f'(batch {args.contrastive_batch_size}) | head: {args.head_epochs} epoch(s) '
              f'(batch {args.batch_size}, lr {args.head_learning_rate}, patience {args.patience})')
        print('  Run --smoke on the GPU host to verify the full path.')
        return

    device = resolve_device(args.device)
    if not _torch_available():
        raise SystemExit('No CUDA torch found (run from the Kaggle/Colab notebook).')
    amp = not args.no_amp and device == 'cuda'
    print(f'SetFit baseline | device: {device} | mixed precision: {"ON (fp16)" if amp else "off"}')
    print(f'languages: {args.languages if args.languages else "EN IT NL"} | '
          f'split_seed: {args.split_seed} | seed: {args.seed} | out: {args.out} | '
          f'head: {args.head} | head_only: {args.head_only}')

    torch, np, pd, data_mod = load_deps()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = data_mod.load_frame(languages=args.languages, max_len=args.max_len,
                                 label_column=args.label_column)
    if args.smoke:
        df_all = df_all.sample(n=min(args.smoke, len(df_all)),
                               random_state=args.seed).reset_index(drop=True)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    n_classes = int(df_train['st_y'].nunique())
    dist = {str(k): int(v) for k, v in df_train['st_y'].value_counts().items()}
    print(f'Train: {len(df_train)} | Val: {len(df_val)} | st_y classes: {n_classes} | '
          f'train label counts: {dist}')

    train_texts = build_texts(df_train, data_mod, args.max_len)
    val_texts = build_texts(df_val, data_mod, args.max_len)
    val_labels = df_val['st_y'].values
    train_labels = df_train['st_y'].values

    st, used_backbone = build_sentence_model(args)
    token_audit = audit_token_lengths(val_texts, df_val['lang'].values, st, args.max_seq_length)

    model = build_setfit_model(args, st, device, n_classes)
    model.to(device)

    started = time.time()
    if args.head_only:
        print('HEAD-ONLY: skipping the contrastive phase; training the classifier head on frozen '
              'pretrained embeddings (the non-contrastive control).')
    else:
        training_args = make_training_args(args, amp)
        trainer = make_trainer(args, model, training_args)
        print(f'Contrastive embedding phase: {args.contrastive_epochs} epoch(s) x '
              f'{args.num_iterations} iterations, batch {args.contrastive_batch_size}, '
              f'lr {args.learning_rate}, amp {"on" if amp else "off"}.')
        trainer.train_embeddings(train_texts, list(train_labels), args=training_args)
        print('Contrastive phase done.')

    history, best_epoch = train_classifier_phase(
        args, model, train_texts, train_labels, val_texts, val_labels, device)
    print(f'Classifier phase done in {time.time() - started:.0f}s')

    metrics = evaluate(model, val_texts, val_labels)
    print('--- VALIDATION (best-epoch head) ---')
    for key, value in metrics.items():
        if key not in ('preds', 'labels'):
            print(f'  {key}: {value:.4f}')

    per_lang = {}
    for lang in sorted(df_val['lang'].unique()):
        rows = df_val['lang'] == lang
        lm = evaluate(model, [t for t, keep in zip(val_texts, rows) if keep],
                      df_val.loc[rows, 'st_y'].values)
        per_lang[lang] = {k: round(float(v), 6)
                          for k, v in lm.items() if k not in ('preds', 'labels')}
        print(f"  [{lang}] acc {lm['acc']:.4f} | macro_f1 {lm['macro_f1']:.4f}")

    if not history:
        history = [{
            'epoch': 1,
            'train_loss': '',
            'val_macro_f1': f"{metrics['macro_f1']:.6f}",
            'val_acc': f"{metrics['acc']:.6f}",
        }]

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    history_path = f'{args.out}_history.csv'
    with open(history_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    write_run_metadata(args, {
        'model_id': used_backbone,
        'n_classes': n_classes,
        'train_size': int(len(df_train)),
        'val_size': int(len(df_val)),
        'train_label_counts': dist,
        'val_label_counts': {str(k): int(v) for k, v in df_val['st_y'].value_counts().items()},
        'best_epoch': best_epoch,
        'val_metrics': {k: round(float(v), 6)
                        for k, v in metrics.items() if k not in ('preds', 'labels')},
        'per_language': per_lang,
        'token_audit': token_audit,
    }, f'{args.out}_run.json')

    print(f'\nSaved model -> {args.out} (save_pretrained)')
    print(f'Wrote history -> {history_path}')
    print(f'Wrote metadata -> {args.out}_run.json')

    if args.smoke:
        from setfit import SetFitModel
        print('Reload check: SetFitModel.from_pretrained ->')
        reloaded = SetFitModel.from_pretrained(args.out)
        reloaded.to('cpu')
        proba = np.asarray(reloaded.predict_proba(val_texts[: min(8, len(val_texts))]))
        print(f'  predict_proba on {proba.shape[0]} val rows -> shape {proba.shape}')


if __name__ == '__main__':
    main()