#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SetFit contrastive baseline for Task A (stereotype detection).

Non-prompt control for the prompt-MLM pipeline (train.py). Same mmBERT family,
same video-grouped split, same seeds, same macro-F1, but the model is SetFit:
contrastive fine-tuning of a sentence-embedding backbone followed by a small
head trained on the frozen embeddings. The text is fed WITHOUT the
question/answer [MASK] framing, isolating whether the prompt vehicle adds value.

Backbone: `vllm-sr/mmbert-embed-32k-2d-matryoshka` (mmBERT-Embed, a fine-tune of
`jhu-clsp/mmbert-base`), which loads directly as a SentenceTransformer.
`--backbone jhu-clsp/mmbert-base` wraps the raw base with mean pooling;
load failures auto-fall-back to multilingual MiniLM.

Usage:
    python src/run_setfit.py --split-seed 40 --out weights/setfit_s40.pt
    python src/run_setfit.py --dry-run            # config only, no torch/setfit
    python src/run_setfit.py --smoke 256 --num-iterations 1 --epochs 2
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
    PATIENCE,
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
                        help='Tokenizer truncation cap passed to the SentenceTransformer')
    parser.add_argument('--num-iterations', type=int, default=5,
                        help='SetFit contrastive iterations (0 disables the contrastive phase)')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help='Head-training epochs (early stopping on val macro-F1)')
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out', default=None, help='Checkpoint path (default: weights/setfit_s<seed>.pt)')
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (default: fp16 on CUDA)')
    parser.add_argument('--smoke', type=int, default=0,
                        help='Train on a random subset of N rows (quick sanity run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration and split sizes, then exit without torch/setfit')
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = f'weights/setfit_s{args.split_seed}.pt'
    if args.num_iterations < 0:
        parser.error('--num-iterations must be >= 0')
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
    """Imports torch, setfit, sentence-transformers and the local modules.

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


def build_texts(df, data_mod, max_len: int) -> list:
    from config import PromptBudget

    budget = PromptBudget(max_len=max_len)
    texts = []
    for _, row in df.iterrows():
        context = data_mod.build_context(row['yt_comment'], row['yt_title'], row['yt_description'])
        texts.append(data_mod.truncate_words(context, budget.total_words))
    return texts


def f1_metric(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def make_trainer(args, model, train_df, val_df, amp: bool):
    import inspect
    from setfit import SetFitTrainer

    signature = inspect.signature(SetFitTrainer.__init__)
    params = set(signature.parameters)
    kwargs = {
        'model': model,
        'train_dataset': train_df,
        'eval_dataset': val_df,
        'column_mapping': {'text': 'text', 'label': 'label'},
        'metric': f1_metric,
        'num_iterations': args.num_iterations,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
    }
    # Arg names drift across setfit releases; send only what this version knows.
    # Groups that cannot be honoured are reported loudly instead of silently
    # changing the experiment (e.g. no early stopping without the callback).
    optional = {
        'num_epochs': ('num_epochs', 'num_epochs_head', args.epochs),
        'contrastive_epochs': ('num_epochs_contrastive', 1),
        'batch_size': ('batch_size', 'batch_size_head', 'batch_size_contrastive', args.batch_size),
        'amp': ('use_amp', 'fp16', amp),
        'early_stop': ('early_stopping_patience', args.patience),
        'early_stop_threshold': ('early_stopping_threshold', 1e-4),
    }
    applied, missing = [], []
    for group, spec in optional.items():
        names, value = spec[:-1], spec[-1]
        chosen = next((n for n in names if n in params), None)
        if chosen is not None:
            kwargs[chosen] = value
            applied.append(f'{group}={chosen}')
        else:
            missing.append(group)
    print(f'SetFitTrainer options applied ({len(applied)}): {", ".join(applied)}')
    if missing:
        print(f'WARNING: installed setfit ignores: {", ".join(missing)}')
    return SetFitTrainer(**kwargs)


def evaluate(model, texts, labels):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    proba = model.predict_proba(texts)
    proba = np.asarray(proba)
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
              f'num_iterations: {args.num_iterations} | head epochs: {args.epochs} | '
              f'patience: {args.patience} | batch_size: {args.batch_size}')
        print('  Run --smoke on the GPU host to verify the full path.')
        return

    device = resolve_device(args.device)
    if not _torch_available():
        raise SystemExit('No CUDA torch found (run from the Kaggle/Colab notebook).')
    amp = not args.no_amp and device == 'cuda'
    print(f'SetFit baseline | device: {device} | mixed precision: {"ON (fp16)" if amp else "off"}')
    print(f'languages: {args.languages if args.languages else "EN IT NL"} | split_seed: {args.split_seed} | '
          f'out: {args.out} | contrastive iters: {args.num_iterations} | head epochs: {args.epochs}')

    torch, np, pd, data_mod = load_deps()
    if args.num_iterations == 0:
        print('Contrastive phase disabled (--num-iterations 0): head-only training.')

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
    train_df = pd.DataFrame({'text': train_texts, 'label': df_train['st_y'].values})
    val_df = pd.DataFrame({'text': val_texts, 'label': df_val['st_y'].values})
    val_labels = val_df['label'].values

    from setfit import SetFitModel

    st, used_backbone = build_sentence_model(args)
    model = SetFitModel.from_pretrained(st)
    model.to(device)

    trainer = make_trainer(args, model, train_df, val_df, amp)
    started = time.time()
    trainer.fit()
    print(f'fit finished in {time.time() - started:.0f}s')

    metrics = evaluate(model, val_texts, val_labels)
    print("--- VALIDATION (after fit) ---")
    for key, value in metrics.items():
        if key not in ('preds', 'labels'):
            print(f'  {key}: {value:.4f}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'backbone': used_backbone, 'split_seed': args.split_seed,
                'model': model.state_dict()}, args.out)
    history = [{
        'epoch': 1,
        'train_loss': '',
        'val_macro_f1': f"{metrics['macro_f1']:.6f}",
        'val_acc': f"{metrics['acc']:.6f}",
    }]
    history_path = os.path.splitext(args.out)[0] + '_history.csv'
    with open(history_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f'\nSaved checkpoint -> {args.out}')
    print(f'Wrote metrics -> {history_path}')

    print('--- PER LANGUAGE (val) ---')
    for lang in sorted(df_val['lang'].unique()):
        rows = df_val['lang'] == lang
        lm = evaluate(model, [t for t, keep in zip(val_texts, rows) if keep],
                      df_val.loc[rows, 'st_y'].values)
        print(f"  [{lang}] acc {lm['acc']:.4f} | macro_f1 {lm['macro_f1']:.4f}")


if __name__ == '__main__':
    main()