#!/usr/bin/env python3
"""pooled_v diagnostics: does the head use the shortcut, and what does it cost?

Runs on a trained checkpoint only - no retraining, a handful of forward passes:

    python src/diag_pooled_v.py --checkpoint weights/xattn_sep_u2_l3_s42.pt
    python src/diag_pooled_v.py --checkpoint ... --json weights/diag_pv.json

**A. quality** - macro-F1 / acc / AUC overall and per language.

**B. Inference-time ablation.** Re-run *only the head* with the pooled-v half of
its input zeroed, permuted within a language, or shuffled globally. The encoders
are not re-run, so this is exact and ~free, and it needs no second training run.

- `pooled_v := 0` asks whether the shortcut is load-bearing at all.
- The permutations keep pooled_v's marginal distribution and only break its
  pairing with the comment, so they separate "the head *uses* the video content"
  from "the head merely tolerates the extra 384 inputs".
- A model that has learned to ignore pooled_v scores the same on all three.

What B does *not* say: it measures the reliance of a model that was **trained with**
pooled_v, so it cannot show whether training with or without the branch changes the
final metric. That question needs the train-time arm (both variants, several seeds,
same video split, mean +/- spread) - not this section.

**C. Representation.** The cross-attention entropy over the video keys (is the
attention peaked or diffuse?) and the size of the two pooled vectors. Entropy and
attention maps localise *where* the model looks; they are not, on their own,
evidence that the model causally relies on the video. Use E for that.

**D. Probes.** A logistic regression on the *frozen* pooled vectors, fitted on
train rows and scored on val: how much label information does each path carry on
its own? Together with C this tests the pre-registered prediction that removing
`pooled_v` forces the cross-attention to work harder, i.e. entropy down and the
`pooled_c` probe up.

**E. Video-context perturbation.** Change the video *input* and re-run the whole
model: permute videos within / across languages, or replace every unmasked video
token with a constant. Every arm keeps the per-row mask and sequence length, so
only the content changes. Because the paired video is destroyed while the comment
stays well-formed, a macro-F1 drop is a direct, causal statement that the model
uses the paired video - which is exactly what B and C cannot establish. It is only
as specific as the checkpoint is trained (an under-trained model can move its output
under any input change), so it is meant to run on the same checkpoint the other
sections do, and its `as-is` arm is checked against section A.

`--probe-train-rows` sets how many TRAIN rows get encoded for the probe fit
(0 skips section D and the extra encoder pass). `--max-val-rows` truncates val
and is only a debug aid.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
import model_xattn
from config import SPLIT_SEED, VAL_FRACTION
from model import load_tokenizer
from train_xattn import make_loader


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=os.path.join('weights', 'best_xattn_stereotype.pt'))
    p.add_argument('--backbone', default=model_xattn.DEFAULT_BACKBONE)
    p.add_argument('--languages', nargs='*', default=None, help='Subset of EN/IT/NL (default: all)')
    p.add_argument('--label-column', default='stereotype')
    p.add_argument('--max-len', type=int, default=160)
    p.add_argument('--video-max-len', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--heads', type=int, default=4,
                   help='Attention heads. NOT stored in the state dict, so it has to match '
                        'training; it is printed back so a mismatch is at least visible.')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    p.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    p.add_argument('--probe-train-rows', type=int, default=3000, help='0 = skip section D')
    p.add_argument('--max-val-rows', type=int, default=0, help='0 = all (debug aid)')
    p.add_argument('--device', default=None)
    p.add_argument('--json', default=None, help='Also write the numbers to this file')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- io

def load_state(path):
    """State dict out of a checkpoint, tolerating a wrapping dict / DataParallel."""
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not (isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values())):
        raise SystemExit(f'{path}: does not look like a state dict')
    if all(k.startswith('module.') for k in obj):
        obj = {k[len('module.'):]: v for k, v in obj.items()}
    return obj


def infer_arch(state):
    """Architecture flags readable from the state dict alone.

    Only the *forward* architecture matters here: every weight is loaded from the
    checkpoint, so the LR groups, the unfreeze depth and the two-phase schedule
    leave no trace and do not have to be reproduced.
    """
    d = int(state['blocks.0.norm_q.weight'].numel())
    head_in = int(state['head.2.weight'].shape[1])
    layers = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))
    if any(k.startswith('gate.') for k in state):
        fusion = 'gated'
    elif head_in == 2 * d:
        fusion = 'concat'
    else:
        fusion = 'none'
    return {
        'd_model': d,
        'layers': layers,
        'fusion': fusion,
        'use_pooled_v': fusion == 'concat',
        'separate_encoders': any(k.startswith('encoder_v.') for k in state),
    }


def build_from_state(state, backbone, heads, dropout, device):
    arch = infer_arch(state)
    net = model_xattn.build_model(
        backbone, unfreeze_layers=0, heads=heads, layers=arch['layers'], dropout=dropout,
        fusion=arch['fusion'], warm_start=False, separate_encoders=arch['separate_encoders'])
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net, arch


# ---------------------------------------------------------------------- metrics

def metrics(prob_yes, labels):
    """Accuracy, macro-F1, AUC and the yes-class precision/recall."""
    from sklearn.metrics import roc_auc_score

    prob_yes = np.asarray(prob_yes, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    pred = (prob_yes >= 0.5).astype(int)
    out = {'n': int(labels.size), 'acc': float((pred == labels).mean())}
    f1s = []
    for c in (0, 1):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        if c == 1:
            out['precision_yes'], out['recall_yes'] = prec, rec
    out['macro_f1'] = float(np.mean(f1s))
    out['auc'] = (float(roc_auc_score(labels, prob_yes))
                  if len(set(labels.tolist())) > 1 else float('nan'))
    return out


def head_prob_yes(net, features, device):
    """Only the head, re-run on features we built ourselves."""
    with torch.no_grad():
        logits = net.head(features.to(device))
    return torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def predict(net, loader, device):
    """Whole-model forward over a loader -> P(yes). Used by the perturbation test."""
    probs = []
    for cids, cmask, vids, vmask, _ in loader:
        logits = net(cids.to(device), cmask.to(device), vids.to(device), vmask.to(device))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def shifted_index(langs, within_language, seed=0):
    """i -> the index of a different row: same language, or anywhere.

    Within a language the shift is cyclic, so no row keeps its own pooled_v. The
    global case is a random permutation (a handful of fixed points out of ~700
    rows is possible and harmless).
    """
    n = len(langs)
    if not within_language:
        return np.random.default_rng(seed).permutation(n)
    idx = np.arange(n)
    for lang in np.unique(langs):
        rows = np.nonzero(langs == lang)[0]
        if len(rows) > 1:
            idx[rows] = np.roll(rows, 1)
    return idx


# ------------------------------------------------------------------------- main

@torch.no_grad()
def forward_pass(net, loader, device):
    """Encode the set once and keep everything the diagnostics need."""
    probs, labels, pooled_c, pooled_v = [], [], [], []
    n_layer = len(net.blocks)
    ent_sum, ent_cnt, top1_sum = np.zeros(n_layer), np.zeros(n_layer), np.zeros(n_layer)
    pool_ent_sum = pool_n = vlen_sum = 0.0
    head_gap = 0.0

    for cids, cmask, vids, vmask, label in loader:
        cids, cmask = cids.to(device), cmask.to(device)
        vids, vmask = vids.to(device), vmask.to(device)
        logits, aux = net(cids, cmask, vids, vmask, return_attn=True)

        pooled_c.append(aux['pooled_c'].float().cpu())
        pooled_v.append(aux['pooled_v'].float().cpu() if aux['pooled_v'] is not None
                        else torch.zeros_like(aux['pooled_c']))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(label.numpy())

        # The ablation below assumes head(fuse(pooled_c, pooled_v)) is bit-for-bit
        # the model's own forward; check it instead of trusting it.
        feats = net.fuse(aux['pooled_c'], aux['pooled_v'])
        head_gap = max(head_gap, float((net.head(feats) - logits).abs().max()))

        # Per-head entropy / top-1, averaged over the *unmasked* query rows. The
        # count has to include the head axis, or the mean comes out Hx too big.
        q_mask = cmask.unsqueeze(1).float()                      # (B, 1, Lq)
        for layer, w in enumerate(aux['attn']):                  # (B, H, Lq, Lkv)
            p = w.clamp_min(1e-12)
            ent_sum[layer] += float(((-(p * p.log()).sum(-1)) * q_mask).sum())
            top1_sum[layer] += float((w.max(-1).values * q_mask).sum())
            ent_cnt[layer] += float(q_mask.sum()) * w.shape[1]
        # Padded comment tokens already carry probability 0 in the pooling
        # softmax (they were set to -inf), so no masking is needed here.
        pool = aux['pool_weights'].clamp_min(1e-12)
        pool_ent_sum += float((-(pool * pool.log()).sum(-1)).sum())
        pool_n += float(pool.shape[0])
        vlen_sum += float(vmask.sum())

    return {
        'probs': np.concatenate(probs),
        'labels': np.concatenate(labels),
        'pooled_c': torch.cat(pooled_c).numpy(),
        'pooled_v': torch.cat(pooled_v).numpy(),
        'has_pooled_v': net.use_pooled_v,
        'self_check_head_gap': head_gap,
        'entropy_bits': [e / (c * math.log(2)) for e, c in zip(ent_sum, ent_cnt)],
        'top1_weight': [t / c for t, c in zip(top1_sum, ent_cnt)],
        'pool_entropy_bits': pool_ent_sum / (pool_n * math.log(2)),
        'mean_video_tokens': vlen_sum / max(pool_n, 1.0),
    }


@torch.no_grad()
def embed(net, loader, device):
    pc, pv = [], []
    for cids, cmask, vids, vmask, _ in loader:
        _, aux = net(cids.to(device), cmask.to(device), vids.to(device), vmask.to(device),
                     return_attn=True)
        pc.append(aux['pooled_c'].float().cpu())
        pv.append(aux['pooled_v'].float().cpu() if aux['pooled_v'] is not None
                  else torch.zeros_like(aux['pooled_c']))
    return torch.cat(pc), torch.cat(pv)


def probe(train_x, train_y, val_x, val_y):
    """Logistic regression on frozen features -> the usual metrics on val."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def arr(t):
        return t.numpy() if torch.is_tensor(t) else np.asarray(t)

    scaler = StandardScaler().fit(arr(train_x))
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(arr(train_x)), arr(train_y))
    return metrics(clf.predict_proba(scaler.transform(arr(val_x)))[:, 1], arr(val_y))


def sample_rows(df, n, seed=0):
    """Stratified-by-language positional sample, for a cheaper probe fit."""
    if n <= 0 or n >= len(df):
        return np.arange(len(df))
    rng = np.random.default_rng(seed)
    per = max(1, n // df['lang'].nunique())
    idx = []
    for _, group in df.groupby('lang'):
        pos = group.index.to_numpy()
        idx.extend(rng.choice(pos, size=min(per, len(pos)), replace=False).tolist())
    return np.sort(np.asarray(idx))


def main():
    args = parse_args()
    state = load_state(args.checkpoint)
    net, arch = build_from_state(state, args.backbone, args.heads, args.dropout, args.device)
    device = next(net.parameters()).device
    del state

    df_all = data_mod.load_frame(languages=args.languages, label_column=args.label_column)
    df_train, df_val = data_mod.split_by_video(df_all, args.val_fraction, args.split_seed)
    df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)
    if args.max_val_rows:
        df_val = df_val.iloc[:args.max_val_rows].reset_index(drop=True)
    langs = df_val['lang'].to_numpy()

    tokenizer = load_tokenizer(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    print(f'checkpoint          {args.checkpoint}')
    print(f'arch                d={arch["d_model"]} layers={arch["layers"]} heads={args.heads} '
          f'use_pooled_v={arch["use_pooled_v"]} separate_encoders={arch["separate_encoders"]}')
    print(f'data                train {len(df_train)} | val {len(df_val)} '
          f'({", ".join(f"{k} {v}" for k, v in df_val["lang"].value_counts().sort_index().items())})')

    val_loader = make_loader(df_val, tokenizer, args.batch_size, args.max_len, args.video_max_len)
    res = forward_pass(net, val_loader, device)
    print(f'self-check          head(cat[pooled_c, pooled_v]) vs forward logits: '
          f'max |diff| {res["self_check_head_gap"]:.2e}')

    # ---- A. quality --------------------------------------------------------
    overall = metrics(res['probs'], res['labels'])
    per_lang = {}
    print('\n--- A. quality ---')
    print(f'macro_f1 {overall["macro_f1"]:.4f} | acc {overall["acc"]:.4f} | auc {overall["auc"]:.4f}')
    for lang in sorted(set(langs.tolist())):
        m = langs == lang
        per_lang[lang] = metrics(res['probs'][m], res['labels'][m])
        ml = per_lang[lang]
        print(f'  {lang:<3} n={ml["n"]:<4} macro_f1 {ml["macro_f1"]:.4f} acc {ml["acc"]:.4f} '
              f'recall_yes {ml["recall_yes"]:.4f} precision_yes {ml["precision_yes"]:.4f} '
              f'auc {ml["auc"]:.4f}')

    # ---- B. inference-time ablation ---------------------------------------
    ablation = {}
    print('\n--- B. inference-time pooled_v ablation (head re-run only, no re-encoding) ---')
    if not arch['use_pooled_v']:
        print('  n/a: this checkpoint has no pooled_v in the head (the default since the '
              'flip; i.e. trained without --pooled-v).')
        print('  Compare its section C/D against a with-pooled_v checkpoint instead.')
    else:
        pc = torch.from_numpy(res['pooled_c'])
        conds = {
            'as trained': res['pooled_v'],
            'pooled_v := 0': np.zeros_like(res['pooled_v']),
            'permuted within language': res['pooled_v'][shifted_index(langs, True)],
            'permuted across language': res['pooled_v'][shifted_index(langs, False)],
        }
        print(f'  {"condition":<26} {"macro_f1":>8} {"delta":>8} {"acc":>7} {"delta":>8} '
              f'{"recall_yes":>10} {"auc":>7}')
        for name, v in conds.items():
            feats = torch.cat([pc, torch.from_numpy(np.ascontiguousarray(v))], dim=-1).float()
            m = metrics(head_prob_yes(net, feats, device), res['labels'])
            ablation[name] = m
            print(f'  {name:<26} {m["macro_f1"]:>8.4f} {m["macro_f1"] - overall["macro_f1"]:>+8.4f} '
                  f'{m["acc"]:>7.4f} {m["acc"] - overall["acc"]:>+8.4f} '
                  f'{m["recall_yes"]:>10.4f} {m["auc"]:>7.4f}')
        zero = ablation['pooled_v := 0']
        d_zero = overall['macro_f1'] - zero['macro_f1']
        if d_zero > 0.02:
            verdict = f'pooled_v is load-bearing: zeroing it costs {d_zero:+.4f} macro-F1'
        elif d_zero > 0.005:
            verdict = (f'pooled_v is used, but the cost ({d_zero:+.4f}) is inside the run-to-run '
                       f'spread - repeat before believing it')
        else:
            verdict = (f'pooled_v looks like dead weight here: zeroing it costs only '
                       f'{d_zero:+.4f} macro-F1')
        print(f'  reading: {verdict}')

    # ---- C. representation -------------------------------------------------
    ent = res['entropy_bits']
    print('\n--- C. representation ---')
    print('  cross-attn entropy per layer (bits)  ' + ' '.join(f'{e:.3f}' for e in ent) +
          f' | mean {np.mean(ent):.3f} (ceiling log2(mean valid video tokens '
          f'{res["mean_video_tokens"]:.0f}) = {math.log2(max(res["mean_video_tokens"], 2.0)):.2f})')
    print('  cross-attn top-1 weight per layer    ' +
          ' '.join(f'{t:.3f}' for t in res['top1_weight']) +
          f' | mean {np.mean(res["top1_weight"]):.3f}')
    print(f'  comment-pool entropy {res["pool_entropy_bits"]:.3f} bits')
    norm_c = float(np.linalg.norm(res['pooled_c'], axis=1).mean())
    if arch['use_pooled_v']:
        norm_v = float(np.linalg.norm(res['pooled_v'], axis=1).mean())
        print(f'  mean ||pooled_c|| {norm_c:.2f} | ||pooled_v|| {norm_v:.2f} | ratio {norm_v / norm_c:.2f}')
    else:
        print(f'  mean ||pooled_c|| {norm_c:.2f} | (no pooled_v)')

    # ---- D. probes ---------------------------------------------------------
    probes = {}
    if args.probe_train_rows > 0:
        sub = df_train.iloc[sample_rows(df_train, args.probe_train_rows)]
        tr_loader = make_loader(sub, tokenizer, args.batch_size, args.max_len, args.video_max_len)
        tr_c, tr_v = embed(net, tr_loader, device)
        va_c, va_v = torch.from_numpy(res['pooled_c']), torch.from_numpy(res['pooled_v'])
        y_tr, y_va = sub['st_y'], res['labels']
        print(f'\n--- D. linear probes (fit on {len(sub)} train rows, eval on {len(df_val)} val rows) ---')
        variants = [('pooled_c (fused comment side)', tr_c, va_c)]
        if arch['use_pooled_v']:
            variants.append(('pooled_v (video mean-pool)', tr_v, va_v))
            variants.append(('pooled_c + pooled_v', torch.cat([tr_c, tr_v], 1),
                             torch.cat([va_c, va_v], 1)))
        print(f'  {"features":<32} {"macro_f1":>8} {"acc":>7} {"recall_yes":>10} {"auc":>7}')
        for name, tr_x, va_x in variants:
            m = probe(tr_x, y_tr, va_x, y_va)
            probes[name] = m
            print(f'  {name:<32} {m["macro_f1"]:>8.4f} {m["acc"]:>7.4f} '
                  f'{m["recall_yes"]:>10.4f} {m["auc"]:>7.4f}')

    # ---- E. video-context perturbation (input-level causal test) -----------
    # B and C act on the head's features and on the representation; neither changes
    # what the cross-attention actually saw, and neither is a statement about the
    # model *relying* on the paired video. Here the video input itself is changed and
    # the whole model is re-run. If the model uses the paired video, breaking the
    # pairing or emptying the content must cost macro-F1. If it barely moves, the
    # attention it spends on the video is not evidence that it needs the video.
    #
    # All arms keep the batch layout and the per-row video mask intact: the permuted
    # arms swap whole (ids, mask) pairs between rows, and the constant arm replaces
    # every unmasked token id with one constant while leaving the mask alone. So the
    # only thing that changes is the video content, not the sequence length.
    print('\n--- E. video-context perturbation (whole model, video input changed) ---')
    from torch.utils.data import DataLoader
    from train_xattn import collate_paired

    ds = val_loader.dataset
    items = [ds[i] for i in range(len(ds))]
    idx_in = shifted_index(langs, within_language=True)
    idx_out = shifted_index(langs, within_language=False)
    const_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def rebuild(perm=None, constant=False):
        out = []
        for k, (cids, cmask, _vids, _vmask, lab) in enumerate(items):
            j = k if perm is None else int(perm[k])
            vids, vmask = items[j][2], items[j][3]
            if constant:
                vids = [const_id if m else t for t, m in zip(vids, vmask)]
            out.append((cids, cmask, vids, vmask, lab))
        return out

    def run(perturbed):
        loader = DataLoader(perturbed, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_paired(b, tokenizer.pad_token_id))
        return predict(net, loader, device)

    variants_e = {
        'as-is (paired video)': rebuild(),
        'video permuted within lang': rebuild(perm=idx_in),
        'video permuted across lang': rebuild(perm=idx_out),
        'video content := constant': rebuild(constant=True),
    }
    perturb = {}
    print(f'  {"condition":<28} {"macro_f1":>8} {"delta":>8} {"acc":>7} '
          f'{"recall_yes":>10} {"auc":>7}')
    for name, perturbed in variants_e.items():
        m = metrics(run(perturbed), res['labels'])
        perturb[name] = m
        print(f'  {name:<28} {m["macro_f1"]:>8.4f} {m["macro_f1"] - overall["macro_f1"]:>+8.4f} '
              f'{m["acc"]:>7.4f} {m["recall_yes"]:>10.4f} {m["auc"]:>7.4f}')
    # The as-is arm re-runs the model, so it must reproduce section A exactly.
    e_as_is = abs(perturb['as-is (paired video)']['macro_f1'] - overall['macro_f1'])
    if e_as_is > 1e-9:
        print(f'  WARNING: as-is arm differs from section A by {e_as_is:.2e} (expected 0)')
    d_pair = overall['macro_f1'] - perturb['video permuted within lang']['macro_f1']
    d_const = overall['macro_f1'] - perturb['video content := constant']['macro_f1']
    if d_const > 0.02:
        perturb_verdict = (f'the model uses the paired video: emptying it costs '
                           f'{d_const:+.4f} macro-F1 (breaking the pairing costs {d_pair:+.4f})')
    elif d_const > 0.005:
        perturb_verdict = (f'the model uses the video, but the cost ({d_const:+.4f}) is inside '
                           f'the run-to-run spread - repeat before believing it')
    else:
        perturb_verdict = (f'the model barely uses the paired video (emptying it costs only '
                           f'{d_const:+.4f} macro-F1): its attention on the video is not '
                           f'evidence of reliance')
    print(f'  reading: {perturb_verdict}')

    if args.json:
        payload = {'checkpoint': args.checkpoint, 'arch': arch, 'heads': args.heads,
                   'quality': {'overall': overall, 'per_language': per_lang},
                   'ablation': ablation,
                   'perturbation': perturb,
                   'representation': {k: res[k] for k in
                                      ('entropy_bits', 'top1_weight', 'pool_entropy_bits',
                                       'mean_video_tokens', 'self_check_head_gap')},
                   'probes': probes}
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
