"""Paired video-cluster bootstrap for Task B probability dumps.

The validation rows are comments nested in videos, so resampling rows i.i.d.
understates uncertainty. This script resamples whole videos with replacement,
keeping every comment from a selected video, and compares two probability
dumps on exactly the same resampled rows.

Examples
--------
Compare two saved ensembles::

    python src/eval_bootstrap.py \
        --labels weights/val_labels_taskb.npy \
        --videos weights/val_videos_taskb.npy \
        --prefix CW=weights --prefix CWI=weights

Compare individual probability files::

    python src/eval_bootstrap.py \
        --labels weights/val_labels_taskb.npy \
        --videos weights/val_videos_taskb.npy \
        --probs CW=weights/CW_s42_val_probs.npy \
        --probs CWI=weights/CWI_s42_val_probs.npy

`--prefix NAME=DIR` averages every ``DIR/NAME_s*_val_probs.npy``. The output is
printed and, with ``--json-out``, written as JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y: np.ndarray, probs: np.ndarray) -> float:
    pred = probs.argmax(1)
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


def per_class_f1(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    vals = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"no": float(vals[0]), "implicit": float(vals[1]), "explicit": float(vals[2])}


def load_prefix(directory: str, prefix: str) -> tuple[np.ndarray, list[str]]:
    paths = sorted(glob.glob(os.path.join(directory, f"{prefix}_s*_val_probs.npy")))
    if not paths:
        raise FileNotFoundError(f"no {prefix}_s*_val_probs.npy in {directory}")
    arrays = [np.load(p) for p in paths]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError(f"inconsistent shapes for {prefix}: {[a.shape for a in arrays]}")
    return np.mean(arrays, axis=0), paths


def load_probs(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected [N, C], got {arr.shape}")
    return arr


def cluster_groups(videos: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(videos.astype(str), return_inverse=True)
    return [np.where(inv == i)[0] for i in range(int(inv.max()) + 1)]


def paired_bootstrap(
    y: np.ndarray,
    groups: list[np.ndarray],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    observed = macro_f1(y, probs_a) - macro_f1(y, probs_b)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picked = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([groups[j] for j in picked])
        diffs[i] = macro_f1(y[idx], probs_a[idx]) - macro_f1(y[idx], probs_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap-tail p-value; conservative when the distribution is
    # centred exactly on zero.
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "observed": float(observed),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(min(1.0, p)),
        "bootstrap_mean": float(diffs.mean()),
        "n_boot": int(n_boot),
        "n_videos": int(n_groups),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--langs", default=None)
    ap.add_argument("--prefix", action="append", default=[], metavar="NAME=DIR",
                    help="average NAME_s*_val_probs.npy in DIR")
    ap.add_argument("--probs", action="append", default=[], metavar="NAME=PATH",
                    help="single probability dump")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    y = np.load(args.labels).astype(int).ravel()
    videos = np.load(args.videos, allow_pickle=True).astype(str).ravel()
    if len(y) != len(videos):
        raise ValueError(f"labels/videos length mismatch: {len(y)} vs {len(videos)}")
    langs = None
    if args.langs:
        langs = np.load(args.langs, allow_pickle=True).astype(str).ravel()
        if len(langs) != len(y):
            raise ValueError(f"langs length mismatch: {len(langs)} vs {len(y)}")
    groups = cluster_groups(videos)

    if not args.prefix and not args.probs:
        raise SystemExit("provide at least one --prefix or --probs")

    entries: dict[str, dict] = {}
    for spec in args.prefix:
        if "=" not in spec:
            raise SystemExit(f"bad --prefix {spec!r}; use NAME=DIR")
        name, directory = spec.split("=", 1)
        probs, paths = load_prefix(directory, name)
        if len(probs) != len(y):
            raise ValueError(f"{name}: {len(probs)} probs vs {len(y)} labels")
        entries[name] = {"probs": probs, "paths": paths, "kind": "ensemble"}
    for spec in args.probs:
        if "=" not in spec:
            raise SystemExit(f"bad --probs {spec!r}; use NAME=PATH")
        name, path = spec.split("=", 1)
        probs = load_probs(path)
        if len(probs) != len(y):
            raise ValueError(f"{name}: {len(probs)} probs vs {len(y)} labels")
        entries[name] = {"probs": probs, "paths": [path], "kind": "single"}

    report: dict = {
        "n_rows": int(len(y)),
        "n_videos": int(len(groups)),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "entries": {},
        "comparisons": [],
    }
    print(f"rows {len(y)} | videos {len(groups)} | bootstrap {args.n_boot}")
    print(f"{'name':18} {'macroF1':>8} {'no':>7} {'imp':>7} {'exp':>7}  source")
    for name, e in entries.items():
        pc = per_class_f1(y, e["probs"])
        report["entries"][name] = {
            "macro_f1": macro_f1(y, e["probs"]),
            **pc,
            "kind": e["kind"],
            "paths": e["paths"],
        }
        print(f"{name:18} {macro_f1(y, e['probs']):8.4f} {pc['no']:7.4f} "
              f"{pc['implicit']:7.4f} {pc['explicit']:7.4f}  "
              f"{e['kind']}:{len(e['paths'])}")

    print("\npaired video-cluster bootstrap (positive = first model better)")
    for (name_a, e_a), (name_b, e_b) in combinations(entries.items(), 2):
        res = paired_bootstrap(y, groups, e_a["probs"], e_b["probs"],
                               args.n_boot, args.seed)
        delta_pc = {
            k: per_class_f1(y, e_a["probs"])[k] - per_class_f1(y, e_b["probs"])[k]
            for k in ("no", "implicit", "explicit")
        }
        res.update({
            "a": name_a,
            "b": name_b,
            "delta_per_class": delta_pc,
        })
        if langs is not None:
            res["per_language"] = {}
            for lang in sorted(set(langs.tolist())):
                msk = langs == lang
                res["per_language"][lang] = {
                    "n": int(msk.sum()),
                    "delta": float(macro_f1(y[msk], e_a["probs"][msk])
                                   - macro_f1(y[msk], e_b["probs"][msk])),
                }
        report["comparisons"].append(res)
        print(f"  {name_a} - {name_b}: {res['observed']:+.4f} "
              f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}] "
              f"p={res['p_two_sided']:.4f} | "
              f"d no {delta_pc['no']:+.3f} imp {delta_pc['implicit']:+.3f} "
              f"exp {delta_pc['explicit']:+.3f}")
        if langs is not None:
            per = " ".join(f"{lg} {v['delta']:+.3f}" for lg, v in res["per_language"].items())
            print(f"    per-language: {per}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
