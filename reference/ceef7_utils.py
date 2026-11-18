#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

EPS = 1e-8
TOPICS = ("positive", "neutral", "negative")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def norm_id(x: object) -> str:
    s = str(x).strip()
    if s.endswith("_P"):
        s = s[:-2]
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    m = re.fullmatch(r"([A-Za-z]+)[_-]?(\d+)", s)
    if m:
        return f"{m.group(1).lower()}_{int(m.group(2))}"
    return s


def recruitment_batch(ids: Sequence[object]) -> np.ndarray:
    out = []
    for x in ids:
        s = norm_id(x)
        out.append(0 if s.startswith("t_") else 1 if s.startswith("v_") else 2)
    return np.asarray(out, dtype=int)


def joint_strata(y: Sequence[int], ids: Sequence[object]) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    b = recruitment_batch(ids)
    if np.any(b == 2):
        raise ValueError("Joint source stratification requires t_/v_ subject IDs.")
    return y * 2 + b


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(s) == 0:
        return 0.0
    qs = np.linspace(0.01, 0.99, 199)
    grid = np.unique(np.r_[np.quantile(s, qs), s.min() - 1e-9, s.max() + 1e-9])
    vals = [f1_score(y, s >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(vals))])


def prior_threshold(s: np.ndarray, prevalence: float) -> float:
    q = float(np.clip(1.0 - prevalence, 0.01, 0.99))
    return float(np.quantile(np.asarray(s, dtype=float), q))


def metric_row(y: np.ndarray, s: np.ndarray, pred: np.ndarray | None = None,
               threshold: float | None = None) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if pred is None:
        if threshold is None:
            raise ValueError("pred or threshold is required")
        pred = (s >= threshold).astype(int)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(y, s),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppr": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold) if threshold is not None else float("nan"),
    }


def make_outer_folds(ids: Sequence[object], y: Sequence[int], repeats: int,
                     folds: int, seed: int) -> List[Dict[str, object]]:
    ids = np.asarray([str(x) for x in ids])
    y = np.asarray(y, dtype=int)
    strata = joint_strata(y, ids)
    out: List[Dict[str, object]] = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1009 * r)
        for f, (tr, va) in enumerate(skf.split(np.zeros(len(y)), strata)):
            counts = {}
            for split_name, idx in (("train", tr), ("val", va)):
                counts[split_name] = {
                    "t_neg": int(np.sum((recruitment_batch(ids[idx]) == 0) & (y[idx] == 0))),
                    "t_pos": int(np.sum((recruitment_batch(ids[idx]) == 0) & (y[idx] == 1))),
                    "v_neg": int(np.sum((recruitment_batch(ids[idx]) == 1) & (y[idx] == 0))),
                    "v_pos": int(np.sum((recruitment_batch(ids[idx]) == 1) & (y[idx] == 1))),
                }
            out.append({
                "repeat": r,
                "fold": f,
                "train_idx": tr.tolist(),
                "val_idx": va.tolist(),
                "train_ids": ids[tr].tolist(),
                "val_ids": ids[va].tolist(),
                "counts": counts,
            })
    return out


def make_inner_splits(ids: Sequence[object], y: Sequence[int], folds: int, seed: int):
    ids = np.asarray(ids)
    y = np.asarray(y, dtype=int)
    strata = joint_strata(y, ids)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strata))


def ids_sha256(ids: Sequence[object]) -> str:
    payload = "\n".join(str(x) for x in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_fold_manifest(path: str | Path, ids: Sequence[object], y: Sequence[int],
                        folds: List[Dict[str, object]], metadata: Mapping[str, object] | None = None) -> None:
    obj = {
        "version": "ceef7-joint-label-batch-folds-v1",
        "ids_sha256": ids_sha256(ids),
        "n": len(ids),
        "labels": np.bincount(np.asarray(y, dtype=int), minlength=2).tolist(),
        "ids": [str(x) for x in ids],
        "folds": folds,
        "metadata": dict(metadata or {}),
    }
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fold_manifest(path: str | Path, ids: Sequence[object]) -> Dict[str, object]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("ids_sha256") != ids_sha256(ids):
        raise RuntimeError("fold_indices.json subject ID order/hash does not match current data")
    return obj


def paired_bootstrap_delta(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                           n_boot: int, seed: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(safe_auc(y[idx], a[idx]) - safe_auc(y[idx], b[idx]))
    v = np.asarray(vals, dtype=float)
    delta = safe_auc(y, a) - safe_auc(y, b)

    p = 2.0 * min((np.sum(v <= 0) + 1) / (len(v) + 1), (np.sum(v >= 0) + 1) / (len(v) + 1))
    return {
        "delta_auc": float(delta),
        "delta_auc_lo": float(np.percentile(v, 2.5)),
        "delta_auc_hi": float(np.percentile(v, 97.5)),
        "p_boot": float(min(p, 1.0)),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def summarize_repeats(rows: pd.DataFrame, group_col: str = "model") -> pd.DataFrame:
    metrics = [c for c in ("auc", "f1", "bacc", "recall", "specificity", "ppr") if c in rows]
    out = []
    for name, g in rows.groupby(group_col):
        r = {group_col: name, "n_repeats": int(len(g))}
        for c in metrics:
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        out.append(r)
    return pd.DataFrame(out)


def json_dump(obj: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
