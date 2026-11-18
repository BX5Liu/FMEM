#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

EPS = 1e-8
SEGMENTS = ("early", "middle", "late")


def norm_id(x: object) -> str:
    s = str(x).strip()
    if s.endswith("_P"):
        s = s[:-2]
    m = re.search(r"\d+", s)
    return str(int(m.group(0))) if m else s


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return p.read_text(encoding=enc, errors="ignore").strip()
        except Exception:
            continue
    return ""


def split_text_three(text: str) -> List[str]:


    text = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not text:
        return ["No valid participant response."] * 3

    units = [u.strip() for u in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if u.strip()]
    if len(units) < 3:
        toks = text.split()
        if len(toks) >= 3:
            cuts = np.array_split(np.arange(len(toks)), 3)
            return [" ".join(toks[idx[0]:idx[-1] + 1]) if len(idx) else "" for idx in cuts]
        chars = list(text)
        cuts = np.array_split(np.arange(len(chars)), 3)
        return ["".join(chars[idx[0]:idx[-1] + 1]) if len(idx) else text for idx in cuts]

    cuts = np.array_split(np.arange(len(units)), 3)
    return [" ".join(units[idx[0]:idx[-1] + 1]) if len(idx) else "" for idx in cuts]


def interviewer_text_warning(text: str) -> bool:


    pats = (
        r"(?im)^\s*(ellie|interviewer|therapist|virtual\s+agent|agent)\s*[:\t]",
    )
    return any(re.search(p, text or "") for p in pats)


def participant_only_text(text: str) -> str:


    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    label_re = re.compile(
        r"^\s*(ellie|interviewer|therapist|virtual\s+agent|agent|participant|patient)\s*[:\t]\s*(.*)$",
        re.I,
    )
    lines = raw.split("\n")
    if not any(label_re.match(line) for line in lines):
        return raw.strip()
    keep: List[str] = []
    current = None
    for line in lines:
        m = label_re.match(line)
        if m:
            who = m.group(1).lower()
            current = "participant" if who in {"participant", "patient"} else "interviewer"
            if current == "participant" and m.group(2).strip():
                keep.append(m.group(2).strip())
        elif current == "participant" and line.strip():
            keep.append(line.strip())
    return "\n".join(keep).strip()


def resolve_processed_folder(root: str | Path, sid: object) -> Path:
    root = Path(root)
    sidn = norm_id(sid)
    direct = root / f"{sidn}_P"
    if direct.is_dir():
        return direct
    matches = [p for p in root.glob("*_P") if norm_id(p.name) == sidn]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot resolve processed folder for subject {sidn} under {root}")
    raise RuntimeError(f"Ambiguous processed folders for subject {sidn}: {matches}")


def read_binary_label(folder: str | Path, threshold: float = 10.0) -> int:
    folder = Path(folder)

    for name in ("binary_label.txt", "label_binary.txt", "PHQ_Binary.txt"):
        p = folder / name
        if p.exists():
            raw = read_text(p).split()
            if raw:
                val = int(float(raw[0]))
                if val not in (0, 1):
                    raise ValueError(f"Expected binary label in {p}, got {val}")
                return val


    for name in ("label.txt", "new_label.txt", "phq8_score.txt", "PHQ8_Score.txt"):
        p = folder / name
        if p.exists():
            raw = read_text(p).split()
            if raw:
                return int(float(raw[0]) >= threshold)
    raise FileNotFoundError(f"No usable label file in {folder}")


def interviewer_mode(sid: object) -> str:
    n = int(norm_id(sid))
    if 300 <= n <= 492:
        return "woz"
    if 600 <= n <= 718:
        return "ai"
    return "unknown"


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject_id", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(norm_id).astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].astype(int)
    if "source_id" not in df:
        df["source_id"] = df["subject_id"]
    else:
        df["source_id"] = df["source_id"].map(norm_id).astype(str)
    if "interviewer_mode" not in df:
        df["interviewer_mode"] = df["subject_id"].map(interviewer_mode)
    if df["subject_id"].duplicated().any():
        dups = df.loc[df["subject_id"].duplicated(False), "subject_id"].tolist()
        raise ValueError(f"Manifest subject_id values must be unique; duplicates={dups[:20]}")
    return df


def temporal_payload(ids: Sequence[object], labels: Sequence[int], X: np.ndarray,
                     feature_names: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError(f"Expected [N,3,D], got {X.shape}")
    payload: Dict[str, np.ndarray] = {
        "subject_ids": np.asarray([norm_id(x) for x in ids]),
        "labels": np.asarray(labels, dtype=int),
        "features": X,
        "early": X[:, 0],
        "middle": X[:, 1],
        "late": X[:, 2],
    }
    if feature_names is not None:
        payload["feature_names"] = np.asarray(list(feature_names))
    return payload


def load_temporal_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    id_key = next((k for k in ("subject_ids", "ids", "id", "subject_id") if k in d), None)
    if id_key is None:
        raise KeyError(f"No subject ID key in {path}; keys={list(d.keys())}")
    ids = np.asarray([norm_id(x) for x in d[id_key]])
    y = None
    for k in ("labels", "label", "y"):
        if k in d:
            y = np.asarray(d[k], dtype=int)
            break
    if "features" in d:
        X = np.asarray(d["features"], dtype=np.float32)
    elif "profiles" in d:
        X = np.asarray(d["profiles"], dtype=np.float32)
    elif all(k in d for k in SEGMENTS):
        X = np.stack([d[k] for k in SEGMENTS], axis=1).astype(np.float32)
    else:
        raise KeyError(f"No temporal feature arrays in {path}; keys={list(d.keys())}")
    if X.ndim == 2:
        X = np.repeat(X[:, None, :], 3, axis=1)
    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError(f"Expected [N,3,D] in {path}, got {X.shape}")
    if len(ids) != len(X):
        raise ValueError(f"ID/feature length mismatch in {path}: {len(ids)} vs {len(X)}")
    return ids, y, X


def align_features(manifest: pd.DataFrame, paths: Dict[str, str | Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    maps: Dict[str, Dict[str, int]] = {}
    arrays: Dict[str, np.ndarray] = {}
    labels_from_npz: Dict[str, np.ndarray | None] = {}
    for name, p in paths.items():
        ids, y, X = load_temporal_npz(p)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Duplicate IDs in {p}")
        maps[name] = {sid: i for i, sid in enumerate(ids)}
        arrays[name] = X
        labels_from_npz[name] = y

    ids_out: List[str] = []
    y_out: List[int] = []
    split_out: List[str] = []
    aligned: Dict[str, List[np.ndarray]] = {k: [] for k in paths}
    missing: Dict[str, List[str]] = {k: [] for k in paths}

    for _, r in manifest.iterrows():
        sid = str(r["subject_id"])
        bad = False
        for name in paths:
            if sid not in maps[name]:
                missing[name].append(sid)
                bad = True
        if bad:
            continue
        ids_out.append(sid)
        y_out.append(int(r["label"]))
        split_out.append(str(r["split"]))
        for name in paths:
            idx = maps[name][sid]
            yn = labels_from_npz[name]
            if yn is not None and int(yn[idx]) != int(r["label"]):
                raise ValueError(f"Label mismatch subject={sid} branch={name}: npz={yn[idx]} manifest={r['label']}")
            aligned[name].append(arrays[name][idx])

    for name, vals in missing.items():
        if vals:
            print(f"[WARN] {name}: missing {len(vals)} manifest IDs; first={vals[:10]}", flush=True)
    if not ids_out:
        raise RuntimeError("No subjects aligned across manifest and feature files")
    return (
        np.asarray(ids_out),
        np.asarray(y_out, dtype=int),
        np.asarray(split_out),
        {k: np.asarray(v, dtype=np.float32) for k, v in aligned.items()},
    )


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def subject_relative(x: np.ndarray) -> np.ndarray:
    z = l2(x)
    return l2(z - z.mean(axis=1, keepdims=True))


def safe_auc(y: Sequence[int], s: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def best_f1_threshold(y: Sequence[int], s: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    grid = np.unique(np.r_[np.quantile(s, np.linspace(0.01, 0.99, 199)), s.min() - 1e-9, s.max() + 1e-9])
    vals = [f1_score(y, s >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(vals))])


def prior_threshold(s: Sequence[float], prevalence: float) -> float:
    return float(np.quantile(np.asarray(s, dtype=float), np.clip(1.0 - prevalence, 0.01, 0.99)))


def metric_row(y: Sequence[int], s: Sequence[float], threshold: float) -> Dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(y, s),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppr": float(pred.mean()),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_auc(y: Sequence[int], s: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(safe_auc(y[idx], s[idx]))
    v = np.asarray(vals)
    return safe_auc(y, s), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_bootstrap_delta(y: Sequence[int], a: Sequence[float], b: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(safe_auc(y[idx], a[idx]) - safe_auc(y[idx], b[idx]))
    v = np.asarray(vals)
    delta = safe_auc(y, a) - safe_auc(y, b)
    p = 2 * min((np.sum(v <= 0) + 1) / (len(v) + 1), (np.sum(v >= 0) + 1) / (len(v) + 1))
    return {
        "delta_auc": float(delta),
        "delta_auc_lo": float(np.percentile(v, 2.5)),
        "delta_auc_hi": float(np.percentile(v, 97.5)),
        "p_boot": float(min(p, 1.0)),
    }


def json_dump(obj: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
