#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

TOPICS = ("positive", "neutral", "negative")


def _unique_normalized_map(raw_ids: Sequence[object]) -> Dict[str, int]:
    buckets: Dict[str, List[int]] = {}
    for i, x in enumerate(raw_ids):
        buckets.setdefault(norm_id(x), []).append(i)
    return {k: v[0] for k, v in buckets.items() if len(v) == 1}


def _resolve_index(raw_id: object, exact_map: Dict[str, int], norm_map: Dict[str, int]) -> Optional[int]:
    s = str(raw_id).strip()
    if s in exact_map:
        return exact_map[s]
    return norm_map.get(norm_id(s))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def norm_id(x: object) -> str:


    s = str(x).strip()

    if s.endswith("_P"):
        s = s[:-2]


    if re.fullmatch(r"\d+", s):
        return str(int(s))


    m = re.fullmatch(r"([A-Za-z]+)[_-]?(\d+)", s)
    if m:
        prefix = m.group(1).lower()
        number = int(m.group(2))
        return f"{prefix}_{number}"

    return s


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, object]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": safe_auc(y, p),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppr": float(pred.mean()),
        "threshold": float(threshold),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, target_ppr: Optional[float] = None) -> Dict[str, object]:

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if target_ppr is None:
        target_ppr = float(y.mean())
    grid = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 197),
        np.quantile(p, np.linspace(0.01, 0.99, 199)),
    ]))
    pred = p[None, :] >= grid[:, None]
    pos = y[None, :] == 1
    neg = ~pos
    tp = np.sum(pred & pos, axis=1).astype(float)
    fp = np.sum(pred & neg, axis=1).astype(float)
    fn = np.sum((~pred) & pos, axis=1).astype(float)
    tn = np.sum((~pred) & neg, axis=1).astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    specificity = tn / np.maximum(tn + fp, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    acc = (tp + tn) / max(len(y), 1)
    bacc = 0.5 * (recall + specificity)
    ppr = np.mean(pred, axis=1)
    auc = safe_auc(y, p)
    score = 0.66 * f1 + 0.18 * bacc + 0.16 * auc - 0.035 * np.abs(ppr - target_ppr)
    i = int(np.argmax(score))
    return {
        "f1": float(f1[i]), "acc": float(acc[i]), "bacc": float(bacc[i]), "auc": float(auc),
        "precision": float(precision[i]), "recall": float(recall[i]), "specificity": float(specificity[i]),
        "ppr": float(ppr[i]), "threshold": float(grid[i]),
        "cm": [[int(tn[i]), int(fp[i])], [int(fn[i]), int(tp[i])]],
        "selection_score": float(score[i]),
    }


def _topic_ids(d: np.lib.npyio.NpzFile, topic: str) -> np.ndarray:
    return np.asarray([norm_id(x) for x in d[f"{topic}_ids"]])


def _build_label_index(data_root: str) -> Dict[str, float]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    exact: Dict[str, float] = {}
    numeric: Dict[str, List[float]] = {}
    for p in root.rglob("new_label.txt"):
        try:
            score = float(p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0])
        except Exception:
            continue
        folder = p.parent.name
        exact[folder] = score
        nid = norm_id(folder)
        numeric.setdefault(nid, []).append(score)
    out = dict(exact)
    for nid, vals in numeric.items():

        if len(set(round(v, 6) for v in vals)) == 1:
            out[nid] = vals[0]
    return out


def _resolve_score(label_index: Dict[str, float], raw_id: object) -> float:
    s = str(raw_id).strip()
    for key in (s, norm_id(s)):
        if key in label_index:
            return float(label_index[key])
    return float("nan")


def load_clinical_evidence(
    path: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if not path:
        return np.array([]), np.array([]), None, None

    d = np.load(path, allow_pickle=True)
    ids_key = "subject_ids" if "subject_ids" in d else "ids"


    ids = np.asarray([
        str(x).strip()
        for x in d[ids_key]
    ])

    profile = None
    summary = None

    if "profiles" in d:
        profile = np.asarray(
            d["profiles"],
            dtype=np.float32,
        )
    elif all(f"{t}_profile" in d for t in TOPICS):
        profile = np.stack(
            [d[f"{t}_profile"] for t in TOPICS],
            axis=1,
        ).astype(np.float32)

    if "summary_features" in d:
        summary = np.asarray(
            d["summary_features"],
            dtype=np.float32,
        )

    return ids, np.arange(len(ids)), profile, summary


def load_dataset(
    audio_npz: str,
    text_npz: str,
    data_root: str = "",
    evidence_npz: str = "",
    threshold: float = 53.0,
    expected_n: int = 162,
    expected_neg: int = 132,
    expected_pos: int = 30,
    strict_expected: bool = True,
) -> Dict[str, np.ndarray]:
    ad = np.load(audio_npz, allow_pickle=True)
    td = np.load(text_npz, allow_pickle=True)
    aid_raw = np.asarray(ad["positive_ids"])
    tid_raw = np.asarray(td["positive_ids"])
    aid_exact = {str(x).strip(): i for i, x in enumerate(aid_raw)}
    tid_exact = {str(x).strip(): i for i, x in enumerate(tid_raw)}
    aid_norm_unique = _unique_normalized_map(aid_raw)
    tid_norm_unique = _unique_normalized_map(tid_raw)

    evidence_ids, _, evidence_profile, evidence_summary = load_clinical_evidence(evidence_npz)
    evidence_exact = {str(x).strip(): i for i, x in enumerate(evidence_ids)} if evidence_npz else {}
    evidence_norm_unique = _unique_normalized_map(evidence_ids) if evidence_npz else {}
    label_index = _build_label_index(data_root) if data_root else {}

    rows: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    for ai, raw_sid in enumerate(aid_raw):
        sid_exact = str(raw_sid).strip()
        sid = norm_id(raw_sid)
        ti = _resolve_index(raw_sid, tid_exact, tid_norm_unique)
        if ti is None:
            continue
        ei = _resolve_index(raw_sid, evidence_exact, evidence_norm_unique) if evidence_npz else None
        if evidence_npz and ei is None:
            continue
        for topic in TOPICS:
            a_topic = str(ad[f"{topic}_ids"][ai]).strip()
            t_topic = str(td[f"{topic}_ids"][ti]).strip()
            if a_topic != sid_exact and norm_id(a_topic) != sid:
                raise ValueError(f"Audio topic ID mismatch: {sid_exact}/{topic}: {a_topic}")
            if t_topic != sid_exact and norm_id(t_topic) != sid:
                raise ValueError(f"Text topic ID mismatch: {sid_exact}/{topic}: {t_topic}")

        a_label = int(ad["positive_labels"][ai]) if "positive_labels" in ad else None
        t_label = int(td["positive_labels"][ti]) if "positive_labels" in td else None
        score = _resolve_score(label_index, raw_sid) if label_index else float("nan")
        raw_label = int(score >= threshold) if np.isfinite(score) else None

        if a_label is not None and t_label is not None and a_label != t_label:
            raise ValueError(f"Audio/text label mismatch for {raw_sid}: {a_label} vs {t_label}")
        if raw_label is not None:
            label = raw_label
        elif a_label is not None:
            label = a_label
        elif t_label is not None:
            label = t_label
        else:
            raise ValueError(f"No label available for {raw_sid}")

        audit.append({
            "subject_id": str(raw_sid), "norm_id": sid,
            "audio_label": a_label, "text_label": t_label,
            "sds_score": score if np.isfinite(score) else None,
            "resolved_label": label,
            "label_mismatch_raw_vs_audio": bool(raw_label is not None and a_label is not None and raw_label != a_label),
        })

        t = np.stack([np.asarray(td[f"{topic}_features"][ti], dtype=np.float32) for topic in TOPICS], axis=0)
        asem = np.stack([np.asarray(ad[f"{topic}_semantic"][ai], dtype=np.float32) for topic in TOPICS], axis=0)
        basic = np.stack([np.asarray(ad[f"{topic}_basic"][ai], dtype=np.float32) for topic in TOPICS], axis=0)
        cue = np.stack([np.asarray(ad[f"{topic}_emotion"][ai], dtype=np.float32) for topic in TOPICS], axis=0)
        profile = evidence_profile[ei] if evidence_profile is not None else cue
        summary = evidence_summary[ei] if evidence_summary is not None else np.zeros((0,), dtype=np.float32)
        rows.append({
            "subject_id": str(raw_sid), "norm_id": sid, "y": label, "sds": score,
            "text": t, "audio_sem": asem, "basic": basic, "cue": cue,
            "profile": np.asarray(profile, dtype=np.float32),
            "summary": np.asarray(summary, dtype=np.float32),
        })

    out = {
        "ids": np.asarray([r["subject_id"] for r in rows]),
        "norm_ids": np.asarray([r["norm_id"] for r in rows]),
        "y": np.asarray([r["y"] for r in rows], dtype=int),
        "sds": np.asarray([r["sds"] for r in rows], dtype=np.float32),
        "text": np.asarray([r["text"] for r in rows], dtype=np.float32),
        "audio_sem": np.asarray([r["audio_sem"] for r in rows], dtype=np.float32),
        "basic": np.asarray([r["basic"] for r in rows], dtype=np.float32),
        "cue": np.asarray([r["cue"] for r in rows], dtype=np.float32),
        "profile": np.asarray([r["profile"] for r in rows], dtype=np.float32),
        "summary": np.asarray([r["summary"] for r in rows], dtype=np.float32),
        "audit": np.asarray(audit, dtype=object),
    }
    dist = np.bincount(out["y"], minlength=2)
    if strict_expected:
        if len(out["y"]) != expected_n or dist[0] != expected_neg or dist[1] != expected_pos:
            mismatch_n = sum(bool(x["label_mismatch_raw_vs_audio"]) for x in audit)
            raise RuntimeError(
                "Resolved label distribution does not match the expected EATD-Corpus protocol. "
                f"Got N={len(out['y'])}, labels={dist.tolist()}, expected N={expected_n}, "
                f"labels=[{expected_neg},{expected_pos}], raw-vs-NPZ mismatches={mismatch_n}. "
                "Do not continue until DATA_ROOT and SDS threshold are correct."
            )
    return out


def load_fixed_splits(path: str, ids: Sequence[object]) -> List[Tuple[np.ndarray, np.ndarray]]:
    d = np.load(path, allow_pickle=True)
    exact = {str(s).strip(): i for i, s in enumerate(ids)}
    norm_unique = _unique_normalized_map(ids)
    def resolve_many(values):
        out = []
        missing = []
        for x in values:
            i = _resolve_index(x, exact, norm_unique)
            if i is None:
                missing.append(str(x))
            else:
                out.append(i)
        if missing:
            raise RuntimeError(f"Split IDs missing or ambiguous: {missing[:20]}")
        return np.asarray(out, dtype=int)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in (1, 2, 3):
        tr = resolve_many(d[f"fold{fold}_train_ids"])
        va = resolve_many(d[f"fold{fold}_val_ids"])
        folds.append((tr, va))
    seen = np.concatenate([v for _, v in folds])
    if len(seen) != len(ids) or len(np.unique(seen)) != len(ids):
        raise RuntimeError(f"Validation folds must cover every subject exactly once: rows={len(seen)}, unique={len(np.unique(seen))}, N={len(ids)}")
    return folds


def topic_expand(z: np.ndarray) -> np.ndarray:

    z = np.asarray(z, dtype=np.float32)
    p, n, g = z[:, 0], z[:, 1], z[:, 2]
    return np.concatenate([
        p, n, g,
        z.mean(axis=1), z.std(axis=1), z.max(axis=1), z.min(axis=1),
        g - p, g - n, n - p,
        np.abs(g - p), np.abs(g - n), np.abs(n - p),
    ], axis=1).astype(np.float32)


@dataclass
class FactoryConfig:
    text_pca: int = 64
    audio_pca: int = 48
    summary_pca: int = 48
    cca_dim: int = 8
    seed: int = 42


class FeatureFactory:
    def __init__(self, cfg: FactoryConfig):
        self.cfg = cfg
        self.text_center = StandardScaler(with_std=False)
        self.audio_center = StandardScaler(with_std=False)
        self.text_pca: Optional[PCA] = None
        self.audio_pca: Optional[PCA] = None
        self.summary_center = StandardScaler(with_std=False)
        self.summary_pca: Optional[PCA] = None
        self.basic_scaler = StandardScaler()
        self.profile_scaler = RobustScaler()
        self.cca: Optional[CCA] = None

    def fit(self, data: Dict[str, np.ndarray], idx: np.ndarray) -> "FeatureFactory":
        t = data["text"][idx].reshape(-1, data["text"].shape[-1])
        a = data["audio_sem"][idx].reshape(-1, data["audio_sem"].shape[-1])
        tc = self.text_center.fit_transform(t)
        ac = self.audio_center.fit_transform(a)
        nt = min(self.cfg.text_pca, tc.shape[0] - 1, tc.shape[1])
        na = min(self.cfg.audio_pca, ac.shape[0] - 1, ac.shape[1])
        self.text_pca = PCA(n_components=max(2, nt), whiten=True, random_state=self.cfg.seed).fit(tc)
        self.audio_pca = PCA(n_components=max(2, na), whiten=True, random_state=self.cfg.seed + 1).fit(ac)
        self.basic_scaler.fit(data["basic"][idx].reshape(-1, data["basic"].shape[-1]))
        self.profile_scaler.fit(data["profile"][idx].reshape(-1, data["profile"].shape[-1]))

        if data["summary"].shape[-1] > 0:
            s = data["summary"][idx]
            sc = self.summary_center.fit_transform(s)
            ns = min(self.cfg.summary_pca, sc.shape[0] - 1, sc.shape[1])
            self.summary_pca = PCA(n_components=max(2, ns), whiten=True, random_state=self.cfg.seed + 2).fit(sc)

        tz = self._transform_topic(data["text"][idx], "text")
        az = self._transform_topic(data["audio_sem"][idx], "audio")
        d = min(tz.shape[-1], az.shape[-1], 24)
        if d >= 4 and len(idx) > 20:
            try:
                self.cca = CCA(n_components=min(self.cfg.cca_dim, d, len(idx) - 2), max_iter=2000, scale=True)
                self.cca.fit(tz.mean(axis=1)[:, :d], az.mean(axis=1)[:, :d])
            except Exception:
                self.cca = None
        return self

    def _transform_topic(self, x: np.ndarray, kind: str) -> np.ndarray:
        n, k, d = x.shape
        flat = x.reshape(-1, d)
        if kind == "text":
            assert self.text_pca is not None
            z = self.text_pca.transform(self.text_center.transform(flat))
        else:
            assert self.audio_pca is not None
            z = self.audio_pca.transform(self.audio_center.transform(flat))
        return z.reshape(n, k, -1).astype(np.float32)

    def transform(self, data: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
        t = self._transform_topic(data["text"][idx], "text")
        a = self._transform_topic(data["audio_sem"][idx], "audio")
        b = self.basic_scaler.transform(data["basic"][idx].reshape(-1, data["basic"].shape[-1])).reshape(len(idx), 3, -1)
        c = self.profile_scaler.transform(data["profile"][idx].reshape(-1, data["profile"].shape[-1])).reshape(len(idx), 3, -1)

        text_view = topic_expand(t)
        audio_view = np.concatenate([topic_expand(a), topic_expand(b)], axis=1).astype(np.float32)
        cue_view = topic_expand(c)

        if self.summary_pca is not None:
            s = self.summary_pca.transform(self.summary_center.transform(data["summary"][idx])).astype(np.float32)
            text_view = np.concatenate([text_view, s], axis=1)
            cue_view = np.concatenate([cue_view, s], axis=1)

        cross_parts = []
        if self.cca is not None:
            d = min(t.shape[-1], a.shape[-1], 24)
            try:
                tx, ax = self.cca.transform(t.mean(axis=1)[:, :d], a.mean(axis=1)[:, :d])
                denom = np.linalg.norm(tx, axis=1) * np.linalg.norm(ax, axis=1) + 1e-8
                cos = (np.sum(tx * ax, axis=1) / denom)[:, None]
                cross_parts.extend([tx, ax, tx - ax, np.abs(tx - ax), tx * ax, cos])
            except Exception:
                pass

        rel = np.stack([
            np.linalg.norm(t.std(axis=1), axis=1),
            np.linalg.norm(a.std(axis=1), axis=1),
            np.linalg.norm(c.std(axis=1), axis=1),
            np.linalg.norm(t[:, 2] - t[:, 0], axis=1),
            np.linalg.norm(a[:, 2] - a[:, 0], axis=1),
        ], axis=1).astype(np.float32)
        cross_parts.append(rel)
        cross_view = np.concatenate(cross_parts, axis=1).astype(np.float32)
        fused_view = np.concatenate([text_view, audio_view, cue_view, cross_view], axis=1).astype(np.float32)
        return {
            "text": np.nan_to_num(text_view),
            "audio": np.nan_to_num(audio_view),
            "cue": np.nan_to_num(cue_view),
            "cross": np.nan_to_num(cross_view),
            "fused": np.nan_to_num(fused_view),
            "y": data["y"][idx],
            "ids": data["ids"][idx],
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)


class PLSDA:
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model: Optional[PLSRegression] = None
        self.lo = 0.0
        self.hi = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSDA":
        z = self.scaler.fit_transform(x)
        ncomp = min(self.n_components, z.shape[0] - 1, z.shape[1])
        self.model = PLSRegression(n_components=max(1, ncomp), scale=False, max_iter=1000)
        self.model.fit(z, y.astype(float))
        raw = self.model.predict(z).reshape(-1)
        self.lo, self.hi = float(np.quantile(raw, 0.01)), float(np.quantile(raw, 0.99))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        raw = self.model.predict(self.scaler.transform(x)).reshape(-1)
        z = (raw - self.lo) / max(self.hi - self.lo, 1e-8)
        p = np.clip(z, 0.001, 0.999)
        return np.column_stack([1 - p, p])


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    params: Tuple[Tuple[str, object], ...]
    mixup: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "kind": self.kind, "params": dict(self.params), "mixup": self.mixup}


def candidate_library(level: str = "full") -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    if level == "smoke":
        return [
            CandidateSpec("lr_C0.3", "lr", (("C", 0.3),), False),
            CandidateSpec("pls_4", "pls", (("n_components", 4),), False),
            CandidateSpec("et_d3", "et", (("max_depth", 3), ("min_samples_leaf", 3)), False),
        ]
    lr_cs = (0.1, 0.3, 1.0) if level == "quick" else (0.03, 0.1, 0.3, 1.0, 3.0)
    for c in lr_cs:
        out.append(CandidateSpec(f"lr_C{c}", "lr", (("C", c),), False))
        out.append(CandidateSpec(f"lr_mix_C{c}", "lr", (("C", c),), True))
    svm_cs = (0.2, 1.0) if level == "quick" else (0.05, 0.2, 1.0, 5.0)
    for c in svm_cs:
        out.append(CandidateSpec(f"lin_svm_C{c}", "svm_linear", (("C", c),), False))
    rbf_cs = (1.0,) if level == "quick" else (0.2, 1.0, 5.0)
    for c in rbf_cs:
        out.append(CandidateSpec(f"rbf_svm_C{c}", "svm_rbf", (("C", c), ("gamma", "scale")), False))
    pls_ns = (2, 4, 8) if level == "quick" else (2, 4, 6, 8, 12)
    for n in pls_ns:
        out.append(CandidateSpec(f"pls_{n}", "pls", (("n_components", n),), False))
        if level != "quick" and n in (4, 8):
            out.append(CandidateSpec(f"pls_mix_{n}", "pls", (("n_components", n),), True))
    depths = (3,) if level == "quick" else (3, 5)
    for depth in depths:
        out.append(CandidateSpec(f"et_d{depth}", "et", (("max_depth", depth), ("min_samples_leaf", 3)), False))
    return out


def mixup_minority(x: np.ndarray, y: np.ndarray, seed: int, target_ratio: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    target_pos = int(round(len(neg) * target_ratio))
    need = max(0, target_pos - len(pos))
    if need == 0 or len(pos) < 2:
        return x, y
    synth = []
    scale = np.std(x[pos], axis=0, keepdims=True) + 1e-6
    for _ in range(need):
        i, j = rng.choice(pos, 2, replace=False)
        lam = rng.beta(0.6, 0.6)
        z = lam * x[i] + (1 - lam) * x[j]
        z = z + rng.normal(0, 0.015, size=z.shape) * scale.reshape(-1)
        synth.append(z)
    return np.vstack([x, np.asarray(synth, dtype=np.float32)]), np.concatenate([y, np.ones(need, dtype=int)])


def fit_candidate(spec: CandidateSpec, x: np.ndarray, y: np.ndarray, seed: int):
    xfit, yfit = (mixup_minority(x, y, seed) if spec.mixup else (x, y))
    p = dict(spec.params)
    if spec.kind == "lr":
        model = LogisticRegression(
            C=float(p["C"]), solver="liblinear", class_weight="balanced",
            max_iter=4000, random_state=seed,
        )
    elif spec.kind == "svm_linear":
        model = SVC(kernel="linear", C=float(p["C"]), probability=False, class_weight="balanced", random_state=seed)
    elif spec.kind == "svm_rbf":
        model = SVC(kernel="rbf", C=float(p["C"]), gamma=p.get("gamma", "scale"), probability=False, class_weight="balanced", random_state=seed)
    elif spec.kind == "pls":
        model = PLSDA(int(p["n_components"]))
    elif spec.kind == "et":
        model = ExtraTreesClassifier(
            n_estimators=250, max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_samples_leaf"]),
            class_weight="balanced", max_features="sqrt", random_state=seed, n_jobs=6,
        )
    else:
        raise ValueError(spec.kind)
    model.fit(xfit, yfit)
    return model


def predict_prob(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    s = np.asarray(model.decision_function(x), dtype=float)
    return 1 / (1 + np.exp(-np.clip(s, -30, 30)))


def search_nonnegative_fusion(
    y: np.ndarray,
    probs: Dict[str, np.ndarray],
    keys: Sequence[str],
    step: float = 0.05,
    min_text: float = 0.0,
    min_aux: float = 0.10,
    min_active: int = 2,
    max_single: float = 0.85,
) -> Dict[str, object]:
    units = int(round(1 / step))
    best = None

    if len(keys) == 3:
        combos = ((i, j, units - i - j) for i in range(units + 1) for j in range(units - i + 1))
    elif len(keys) == 4:
        combos = (
            (i, j, k, units - i - j - k)
            for i in range(units + 1)
            for j in range(units - i + 1)
            for k in range(units - i - j + 1)
        )
    else:
        raise ValueError("Fusion supports 3 or 4 branches")
    for combo in combos:
        w = np.asarray(
            combo,
            dtype=float,
        ) / units

        wd = dict(
            zip(keys, w.tolist())
        )

        active = int(
            np.count_nonzero(w > 1e-12)
        )

        if active < int(min_active):
            continue

        if float(np.max(w)) > float(max_single):
            continue

        if (
            "text" in wd
            and wd["text"] < min_text
        ):
            continue

        if (
            sum(
                wd[k]
                for k in keys
                if k != "text"
            )
            < min_aux
        ):
            continue
        p = sum(wd[k] * probs[k] for k in keys)
        m = choose_threshold(y, p)
        score = m["selection_score"]
        if best is None or score > best["selection_score"]:
            best = {**m, "weights": wd, "prob": p}
    if best is None:

        p = probs["text"]
        m = choose_threshold(y, p)
        best = {**m, "weights": {k: 1.0 if k == "text" else 0.0 for k in keys}, "prob": p}
    return best


def json_dump(obj: object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
