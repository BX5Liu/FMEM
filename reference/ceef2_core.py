#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-8


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def topic_contrast(x: np.ndarray) -> np.ndarray:

    z = l2norm(np.asarray(x, dtype=np.float64))
    p, n, g = z[:, 0], z[:, 1], z[:, 2]
    return np.concatenate([p, n, g, g - p, g - n, n - p], axis=1)


class BranchRep:


    def __init__(self, mode: str = "contrast", n_comp: int = 0):
        self.mode = mode
        self.n_comp = int(n_comp)
        self.pca: Optional[PCA] = None
        self.mu: Optional[np.ndarray] = None

    def _base(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "flat":
            return l2norm(x)
        if self.mode == "concat":
            return l2norm(x).reshape(len(x), -1)
        if self.mode == "mean":

            z = l2norm(x)
            return l2norm(z.mean(axis=1))
        return topic_contrast(x)

    def fit(self, x: np.ndarray) -> "BranchRep":
        z = self._base(x)
        self.mu = z.mean(axis=0, keepdims=True)
        if self.n_comp > 0 and self.n_comp < min(z.shape):
            self.pca = PCA(n_components=self.n_comp, whiten=False, random_state=0)
            self.pca.fit(z - self.mu)
        else:
            self.pca = None
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self._base(x)
        if self.pca is not None:
            z = self.pca.transform(z - self.mu)
        return np.nan_to_num(z)


@dataclass(frozen=True)
class BranchSpec:
    rep_mode: str = "contrast"
    n_comp: int = 0
    c: float = 0.1

    @property
    def name(self) -> str:
        return f"{self.rep_mode}_p{self.n_comp}_C{self.c}"


class BranchModel:
    def __init__(self, spec: BranchSpec, seed: int = 42):
        self.spec = spec
        self.seed = int(seed)
        self.rep = BranchRep(spec.rep_mode, spec.n_comp)
        self.scaler = StandardScaler()
        self.clf: Optional[LogisticRegression] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BranchModel":
        z = self.scaler.fit_transform(self.rep.fit(x).transform(x))
        self.clf = LogisticRegression(
            C=float(self.spec.c), solver="liblinear",
            class_weight="balanced", max_iter=5000, random_state=self.seed,
        ).fit(z, np.asarray(y, dtype=int))
        return self

    def decision(self, x: np.ndarray) -> np.ndarray:

        assert self.clf is not None
        z = self.scaler.transform(self.rep.transform(x))
        return np.asarray(self.clf.decision_function(z), dtype=float).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(x), -30, 30)))

    def decision_target_aligned(self, x: np.ndarray) -> np.ndarray:


        assert self.clf is not None
        z = self.rep.transform(x)
        mu = z.mean(axis=0, keepdims=True)
        sd = np.maximum(z.std(axis=0, keepdims=True), EPS)
        return np.asarray(self.clf.decision_function((z - mu) / sd), dtype=float).reshape(-1)

    def source_shift_report(self, x: np.ndarray) -> Dict[str, float]:

        z = self.scaler.transform(self.rep.transform(x))
        per_dim = np.abs(z.mean(axis=0))
        return {
            "mean_abs_z": float(per_dim.mean()),
            "frac_dims_shift_gt2": float((per_dim > 2.0).mean()),
            "max_abs_z": float(per_dim.max()),
        }


def branch_grid(dim: int, is_flat: bool) -> List[BranchSpec]:


    if is_flat:
        return [BranchSpec("flat", 0, c) for c in (0.01, 0.03, 0.1, 0.3)]
    specs: List[BranchSpec] = []
    for c in (0.01, 0.03, 0.1, 0.3):
        specs.append(BranchSpec("mean", 0, c))
        specs.append(BranchSpec("concat", 0, c))
        specs.append(BranchSpec("contrast", 0, c))
    if dim > 64:
        for c in (0.03, 0.1):
            specs.append(BranchSpec("concat", 64, c))
            specs.append(BranchSpec("contrast", 64, c))
    return specs


def f1_at(y, s, thr) -> float:
    return float(f1_score(y, (s >= thr).astype(int), zero_division=0))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 193)))
    return float(grid[int(np.argmax([f1_at(y, s, t) for t in grid]))])


def prior_threshold(s: np.ndarray, prevalence: float, boost: float = 1.0) -> float:


    q = float(np.clip(1.0 - boost * prevalence, 0.02, 0.98))
    return float(np.quantile(s, q))


def full_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s >= thr).astype(int)
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5,
        "bacc": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppr": float(pred.mean()),
        "acc": float((pred == y).mean()),
        "threshold": float(thr),
    }


def simplex_weights(step: float, k: int) -> np.ndarray:
    units = int(round(1.0 / step))

    def rec(remaining: int, slots: int):
        if slots == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in rec(remaining - i, slots - 1):
                yield (i,) + rest

    return np.asarray(list(rec(units, k)), dtype=float) / units


def search_guarded_fusion(
    y: np.ndarray,
    logits: Dict[str, np.ndarray],
    anchor: str,
    residuals: Sequence[str],
    step: float = 0.05,
    residual_cap: float = 0.25,
    residual_total: float = 0.40,
    margin: float = 0.005,
) -> Dict[str, object]:


    z = {k: (logits[k] - logits[k].mean()) / max(logits[k].std(), EPS)
         for k in [anchor, *residuals]}
    prevalence = float(np.mean(y))

    def crit(sv: np.ndarray) -> float:
        auc = float(roc_auc_score(y, sv)) if len(np.unique(y)) > 1 else 0.5
        bacc = float(balanced_accuracy_score(
            y, (sv >= prior_threshold(sv, prevalence)).astype(int)))
        return 0.6 * auc + 0.4 * bacc

    base = crit(z[anchor])
    best = {"score": base, "weights": {anchor: 1.0, **{k: 0.0 for k in residuals}},
            "anchor_only": True, "anchor_score": base}
    if not residuals:
        return best

    units = int(round(residual_cap / step))
    total_units = int(round(residual_total / step))

    def combos(slots: int):
        if slots == 0:
            yield ()
            return
        for i in range(units + 1):
            for rest in combos(slots - 1):
                if i + sum(rest) <= total_units:
                    yield (i,) + rest

    for combo in combos(len(residuals)):
        lam = [c * step for c in combo]
        sv = (1.0 - sum(lam)) * z[anchor]
        for w, k in zip(lam, residuals):
            sv = sv + w * z[k]
        sc = crit(sv)
        if sc > best["score"] and sc >= base + margin:
            best = {"score": sc,
                    "weights": {anchor: 1.0 - sum(lam),
                                **{k: float(w) for k, w in zip(residuals, lam)}},
                    "anchor_only": False, "anchor_score": base}
    return best


def search_fusion(
    y: np.ndarray,
    logits: Dict[str, np.ndarray],
    keys: Sequence[str],
    step: float = 0.1,
    min_branches: int = 1,
    max_single: float = 1.0,
) -> Dict[str, object]:


    z = {k: (logits[k] - logits[k].mean()) / max(logits[k].std(), EPS) for k in keys}
    prevalence = float(np.mean(y))
    grid = simplex_weights(step, len(keys))
    best = None
    for w in grid:
        if int(np.count_nonzero(w > 1e-9)) < min_branches:
            continue
        if float(w.max()) > max_single:
            continue
        s = sum(w[i] * z[k] for i, k in enumerate(keys))
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5
        thr = prior_threshold(s, prevalence)
        bacc = float(balanced_accuracy_score(y, (s >= thr).astype(int)))
        score = 0.6 * auc + 0.4 * bacc
        if best is None or score > best["score"]:
            best = {"score": score, "weights": {k: float(w[i]) for i, k in enumerate(keys)}}
    if best is None:
        best = {"score": 0.0, "weights": {k: 1.0 / len(keys) for k in keys}}
    return best


def apply_fusion(
    logits: Dict[str, np.ndarray],
    weights: Dict[str, float],
    stats: Dict[str, Tuple[float, float]],
) -> np.ndarray:

    out = np.zeros(len(next(iter(logits.values()))), dtype=float)
    for k, w in weights.items():
        mu, sd = stats[k]
        out += w * (logits[k] - mu) / max(sd, EPS)
    return out


@dataclass
class DevResult:
    branch_logits: Dict[str, np.ndarray] = field(default_factory=dict)
    fused_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nested_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))
    branch_specs: Dict[str, BranchSpec] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    fusion_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)


def _select_branch_spec(
    raw: np.ndarray, y: np.ndarray, is_flat: bool, seed: int, folds: int = 3
) -> Tuple[BranchSpec, float]:

    best, best_auc = None, -1.0
    for spec in branch_grid(raw.shape[-1], is_flat):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        s = np.full(len(y), np.nan)
        try:
            for tr, va in skf.split(np.zeros(len(y)), y):
                m = BranchModel(spec, seed).fit(raw[tr], y[tr])
                s[va] = m.decision(raw[va])
            auc = float(roc_auc_score(y, s))
        except Exception:
            continue
        if auc > best_auc:
            best, best_auc = spec, auc
    return (best or BranchSpec()), best_auc


def run_development(
    branches: Dict[str, Tuple[np.ndarray, bool]],
    y: np.ndarray,
    repeats: int = 5,
    folds: int = 3,
    seed: int = 42,
    fusion_step: float = 0.1,
    fusion_mode: str = "free",
    anchor: str = "text",
    residual_cap: float = 0.25,
    residual_total: float = 0.40,
    locked_specs: Optional[Dict[str, BranchSpec]] = None,
    verbose: bool = True,
) -> DevResult:
    n = len(y)
    keys = list(branches.keys())
    prevalence = float(y.mean())

    oof_logit = {k: np.full((repeats, n), np.nan) for k in keys}
    oof_fused = np.full((repeats, n), np.nan)
    nested_hard = np.full((repeats, n), np.nan)

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1009 * r)
        for tr, va in skf.split(np.zeros(n), y):

            inner_logit = {k: np.full(len(tr), np.nan) for k in keys}
            spec_by_branch: Dict[str, BranchSpec] = {}
            for k in keys:
                raw, is_flat = branches[k]
                if locked_specs and k in locked_specs:


                    spec = locked_specs[k]
                else:
                    spec, _ = _select_branch_spec(raw[tr], y[tr], is_flat,
                                                  seed + r, folds)
                spec_by_branch[k] = spec
                inner = StratifiedKFold(n_splits=folds, shuffle=True,
                                        random_state=seed + 77 * r)
                for itr, iva in inner.split(np.zeros(len(tr)), y[tr]):
                    m = BranchModel(spec, seed + r).fit(raw[tr][itr], y[tr][itr])
                    inner_logit[k][iva] = m.decision(raw[tr][iva])

            if fusion_mode == "guarded":
                res_keys = [k for k in keys if k != anchor]
                fuse = search_guarded_fusion(
                    y[tr], inner_logit, anchor, res_keys, step=0.05,
                    residual_cap=residual_cap, residual_total=residual_total)
            else:
                fuse = search_fusion(y[tr], inner_logit, keys, step=fusion_step)
            w = fuse["weights"]
            stats = {k: (float(inner_logit[k].mean()), float(inner_logit[k].std()))
                     for k in keys}
            s_inner = apply_fusion(inner_logit, w, stats)
            thr_inner = best_f1_threshold(y[tr], s_inner)


            outer_logit, outer_stats = {}, {}
            for k in keys:
                raw, _ = branches[k]
                m = BranchModel(spec_by_branch[k], seed + r).fit(raw[tr], y[tr])
                tr_logit = m.decision(raw[tr])
                mu, sd = float(tr_logit.mean()), float(tr_logit.std())
                outer_logit[k] = m.decision(raw[va])
                outer_stats[k] = (mu, sd)

                oof_logit[k][r, va] = (outer_logit[k] - mu) / max(sd, EPS)
            s_va = apply_fusion(outer_logit, w, outer_stats)
            oof_fused[r, va] = s_va
            nested_hard[r, va] = (s_va >= thr_inner).astype(float)

        if verbose:
            print(f"  repeat {r + 1}/{repeats} done", flush=True)

    res = DevResult()
    res.branch_logits = {k: np.nanmean(v, axis=0) for k, v in oof_logit.items()}
    res.fused_logits = np.nanmean(oof_fused, axis=0)
    res.nested_pred = (np.nanmean(nested_hard, axis=0) >= 0.5).astype(int)

    report: Dict[str, object] = {}
    for k in keys:
        s = res.branch_logits[k]
        report[k] = {
            "opt": full_metrics(y, s, best_f1_threshold(y, s)),
            "prior": full_metrics(y, s, prior_threshold(s, prevalence)),
        }
    s = res.fused_logits
    report["fused"] = {
        "opt": full_metrics(y, s, best_f1_threshold(y, s)),
        "prior": full_metrics(y, s, prior_threshold(s, prevalence)),
        "nested_f1": float(f1_score(y, res.nested_pred, zero_division=0)),
        "nested_bacc": float(balanced_accuracy_score(y, res.nested_pred)),
        "nested_ppr": float(res.nested_pred.mean()),
    }
    res.report = report


    final_logit = {}
    for k in keys:
        raw, is_flat = branches[k]
        if locked_specs and k in locked_specs:
            spec = locked_specs[k]
        else:
            spec, _ = _select_branch_spec(raw, y, is_flat, seed, folds)
        res.branch_specs[k] = spec
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        s_k = np.full(n, np.nan)
        for tr, va in skf.split(np.zeros(n), y):
            m = BranchModel(spec, seed).fit(raw[tr], y[tr])
            tr_logit = m.decision(raw[tr])
            s_k[va] = (m.decision(raw[va]) - tr_logit.mean()) / max(tr_logit.std(), EPS)
        final_logit[k] = s_k
    if fusion_mode == "guarded":
        res_keys = [k for k in keys if k != anchor]
        fuse = search_guarded_fusion(y, final_logit, anchor, res_keys, step=0.05,
                                     residual_cap=residual_cap,
                                     residual_total=residual_total)
    else:
        fuse = search_fusion(y, final_logit, keys, step=fusion_step)
    res.fusion_weights = fuse["weights"]


    res.fusion_stats = {}
    for k in keys:
        raw, _ = branches[k]
        m = BranchModel(res.branch_specs[k], seed).fit(raw, y)
        tl = m.decision(raw)
        res.fusion_stats[k] = (float(tl.mean()), float(tl.std()))
    return res


def load_extra_branch(path: str, ids: Sequence[str]) -> np.ndarray:

    import re as _re

    def _norm(x: str) -> str:
        return _re.sub(r"[^0-9a-z]", "", str(x).strip().lower())

    z = np.load(path, allow_pickle=True)
    key = "subject_ids" if "subject_ids" in z else "ids"
    src = [str(x).strip() for x in z[key]]
    pos = {s: i for i, s in enumerate(src)}
    pos.update({_norm(s): i for i, s in enumerate(src)})
    order = []
    for s in ids:
        j = pos.get(str(s).strip(), pos.get(_norm(s), -1))
        if j < 0:
            raise RuntimeError(f"{path} 里找不到受试者 {s}")
        order.append(j)
    prof = np.asarray(z["profiles"], dtype=np.float64)[order]
    if prof.ndim == 2:
        prof = prof[:, None, :]
    return np.nan_to_num(prof)


def parse_locked_specs(items: Sequence[str]) -> Dict[str, BranchSpec]:

    out: Dict[str, BranchSpec] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--branch_spec 需要 NAME=mode:ncomp:C，收到 {it}")
        name, body = it.split("=", 1)
        parts = body.split(":")
        if len(parts) != 3:
            raise ValueError(f"--branch_spec 需要 NAME=mode:ncomp:C，收到 {it}")
        mode, ncomp, c = parts[0].strip(), int(parts[1]), float(parts[2])
        if mode not in ("mean", "concat", "contrast", "flat"):
            raise ValueError(f"未知 mode: {mode}")
        out[name.strip()] = BranchSpec(mode, ncomp, c)
    return out
