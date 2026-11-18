#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cef_common import load_dataset, set_seed
from ceef2_core import load_extra_branch
from ceef7_utils import (
    best_f1_threshold,
    load_fold_manifest,
    make_inner_splits,
    metric_row,
    safe_auc,
    set_all_seeds,
    summarize_repeats,
)
from depression_architecture_models import build_architecture, count_trainable_parameters

EPS = 1e-8


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


class TopicPCAReducer:


    def __init__(self, components: int, seed: int):
        self.components = int(components)
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, x: np.ndarray) -> "TopicPCAReducer":
        flat = l2_normalize(x).reshape(-1, x.shape[-1])
        scaled = self.scaler.fit_transform(flat)
        k = min(self.components, scaled.shape[0] - 1, scaled.shape[1])
        if k < 2:
            raise ValueError("Insufficient samples for topic PCA")
        self.pca = PCA(
            n_components=k,
            whiten=False,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(scaled)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("TopicPCAReducer not fitted")
        n, t = x.shape[:2]
        flat = l2_normalize(x).reshape(-1, x.shape[-1])
        z = self.pca.transform(self.scaler.transform(flat))
        return z.reshape(n, t, -1).astype(np.float32)


class TopicEvidenceScaler:


    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, x: np.ndarray) -> "TopicEvidenceScaler":
        self.scaler.fit(np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1]))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        n, t = x.shape[:2]
        z = self.scaler.transform(np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1]))
        return z.reshape(n, t, -1).astype(np.float32)


@dataclass
class FeatureBuilder:
    text_pca: int
    audio_pca: int
    seed: int
    feature_set: str

    def __post_init__(self):
        self.text_reducer = TopicPCAReducer(self.text_pca, self.seed)
        self.audio_reducer = TopicPCAReducer(self.audio_pca, self.seed + 1)
        self.text_evidence_scaler = TopicEvidenceScaler()
        self.audio_evidence_scaler = TopicEvidenceScaler()

    def fit(
        self,
        text: np.ndarray,
        audio: np.ndarray,
        text_evidence: np.ndarray | None,
        audio_evidence: np.ndarray | None,
    ) -> "FeatureBuilder":
        self.text_reducer.fit(text)
        self.audio_reducer.fit(audio)
        if self.feature_set == "structured":
            if text_evidence is None or audio_evidence is None:
                raise ValueError("structured feature_set requires Observable-12 and Vocal-8")
            self.text_evidence_scaler.fit(text_evidence)
            self.audio_evidence_scaler.fit(audio_evidence)
        return self

    def transform(
        self,
        text: np.ndarray,
        audio: np.ndarray,
        text_evidence: np.ndarray | None,
        audio_evidence: np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        xt = self.text_reducer.transform(text)
        xa = self.audio_reducer.transform(audio)
        if self.feature_set == "structured":
            assert text_evidence is not None and audio_evidence is not None
            xt = np.concatenate([xt, self.text_evidence_scaler.transform(text_evidence)], axis=-1)
            xa = np.concatenate([xa, self.audio_evidence_scaler.transform(audio_evidence)], axis=-1)
        return xt.astype(np.float32), xa.astype(np.float32)


def make_loader(
    text: np.ndarray,
    audio: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    ds = TensorDataset(
        torch.from_numpy(text).float(),
        torch.from_numpy(audio).float(),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def predict(
    model: nn.Module,
    text: np.ndarray,
    audio: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    loader = make_loader(
        text, audio, np.zeros(len(text), dtype=np.float32), batch_size, False, 1
    )
    with torch.no_grad():
        for bt, ba, _ in loader:
            logits, _ = model(bt.to(device), ba.to(device))
            out.append(logits.detach().cpu().numpy())
    return np.concatenate(out).astype(float)


def zscore_from_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    mu = float(np.mean(reference))
    sd = max(float(np.std(reference)), EPS)
    return (np.asarray(values, dtype=float) - mu) / sd


def weighted_bce(y: np.ndarray, device: torch.device) -> nn.Module:
    y = np.asarray(y, dtype=int)
    positive = max(float(np.sum(y == 1)), 1.0)
    negative = max(float(np.sum(y == 0)), 1.0)
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device)
    )


def train_with_early_stopping(
    architecture: str,
    text_train: np.ndarray,
    audio_train: np.ndarray,
    y_train: np.ndarray,
    text_val: np.ndarray,
    audio_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int,
    dropout: float,
    heads: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, int, float]:
    set_all_seeds(seed)
    model = build_architecture(
        architecture,
        text_train.shape[-1],
        audio_train.shape[-1],
        hidden,
        dropout,
        heads,
    ).to(device)
    criterion = weighted_bce(y_train, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(text_train, audio_train, y_train, batch_size, True, seed)

    best_auc = -np.inf
    best_epoch = 1
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for bt, ba, by in loader:
            bt, ba, by = bt.to(device), ba.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bt, ba)
            loss = criterion(logits, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {architecture}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_score = predict(model, text_val, audio_val, device, batch_size)
        val_auc = safe_auc(y_val, val_score)
        if val_auc > best_auc + 1e-5:
            best_auc = float(val_auc)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Early stopping failed to retain a model")
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def train_fixed_epochs(
    architecture: str,
    text: np.ndarray,
    audio: np.ndarray,
    y: np.ndarray,
    hidden: int,
    dropout: float,
    heads: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_all_seeds(seed)
    model = build_architecture(
        architecture, text.shape[-1], audio.shape[-1], hidden, dropout, heads
    ).to(device)
    criterion = weighted_bce(y, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(text, audio, y, batch_size, True, seed)
    for _ in range(max(int(epochs), 1)):
        model.train()
        for bt, ba, by in loader:
            bt, ba, by = bt.to(device), ba.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bt, ba)
            loss = criterion(logits, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {architecture}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def prepare_features(
    builder: FeatureBuilder,
    indices: np.ndarray,
    text: np.ndarray,
    audio: np.ndarray,
    text_evidence: np.ndarray | None,
    audio_evidence: np.ndarray | None,
):
    te = None if text_evidence is None else text_evidence[indices]
    ae = None if audio_evidence is None else audio_evidence[indices]
    return builder.transform(text[indices], audio[indices], te, ae)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_npz", required=True)
    ap.add_argument("--text_npz", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--fold_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--feature_set", choices=["generic", "structured"], default="generic")
    ap.add_argument("--observable_npz", default="")
    ap.add_argument("--vocal_npz", default="")
    ap.add_argument("--architectures", default="acma_style,canamrf_at_style")
    ap.add_argument("--threshold", type=float, default=53.0)
    ap.add_argument("--text_pca", type=int, default=64)
    ap.add_argument("--audio_pca", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.hidden % args.heads != 0 or args.hidden % 2 != 0:
        raise ValueError("hidden must be even and divisible by heads")
    if args.feature_set == "structured" and (not args.observable_npz or not args.vocal_npz):
        raise ValueError("structured mode requires --observable_npz and --vocal_npz")

    set_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(
        args.audio_npz,
        args.text_npz,
        args.data_root,
        "",
        threshold=args.threshold,
        strict_expected=True,
    )
    ids = np.asarray(data["ids"])
    y = np.asarray(data["y"], dtype=int)
    text = np.asarray(data["text"], dtype=np.float64)
    audio = np.asarray(data["audio_sem"], dtype=np.float64)
    observable = (
        load_extra_branch(args.observable_npz, ids)
        if args.feature_set == "structured"
        else None
    )
    vocal = (
        load_extra_branch(args.vocal_npz, ids)
        if args.feature_set == "structured"
        else None
    )

    if text.ndim != 3 or audio.ndim != 3 or text.shape[1] != 3 or audio.shape[1] != 3:
        raise RuntimeError(f"Expected [N,3,D] inputs, got text={text.shape}, audio={audio.shape}")
    if observable is not None and observable.shape[:2] != text.shape[:2]:
        raise RuntimeError(f"Observable-12 alignment mismatch: {observable.shape}")
    if vocal is not None and vocal.shape[:2] != audio.shape[:2]:
        raise RuntimeError(f"Vocal-8 alignment mismatch: {vocal.shape}")

    manifest = load_fold_manifest(args.fold_json, ids)
    folds = list(manifest["folds"])
    repeats = 1 + max(int(item["repeat"]) for item in folds)
    architectures = [x.strip() for x in args.architectures.split(",") if x.strip()]
    model_names = [f"{a}__{args.feature_set}" for a in architectures]
    scores = {name: np.full((repeats, len(y)), np.nan, dtype=float) for name in model_names}
    hard = {name: np.zeros((repeats, len(y)), dtype=int) for name in model_names}
    fold_rows: List[Dict[str, object]] = []
    complexity_rows: List[Dict[str, object]] = []

    for outer_index, item in enumerate(folds):
        repeat = int(item["repeat"])
        fold = int(item["fold"])
        outer_train = np.asarray(item["train_idx"], dtype=int)
        outer_val = np.asarray(item["val_idx"], dtype=int)
        outer_seed = args.seed + 1009 * repeat + 131 * fold
        inner_splits = make_inner_splits(
            ids[outer_train], y[outer_train], args.inner_folds, outer_seed
        )

        for architecture_index, architecture in enumerate(architectures):
            model_name = f"{architecture}__{args.feature_set}"
            inner_oof = np.full(len(outer_train), np.nan, dtype=float)
            best_epochs: List[int] = []
            inner_aucs: List[float] = []

            for inner_fold, (inner_train_rel, inner_val_rel) in enumerate(inner_splits):
                inner_train = outer_train[np.asarray(inner_train_rel, dtype=int)]
                inner_val = outer_train[np.asarray(inner_val_rel, dtype=int)]
                inner_seed = outer_seed + 10000 * architecture_index + 503 * inner_fold
                builder = FeatureBuilder(
                    args.text_pca, args.audio_pca, inner_seed, args.feature_set
                ).fit(
                    text[inner_train],
                    audio[inner_train],
                    None if observable is None else observable[inner_train],
                    None if vocal is None else vocal[inner_train],
                )
                xt_train, xa_train = prepare_features(
                    builder, inner_train, text, audio, observable, vocal
                )
                xt_val, xa_val = prepare_features(
                    builder, inner_val, text, audio, observable, vocal
                )
                model, best_epoch, inner_auc = train_with_early_stopping(
                    architecture,
                    xt_train,
                    xa_train,
                    y[inner_train],
                    xt_val,
                    xa_val,
                    y[inner_val],
                    args.hidden,
                    args.dropout,
                    args.heads,
                    args.lr,
                    args.weight_decay,
                    args.batch_size,
                    args.max_epochs,
                    args.patience,
                    inner_seed,
                    device,
                )
                train_score = predict(model, xt_train, xa_train, device, args.batch_size)
                val_score = predict(model, xt_val, xa_val, device, args.batch_size)
                inner_oof[np.asarray(inner_val_rel, dtype=int)] = zscore_from_reference(
                    train_score, val_score
                )
                best_epochs.append(best_epoch)
                inner_aucs.append(inner_auc)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if np.any(~np.isfinite(inner_oof)):
                raise RuntimeError(f"Inner OOF coverage failure: {model_name}")
            selected_epoch = max(1, int(np.rint(np.median(best_epochs))))
            selected_threshold = best_f1_threshold(y[outer_train], inner_oof)

            outer_builder = FeatureBuilder(
                args.text_pca,
                args.audio_pca,
                outer_seed + 20000 * architecture_index,
                args.feature_set,
            ).fit(
                text[outer_train],
                audio[outer_train],
                None if observable is None else observable[outer_train],
                None if vocal is None else vocal[outer_train],
            )
            xt_train, xa_train = prepare_features(
                outer_builder, outer_train, text, audio, observable, vocal
            )
            xt_val, xa_val = prepare_features(
                outer_builder, outer_val, text, audio, observable, vocal
            )
            outer_model = train_fixed_epochs(
                architecture,
                xt_train,
                xa_train,
                y[outer_train],
                args.hidden,
                args.dropout,
                args.heads,
                args.lr,
                args.weight_decay,
                args.batch_size,
                selected_epoch,
                outer_seed + 30000 * architecture_index,
                device,
            )
            train_score = predict(outer_model, xt_train, xa_train, device, args.batch_size)
            val_score = zscore_from_reference(
                train_score,
                predict(outer_model, xt_val, xa_val, device, args.batch_size),
            )
            scores[model_name][repeat, outer_val] = val_score
            hard[model_name][repeat, outer_val] = (
                val_score >= selected_threshold
            ).astype(int)

            if not complexity_rows:
                pass
            complexity_rows.append(
                {
                    "feature_set": args.feature_set,
                    "architecture": architecture,
                    "repeat": repeat,
                    "fold": fold,
                    "text_input_dim": int(xt_train.shape[-1]),
                    "audio_input_dim": int(xa_train.shape[-1]),
                    "trainable_parameters": count_trainable_parameters(outer_model),
                }
            )
            fold_rows.append(
                {
                    "feature_set": args.feature_set,
                    "architecture": architecture,
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "n_outer_train": len(outer_train),
                    "n_outer_val": len(outer_val),
                    "inner_auc_mean": float(np.mean(inner_aucs)),
                    "inner_auc_std": float(np.std(inner_aucs, ddof=1)),
                    "inner_best_epochs": json.dumps(best_epochs),
                    "selected_epoch_median": selected_epoch,
                    "inner_oof_threshold": selected_threshold,
                    "outer_auc": safe_auc(y[outer_val], val_score),
                }
            )
            print(
                f"feature={args.feature_set} repeat={repeat+1}/{repeats} "
                f"fold={fold+1}/3 architecture={architecture} "
                f"epoch={selected_epoch} inner={np.mean(inner_aucs):.4f} "
                f"outer={safe_auc(y[outer_val], val_score):.4f}",
                flush=True,
            )
            del outer_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for name in model_names:
        if np.any(~np.isfinite(scores[name])):
            raise RuntimeError(f"Outer OOF coverage failure: {name}")

    prefix = f"architecture_{args.feature_set}"
    pd.DataFrame(fold_rows).to_csv(out / f"{prefix}_fold_metrics.csv", index=False)
    complexity = (
        pd.DataFrame(complexity_rows)
        .groupby(["feature_set", "architecture"], as_index=False)
        .agg(
            text_input_dim=("text_input_dim", "first"),
            audio_input_dim=("audio_input_dim", "first"),
            trainable_parameters=("trainable_parameters", "first"),
        )
    )
    complexity.to_csv(out / f"{prefix}_complexity.csv", index=False)

    repeat_rows: List[Dict[str, object]] = []
    subject = pd.DataFrame({"subject_id": ids, "label": y})
    subject_rows: List[Dict[str, object]] = []
    for name in model_names:
        for repeat in range(repeats):
            row = metric_row(y, scores[name][repeat], pred=hard[name][repeat])
            row.update(
                {
                    "feature_set": args.feature_set,
                    "model": name,
                    "architecture": name.split("__", 1)[0],
                    "repeat": repeat,
                }
            )
            repeat_rows.append(row)
        ensemble_score = scores[name].mean(axis=0)
        ensemble_pred = (hard[name].mean(axis=0) >= 0.5).astype(int)
        subject[f"score_{name}"] = ensemble_score
        subject[f"pred_{name}"] = ensemble_pred
        row = metric_row(y, ensemble_score, pred=ensemble_pred)
        row.update(
            {
                "feature_set": args.feature_set,
                "model": name,
                "architecture": name.split("__", 1)[0],
                "evaluation_unit": "subject_after_repeat_ensemble",
            }
        )
        subject_rows.append(row)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out / f"{prefix}_repeat_metrics.csv", index=False)
    summary = summarize_repeats(repeat_df, "model")
    summary["feature_set"] = args.feature_set
    summary["architecture"] = summary["model"].str.split("__").str[0]
    summary = summary.merge(complexity, on=["feature_set", "architecture"], how="left")
    summary.to_csv(out / f"{prefix}_repeat_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(out / f"{prefix}_subject_results.csv", index=False)
    subject.to_csv(out / f"{prefix}_subject_scores.csv", index=False)
    np.savez_compressed(
        out / f"{prefix}_repeat_scores.npz",
        subject_ids=ids,
        labels=y,
        **{f"score_{name}": scores[name] for name in model_names},
    )

    print(f"\n===== {args.feature_set.upper()} DEPRESSION ARCHITECTURES =====")
    print(
        summary[
            [
                "model",
                "auc_mean",
                "auc_std",
                "f1_mean",
                "f1_std",
                "bacc_mean",
                "bacc_std",
                "trainable_parameters",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
