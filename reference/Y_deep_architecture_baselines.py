#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import copy
import json
import math
import warnings
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cef_common import load_dataset, set_seed
from ceef7_utils import (
    best_f1_threshold,
    load_fold_manifest,
    make_inner_splits,
    metric_row,
    safe_auc,
    set_all_seeds,
    summarize_repeats,
)

PERMS = list(permutations(range(3)))
EPS = 1e-8


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


class TopicReducer:
    def __init__(self, n_components: int, seed: int):
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, x: np.ndarray) -> "TopicReducer":
        z = l2(x).reshape(-1, x.shape[-1])
        zs = self.scaler.fit_transform(z)
        k = min(self.n_components, zs.shape[0] - 1, zs.shape[1])
        if k < 2:
            raise ValueError("Not enough samples for PCA")
        self.pca = PCA(n_components=k, whiten=False, svd_solver="randomized", random_state=self.seed)
        self.pca.fit(zs)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        n = len(x)
        z = l2(x).reshape(-1, x.shape[-1])
        out = self.pca.transform(self.scaler.transform(z))
        return out.reshape(n, 3, -1).astype(np.float32)


class EATDGRUBiLSTM(nn.Module):
    def __init__(self, dt: int, da: int, hidden: int, dropout: float):
        super().__init__()
        self.text = nn.LSTM(dt, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.audio = nn.GRU(da, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, xt, xa):
        to, _ = self.text(xt)
        ao, _ = self.audio(xa)
        t = to.mean(dim=1)
        a = ao.mean(dim=1)
        return self.head(torch.cat([t, a], dim=-1)).squeeze(-1), {}


class ModalAttentionFusion(nn.Module):
    def __init__(self, dt: int, da: int, hidden: int, dropout: float):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(dt, hidden), nn.Tanh())
        self.aproj = nn.Sequential(nn.Linear(da, hidden), nn.Tanh())
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xt, xa):
        t = self.tproj(xt).mean(dim=1)
        a = self.aproj(xa).mean(dim=1)
        m = torch.stack([t, a], dim=1)
        w = torch.softmax(self.attn(m).squeeze(-1), dim=1)
        fused = (m * w.unsqueeze(-1)).sum(dim=1)
        return self.head(fused).squeeze(-1), {"modal_weights": w}


class LightweightMulT(nn.Module):
    def __init__(self, dt: int, da: int, hidden: int, dropout: float, heads: int = 4):
        super().__init__()
        self.tproj = nn.Linear(dt, hidden)
        self.aproj = nn.Linear(da, hidden)
        self.t_from_a = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.a_from_t = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xt, xa):
        t, a = self.tproj(xt), self.aproj(xa)
        tc, _ = self.t_from_a(t, a, a)
        ac, _ = self.a_from_t(a, t, t)
        tc = self.encoder(t + tc).mean(dim=1)
        ac = self.encoder(a + ac).mean(dim=1)
        return self.head(torch.cat([tc, ac], dim=-1)).squeeze(-1), {}


class MISAStyle(nn.Module):
    def __init__(self, dt: int, da: int, hidden: int, dropout: float):
        super().__init__()
        self.tbase = nn.Sequential(nn.Linear(dt, hidden), nn.ReLU())
        self.abase = nn.Sequential(nn.Linear(da, hidden), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.private_t = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.private_a = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.rec_t = nn.Linear(hidden * 2, hidden)
        self.rec_a = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, xt, xa):
        tb = self.tbase(xt).mean(dim=1)
        ab = self.abase(xa).mean(dim=1)
        st, sa = self.shared(tb), self.shared(ab)
        pt, pa = self.private_t(tb), self.private_a(ab)
        logits = self.head(torch.cat([st, sa, pt, pa], dim=-1)).squeeze(-1)
        return logits, {"tb": tb, "ab": ab, "st": st, "sa": sa, "pt": pt, "pa": pa,
                        "rt": self.rec_t(torch.cat([st, pt], dim=-1)),
                        "ra": self.rec_a(torch.cat([sa, pa], dim=-1))}


def build_model(name: str, dt: int, da: int, hidden: int, dropout: float) -> nn.Module:
    if name == "eatd_gru_bilstm_style":
        return EATDGRUBiLSTM(dt, da, hidden, dropout)
    if name == "modal_attention":
        return ModalAttentionFusion(dt, da, hidden, dropout)
    if name == "lightweight_mult":
        return LightweightMulT(dt, da, hidden, dropout)
    if name == "misa_style":
        return MISAStyle(dt, da, hidden, dropout)
    raise ValueError(name)


def augment_training(xt: np.ndarray, xa: np.ndarray, y: np.ndarray,
                     protocol: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if protocol == "honest":
        return xt, xa, y
    if protocol != "thesis_train_aug":
        raise ValueError(protocol)
    rng = np.random.default_rng(seed)
    tx, ax, yy = [], [], []
    for i in range(len(y)):
        if y[i] == 1:
            perms = PERMS
        else:
            perms = [PERMS[int(rng.integers(0, len(PERMS)))]]
        for p in perms:
            tx.append(xt[i, list(p)])
            ax.append(xa[i, list(p)])
            yy.append(y[i])
    return np.asarray(tx), np.asarray(ax), np.asarray(yy, dtype=int)


def aux_loss(name: str, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    if name != "misa_style":
        return torch.tensor(0.0, device=next(iter(aux.values())).device) if aux else torch.tensor(0.0)
    recon = torch.mean((aux["rt"] - aux["tb"]) ** 2) + torch.mean((aux["ra"] - aux["ab"]) ** 2)
    shared = torch.mean((aux["st"] - aux["sa"]) ** 2)
    def decor(a, b):
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        return torch.mean((a.T @ b / max(len(a), 1)) ** 2)
    diff = decor(aux["st"], aux["pt"]) + decor(aux["sa"], aux["pa"])
    return 0.10 * recon + 0.05 * shared + 0.05 * diff


def make_loader(xt, xa, y, batch_size, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(xt).float(), torch.from_numpy(xa).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


def predict(model: nn.Module, xt: np.ndarray, xa: np.ndarray, device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    vals = []
    loader = make_loader(xt, xa, np.zeros(len(xt), dtype=np.float32), batch_size, False, 1)
    with torch.no_grad():
        for bt, ba, _ in loader:
            logits, _ = model(bt.to(device), ba.to(device))
            vals.append(logits.detach().cpu().numpy())
    return np.concatenate(vals).astype(float)


def train_with_validation(name: str, xt_train, xa_train, y_train,
                          xt_val, xa_val, y_val, hidden, dropout, lr, weight_decay,
                          batch_size, max_epochs, patience, seed, device):
    set_all_seeds(seed)
    model = build_model(name, xt_train.shape[-1], xa_train.shape[-1], hidden, dropout).to(device)
    pos = max(float(np.sum(y_train == 1)), 1.0)
    neg = max(float(np.sum(y_train == 0)), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(xt_train, xa_train, y_train, batch_size, True, seed)
    best_auc, best_epoch, best_state, stale = -1.0, 1, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for bt, ba, by in loader:
            bt, ba, by = bt.to(device), ba.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(bt, ba)
            loss = criterion(logits, by) + aux_loss(name, aux).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_s = predict(model, xt_val, xa_val, device, batch_size)
        auc = safe_auc(y_val, val_s)
        if auc > best_auc + 1e-5:
            best_auc, best_epoch = auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, int(best_epoch), float(best_auc)


def train_fixed_epochs(name: str, xt, xa, y, hidden, dropout, lr, weight_decay,
                       batch_size, epochs, seed, device):
    set_all_seeds(seed)
    model = build_model(name, xt.shape[-1], xa.shape[-1], hidden, dropout).to(device)
    pos = max(float(np.sum(y == 1)), 1.0)
    neg = max(float(np.sum(y == 0)), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(xt, xa, y, batch_size, True, seed)
    for _ in range(max(int(epochs), 1)):
        model.train()
        for bt, ba, by in loader:
            bt, ba, by = bt.to(device), ba.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(bt, ba)
            loss = criterion(logits, by) + aux_loss(name, aux).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def zscore(train_s: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (s - np.mean(train_s)) / max(np.std(train_s), EPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_npz", required=True)
    ap.add_argument("--text_npz", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--evidence_npz", default="")
    ap.add_argument("--fold_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--protocol", choices=["honest", "thesis_train_aug"], default="honest")
    ap.add_argument("--architectures", default="eatd_gru_bilstm_style,modal_attention,lightweight_mult,misa_style")
    ap.add_argument("--threshold", type=float, default=53.0)
    ap.add_argument("--text_pca", type=int, default=96)
    ap.add_argument("--audio_pca", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args.audio_npz, args.text_npz, args.data_root,
                        args.evidence_npz, threshold=args.threshold, strict_expected=True)
    ids = np.asarray(data["ids"])
    y = np.asarray(data["y"], dtype=int)
    text = np.asarray(data["text"], dtype=np.float64)
    audio = np.asarray(data["audio_sem"], dtype=np.float64)
    manifest = load_fold_manifest(args.fold_json, ids)
    folds = manifest["folds"]
    repeats = max(int(x["repeat"]) for x in folds) + 1
    archs = [x.strip() for x in args.architectures.split(",") if x.strip()]
    n = len(y)
    scores = {a: np.zeros((repeats, n), dtype=float) for a in archs}
    hard = {a: np.zeros((repeats, n), dtype=int) for a in archs}
    fold_rows = []

    for item in folds:
        r, f = int(item["repeat"]), int(item["fold"])
        tr = np.asarray(item["train_idx"], dtype=int)
        va = np.asarray(item["val_idx"], dtype=int)
        fold_seed = args.seed + 1009 * r + 131 * f

        inner_tr_rel, inner_va_rel = make_inner_splits(ids[tr], y[tr], args.inner_folds, fold_seed)[0]
        inner_tr, inner_va = tr[inner_tr_rel], tr[inner_va_rel]

        rt_i = TopicReducer(args.text_pca, fold_seed).fit(text[inner_tr])
        ra_i = TopicReducer(args.audio_pca, fold_seed + 1).fit(audio[inner_tr])
        xt_i_tr, xa_i_tr = rt_i.transform(text[inner_tr]), ra_i.transform(audio[inner_tr])
        xt_i_va, xa_i_va = rt_i.transform(text[inner_va]), ra_i.transform(audio[inner_va])
        xt_i_aug, xa_i_aug, y_i_aug = augment_training(
            xt_i_tr, xa_i_tr, y[inner_tr], args.protocol, fold_seed + 2)

        rt_o = TopicReducer(args.text_pca, fold_seed + 3).fit(text[tr])
        ra_o = TopicReducer(args.audio_pca, fold_seed + 4).fit(audio[tr])
        xt_o_tr, xa_o_tr = rt_o.transform(text[tr]), ra_o.transform(audio[tr])
        xt_o_va, xa_o_va = rt_o.transform(text[va]), ra_o.transform(audio[va])
        xt_o_aug, xa_o_aug, y_o_aug = augment_training(
            xt_o_tr, xa_o_tr, y[tr], args.protocol, fold_seed + 5)

        for ai, arch in enumerate(archs):
            inner_model, best_epoch, inner_auc = train_with_validation(
                arch, xt_i_aug, xa_i_aug, y_i_aug, xt_i_va, xa_i_va, y[inner_va],
                args.hidden, args.dropout, args.lr, args.weight_decay,
                args.batch_size, args.max_epochs, args.patience,
                fold_seed + 100 * ai, device)
            inner_train_s = predict(inner_model, xt_i_tr, xa_i_tr, device, args.batch_size)
            inner_val_s = predict(inner_model, xt_i_va, xa_i_va, device, args.batch_size)
            inner_val_z = zscore(inner_train_s, inner_val_s)
            threshold = best_f1_threshold(y[inner_va], inner_val_z)

            outer_model = train_fixed_epochs(
                arch, xt_o_aug, xa_o_aug, y_o_aug,
                args.hidden, args.dropout, args.lr, args.weight_decay,
                args.batch_size, best_epoch, fold_seed + 100 * ai + 50, device)
            outer_train_s = predict(outer_model, xt_o_tr, xa_o_tr, device, args.batch_size)
            outer_val_s = predict(outer_model, xt_o_va, xa_o_va, device, args.batch_size)
            outer_val_z = zscore(outer_train_s, outer_val_s)
            scores[arch][r, va] = outer_val_z
            hard[arch][r, va] = (outer_val_z >= threshold).astype(int)
            fold_rows.append({
                "protocol": args.protocol, "architecture": arch,
                "repeat": r, "fold": f, "best_epoch": best_epoch,
                "inner_auc": inner_auc, "inner_threshold": threshold,
                "outer_auc": safe_auc(y[va], outer_val_z),
                "n_train_original": len(tr), "n_train_after_augmentation": len(y_o_aug),
            })
            print(f"{args.protocol} r={r+1}/{repeats} f={f+1}/3 {arch} "
                  f"epoch={best_epoch} inner={inner_auc:.4f} outer={safe_auc(y[va], outer_val_z):.4f}", flush=True)
            del inner_model, outer_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(fold_rows).to_csv(out / f"deep_{args.protocol}_folds.csv", index=False)
    repeat_rows, subject_rows = [], []
    pred_out = {"subject_id": ids, "label": y}
    for arch in archs:
        for r in range(repeats):
            row = metric_row(y, scores[arch][r], pred=hard[arch][r])
            row.update({"protocol": args.protocol, "architecture": arch, "repeat": r})
            repeat_rows.append(row)
        s = scores[arch].mean(axis=0)
        p = (hard[arch].mean(axis=0) >= 0.5).astype(int)
        row = metric_row(y, s, pred=p)
        row.update({"protocol": args.protocol, "architecture": arch,
                    "evaluation_unit": "subject_after_repeat_ensemble"})
        subject_rows.append(row)
        pred_out[f"score_{arch}"] = s
        pred_out[f"pred_{arch}"] = p
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out / f"deep_{args.protocol}_repeat_metrics.csv", index=False)
    summarize_repeats(repeat_df, "architecture").to_csv(
        out / f"deep_{args.protocol}_repeat_summary.csv", index=False)
    subject_df = pd.DataFrame(subject_rows).sort_values("auc", ascending=False)
    subject_df.to_csv(out / f"deep_{args.protocol}_subject_ensemble.csv", index=False)
    pd.DataFrame(pred_out).to_csv(out / f"deep_{args.protocol}_predictions.csv", index=False)
    print("\n", subject_df[["architecture", "auc", "f1", "bacc"]].to_string(index=False), sep="")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
