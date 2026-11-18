#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

TOPICS = ("positive", "neutral", "negative")


def norm_id(x: object) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    return str(x).strip()


def load_ids_labels(npz_path: str | Path, threshold: float = 53.0) -> Tuple[List[str], np.ndarray]:
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as z:
        id_key = next((k for k in ("positive_ids", "subject_ids", "ids") if k in z.files), None)
        if id_key is None:
            raise KeyError(f"{p} has no subject ID key; keys={z.files}")
        label_key = next((k for k in ("positive_labels", "labels", "y") if k in z.files), None)
        if label_key is None:
            raise KeyError(f"{p} has no label key; keys={z.files}")
        ids = [norm_id(x) for x in np.asarray(z[id_key]).reshape(-1)]
        raw = np.asarray(z[label_key]).reshape(-1)

    if len(ids) != len(raw):
        raise ValueError(f"ID/label length mismatch in {p}: {len(ids)} vs {len(raw)}")
    if len(set(ids)) != len(ids):
        seen, dup = set(), []
        for sid in ids:
            if sid in seen and sid not in dup:
                dup.append(sid)
            seen.add(sid)
        raise ValueError(f"Duplicate IDs in {p}: {dup[:20]}")

    vals = np.asarray(raw, dtype=float)
    uniq = set(np.unique(vals).tolist())
    labels = vals.astype(int) if uniq.issubset({0.0, 1.0}) else (vals >= threshold).astype(int)
    return ids, labels


def build_folder_index(root: str | Path) -> Dict[str, List[Path]]:

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = set()
    for name in ("positive.txt", "neutral.txt", "negative.txt", "positive.wav", "new_label.txt", "label.txt"):
        for p in root.rglob(name):
            candidates.add(p.parent)
    out: Dict[str, List[Path]] = {}
    for folder in sorted(candidates):
        out.setdefault(norm_id(folder.name), []).append(folder)
    return out


def read_text_bytes_signature(folder: Path) -> str:
    h = hashlib.sha256()
    for topic in TOPICS:
        p = folder / f"{topic}.txt"
        h.update(topic.encode("utf-8"))
        h.update(p.read_bytes() if p.exists() else b"<MISSING>")
    for name in ("new_label.txt", "label.txt"):
        p = folder / name
        if p.exists():
            h.update(p.read_bytes())
            break
    return h.hexdigest()


def completeness_score(folder: Path) -> int:
    score = 0
    for topic in TOPICS:
        score += int((folder / f"{topic}.txt").exists())
        score += int((folder / f"{topic}.wav").exists())
    score += int((folder / "new_label.txt").exists() or (folder / "label.txt").exists())
    return score


def resolve_folders(
    ids: Iterable[str],
    root: str | Path,
) -> Tuple[Dict[str, Path], List[dict]]:

    root = Path(root)
    index = build_folder_index(root)
    resolved: Dict[str, Path] = {}
    diagnostics: List[dict] = []

    for sid0 in ids:
        sid = norm_id(sid0)
        cands = index.get(sid, [])
        if not cands:
            diagnostics.append({
                "subject_id": sid,
                "status": "missing",
                "chosen": "",
                "n_candidates": 0,
                "signature_conflict": False,
                "all_candidates": "",
            })
            continue

        ranked = sorted(
            cands,
            key=lambda p: (-completeness_score(p), len(p.relative_to(root).parts), str(p)),
        )
        signatures = {read_text_bytes_signature(p) for p in ranked}
        chosen = ranked[0]
        resolved[sid] = chosen
        diagnostics.append({
            "subject_id": sid,
            "status": "resolved" if len(ranked) == 1 else "deduplicated",
            "chosen": str(chosen),
            "n_candidates": len(ranked),
            "signature_conflict": len(signatures) > 1,
            "chosen_completeness_score": completeness_score(chosen),
            "all_candidates": " | ".join(str(p) for p in ranked),
        })
    return resolved, diagnostics


def assert_no_signature_conflicts(diag: Sequence[dict], corpus: str) -> None:
    bad = [d for d in diag if bool(d.get("signature_conflict"))]
    if bad:
        examples = [f"{d['subject_id']}: {d.get('all_candidates', '')}" for d in bad[:10]]
        raise RuntimeError(
            f"{corpus}: {len(bad)} frozen IDs have conflicting duplicate transcript folders. "
            "Do not silently choose one copy. Resolve these IDs or rerun with a manually cleaned root. "
            f"Examples={examples}"
        )
