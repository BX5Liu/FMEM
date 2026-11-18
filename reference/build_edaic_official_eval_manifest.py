#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def norm_id(x: object) -> str:
    s = str(x).strip().replace("\\", "/")
    hits = re.findall(r"(?<!\d)(\d{3})(?!\d)", s)
    if hits:
        return str(int(hits[-1]))
    if re.fullmatch(r"\d+\.0", s):
        return str(int(float(s)))
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    raise ValueError(f"Cannot parse E-DAIC participant ID from {x!r}")


def find_column(df: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def interviewer_mode(sid: str) -> str:
    value = int(sid)
    if 300 <= value <= 492:
        return "woz"
    if 600 <= value <= 718:
        return "ai"
    return "unknown"


def read_split(label_dir: Path, split: str) -> pd.DataFrame:
    path = label_dir / f"{split}_split.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    id_col = find_column(df, ("Participant_ID", "participant_id", "subject_id"))
    binary_col = find_column(df, ("PHQ_Binary", "phq_binary"))
    score_col = find_column(df, ("PHQ_Score", "phq_score", "PHQ8_Score"))
    if id_col is None or binary_col is None or score_col is None:
        raise KeyError(
            f"{path} must contain Participant_ID, PHQ_Binary and PHQ_Score; "
            f"columns={list(df.columns)}"
        )
    out = pd.DataFrame({
        "subject_id": df[id_col].map(norm_id),
        "official_split": split,
        "label": pd.to_numeric(df[binary_col], errors="raise").astype(int),
        "phq_score": pd.to_numeric(df[score_col], errors="raise").astype(float),
        "label_source": str(path.resolve()),
    })
    if out.subject_id.duplicated().any():
        dup = out.loc[out.subject_id.duplicated(False), "subject_id"].tolist()
        raise RuntimeError(f"Duplicate IDs in {path}: {dup[:20]}")
    return out


def load_path_manifest(path: Path) -> pd.DataFrame:


    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    id_col = find_column(
        df,
        (
            "subject_id",
            "Participant_ID",
            "participant_id",
            "id",
        ),
    )


    text_col = find_column(
        df,
        (
            "text_path",
            "participant_text_path",
            "transcript_path",
            "processed_text_path",
            "transcript_file",
        ),
    )


    audio_col = find_column(
        df,
        (
            "audio_path",
            "participant_audio_path",
            "wav_path",
            "audio_file",
        ),
    )

    if id_col is None:
        raise KeyError(
            f"No participant ID column in {path}: "
            f"{list(df.columns)}"
        )

    def resolve_manifest_path(value: object) -> str:
        if pd.isna(value):
            return ""

        raw = str(value).strip()

        if not raw or raw.lower() in {
            "nan",
            "none",
            "null",
        }:
            return ""

        candidate = Path(raw).expanduser()


        if candidate.is_file():
            return str(candidate.resolve())


        relative_candidate = path.parent / candidate

        if relative_candidate.is_file():
            return str(relative_candidate.resolve())


        return ""

    out = pd.DataFrame({
        "subject_id": df[id_col].map(norm_id),
        "text_path": (
            df[text_col].map(resolve_manifest_path)
            if text_col is not None
            else ""
        ),
        "audio_path": (
            df[audio_col].map(resolve_manifest_path)
            if audio_col is not None
            else ""
        ),
    })

    if out["subject_id"].duplicated().any():
        duplicated = out.loc[
            out["subject_id"].duplicated(False),
            "subject_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate participant IDs in {path}: "
            f"{duplicated[:20]}"
        )

    print(
        "Path manifest columns: "
        f"id={id_col}, text={text_col}, audio={audio_col}"
    )
    print(
        "Paths resolved directly from manifest: "
        f"text={out['text_path'].ne('').sum()}/{len(out)}, "
        f"audio={out['audio_path'].ne('').sum()}/{len(out)}"
    )

    return out


def resolve_folder(root: Path, sid: str) -> Optional[Path]:
    direct = root / f"{sid}_P"
    if direct.is_dir():
        return direct
    matches = [p for p in root.glob("*_P") if norm_id(p.name) == sid]
    return matches[0] if len(matches) == 1 else None


def resolve_text(processed_root: Path, sid: str) -> Optional[str]:
    folder = resolve_folder(processed_root, sid)
    if folder is None:
        return None
    candidates = [
        folder / "text.txt",
        folder / f"{sid}_Transcript.txt",
        folder / f"{sid}_TRANSCRIPT.txt",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return str(path.resolve())
    return None


def resolve_audio(audio_root: Path, sid: str) -> Optional[str]:
    folder = resolve_folder(audio_root, sid)
    if folder is None:
        return None
    candidates = [
        folder / "audio.wav",
        folder / f"{sid}_AUDIO.wav",
        folder / "participant_audio.wav",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return str(path.resolve())
    wavs = sorted(folder.glob("*.wav"))
    if len(wavs) == 1:
        return str(wavs[0].resolve())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--path_manifest", required=True)
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    processed_root = Path(args.processed_root)
    audio_root = Path(args.audio_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = pd.concat(
        [read_split(label_dir, split) for split in ("train", "dev", "test")],
        ignore_index=True,
    )
    if labels.subject_id.duplicated().any():
        overlap = labels.loc[
            labels.subject_id.duplicated(False),
            ["subject_id", "official_split"],
        ].sort_values("subject_id")
        overlap.to_csv(outdir / "official_split_overlap_error.csv", index=False)
        raise RuntimeError("Official split IDs overlap; see overlap error CSV.")

    paths = load_path_manifest(Path(args.path_manifest))
    merged = labels.merge(paths, on="subject_id", how="left", validate="one_to_one")

    for index, row in merged.iterrows():
        sid = row.subject_id
        text_path = str(row.text_path) if pd.notna(row.text_path) else ""
        audio_path = str(row.audio_path) if pd.notna(row.audio_path) else ""
        if not text_path or not Path(text_path).is_file():
            resolved = resolve_text(processed_root, sid)
            if resolved:
                merged.at[index, "text_path"] = resolved
        if not audio_path or not Path(audio_path).is_file():
            resolved = resolve_audio(audio_root, sid)
            if resolved:
                merged.at[index, "audio_path"] = resolved

    missing = merged[
        ~merged.text_path.map(lambda x: Path(str(x)).is_file())
        | ~merged.audio_path.map(lambda x: Path(str(x)).is_file())
    ].copy()
    missing.to_csv(outdir / "official_manifest_unresolved_paths.csv", index=False)
    if not missing.empty:
        raise RuntimeError(
            f"{len(missing)} official subjects have unresolved text/audio paths; "
            f"see {outdir / 'official_manifest_unresolved_paths.csv'}"
        )

    merged["source_id"] = merged.subject_id
    merged["interviewer_mode"] = merged.subject_id.map(interviewer_mode)
    merged["split"] = merged.official_split
    merged["development_role"] = merged.official_split.map(
        {"train": "development_pool", "dev": "development_pool", "test": "locked_test"}
    )

    columns = [
        "subject_id", "source_id", "official_split", "split", "development_role",
        "label", "phq_score", "interviewer_mode", "text_path", "audio_path",
        "label_source",
    ]
    three_split = merged[columns].sort_values(
        ["official_split", "subject_id"],
        key=lambda s: s.map({"train": 0, "dev": 1, "test": 2}) if s.name == "official_split" else s,
    )
    three_path = outdir / "edaic_official_three_split_manifest.csv"
    three_split.to_csv(three_path, index=False)

    evaluation = three_split.copy()
    evaluation["split"] = evaluation.official_split.map(
        {"train": "train", "dev": "train", "test": "test"}
    )
    eval_path = outdir / "edaic_official_train_dev_test_manifest.csv"
    evaluation.to_csv(eval_path, index=False)

    test_only = evaluation[evaluation.official_split.eq("test")].copy()
    test_path = outdir / "edaic_official_test_only_manifest.csv"
    test_only.to_csv(test_path, index=False)

    summary = (
        three_split.groupby("official_split")
        .agg(
            n=("subject_id", "size"),
            positives=("label", "sum"),
            min_id=("subject_id", lambda x: min(map(int, x))),
            max_id=("subject_id", lambda x: max(map(int, x))),
        )
    )
    expected = {
        "train": (163, 37),
        "dev": (56, 12),
        "test": (56, 17),
    }
    for split, (n_expected, pos_expected) in expected.items():
        if split not in summary.index:
            raise RuntimeError(f"Missing official split: {split}")
        n_actual = int(summary.loc[split, "n"])
        pos_actual = int(summary.loc[split, "positives"])
        if (n_actual, pos_actual) != (n_expected, pos_expected):
            raise RuntimeError(
                f"Unexpected {split} distribution: n={n_actual}, positives={pos_actual}; "
                f"expected n={n_expected}, positives={pos_expected}"
            )

    print("===== E-DAIC OFFICIAL MANIFEST =====")
    print(summary.to_string())
    print(f"Development pool: {(evaluation.split == 'train').sum()} subjects")
    print(f"Locked test:      {(evaluation.split == 'test').sum()} subjects")
    print(f"Saved: {three_path}")
    print(f"Saved: {eval_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
