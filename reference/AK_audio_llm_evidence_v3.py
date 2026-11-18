#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cef_common import TOPICS, norm_id

SR = 16000
ITEMS_EN = [
    "slowed or effortful speech",
    "frequent or unusually long pauses",
    "low vocal energy or a soft, weak voice",
    "flat or monotonous intonation with little pitch variation",
    "a sad, dejected or downcast tone of voice",
    "a tense, anxious or strained tone of voice",
    "low engagement or a disinterested manner of speaking",
    "an unsteady, breathy or strained voice quality",
]
ITEMS_ZH = [
    "说话速度缓慢或发声吃力",
    "停顿频繁或停顿明显偏长",
    "声音能量低、音量弱",
    "语调平淡、音高起伏很小",
    "语气低落或沮丧",
    "语气紧张、焦虑或绷紧",
    "表达投入度低、显得缺乏兴趣",
    "嗓音发虚、不稳、气声或紧绷",
]
SYSTEM = (
    "You annotate vocal delivery. Judge only how the speech sounds, not the "
    "lexical meaning. Answer with exactly one word: Yes or No."
)
YES_WORDS = ("Yes", " Yes", "yes", " yes")
NO_WORDS = ("No", " No", "no", " no")
AUDIO_EXTS = (".wav", ".WAV", ".mp3", ".m4a", ".flac")


def load_ref(path: str, threshold: float) -> Tuple[List[str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    id_key = next((k for k in ("positive_ids", "subject_ids", "ids") if k in z.files), None)
    if id_key is None:
        raise KeyError(f"No ID key in {path}; keys={z.files}")
    ids = [str(x).strip() for x in np.asarray(z[id_key]).reshape(-1)]
    label_key = next((k for k in ("positive_labels", "labels", "y") if k in z.files), None)
    if label_key is None:
        y = np.full(len(ids), -1, dtype=int)
    else:
        raw = np.asarray(z[label_key], dtype=float).reshape(-1)
        uniq = set(np.unique(raw).tolist())
        y = raw.astype(int) if uniq.issubset({0.0, 1.0}) else (raw >= threshold).astype(int)
    if len(ids) != len(y) or len(set(map(norm_id, ids))) != len(ids):
        raise ValueError("Reference IDs/labels are inconsistent or duplicated")
    return ids, y


def find_audio(folder: Path, topic: str) -> Optional[Path]:


    names = [
        f"{topic}_out.wav", f"{topic}.wav", f"{topic}_out.WAV", f"{topic}.WAV",
        f"{topic}.mp3", f"{topic}.m4a", f"{topic}.flac",
    ]
    for name in names:
        p = folder / name
        try:
            if p.is_file() and p.stat().st_size > 44:
                return p
        except OSError:
            continue
    hits = []
    for p in sorted(folder.glob(f"{topic}*")):
        try:
            if p.suffix in AUDIO_EXTS and p.is_file() and p.stat().st_size > 44:
                hits.append(p)
        except OSError:
            continue
    return hits[0] if len(hits) == 1 else None


def folder_signature(folder: Path) -> str:
    h = hashlib.sha256()
    for topic in TOPICS:
        p = find_audio(folder, topic)
        h.update(topic.encode())
        if p is None:
            h.update(b"MISSING")
        else:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}".encode())
    return h.hexdigest()


def resolve_folders(root: str, ids: Sequence[str]) -> Tuple[Dict[str, Path], List[dict]]:
    rootp = Path(root)
    if not rootp.is_dir():
        raise FileNotFoundError(rootp)
    wanted = {norm_id(x) for x in ids}
    buckets: Dict[str, List[Path]] = {x: [] for x in wanted}
    for p in rootp.rglob("*"):
        if not p.is_dir():
            continue
        sid = norm_id(p.name)
        if sid in wanted and any(find_audio(p, t) is not None for t in TOPICS):
            buckets[sid].append(p)
    resolved: Dict[str, Path] = {}
    diag: List[dict] = []
    for sid in wanted:
        cands = list(dict.fromkeys(buckets.get(sid, [])))
        if not cands:
            diag.append({"subject_id": sid, "status": "missing", "candidates": ""})
            continue
        complete = [p for p in cands if all(find_audio(p, t) is not None for t in TOPICS)]
        ranked = sorted(complete or cands, key=lambda p: (len(p.relative_to(rootp).parts), str(p)))
        sigs = {folder_signature(p) for p in ranked}
        if len(ranked) > 1 and len(sigs) > 1:
            raise RuntimeError(
                f"Conflicting duplicate audio folders for {sid}: " + " | ".join(map(str, ranked[:10]))
            )
        resolved[sid] = ranked[0]
        diag.append({
            "subject_id": sid,
            "status": "resolved" if len(ranked) == 1 else "deduplicated_identical",
            "chosen": str(ranked[0]),
            "n_candidates": len(ranked),
        })
    return resolved, diag


def token_ids(tokenizer, words: Sequence[str]) -> List[int]:
    out: List[int] = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
    return sorted(set(out))


def stratified_probe(ids: Sequence[str], y: np.ndarray, k: int) -> List[int]:
    def batch_code(s: object) -> int:
        raw = str(s).strip().lower()
        if raw.startswith("t_") or raw.startswith("t-") or raw.startswith("t"):
            return 0
        if raw.startswith("v_") or raw.startswith("v-") or raw.startswith("v"):
            return 1
        return 0
    batches = np.asarray([batch_code(s) for s in ids])
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(ids)):
        lab = int(y[i]) if y[i] in (0, 1) else 0
        buckets.setdefault((lab, int(batches[i])), []).append(i)
    keys = sorted(buckets)
    out: List[int] = []
    cursor = 0
    while len(out) < min(k, len(ids)) and any(buckets[v] for v in keys):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            out.append(buckets[key].pop(0))
        cursor += 1
    return out or list(range(min(k, len(ids))))


def audio_windows(y: np.ndarray, max_sec: float, n_windows: int) -> List[np.ndarray]:
    max_samples = max(1, int(max_sec * SR))
    if len(y) <= max_samples or n_windows <= 1:
        return [y[:max_samples]]
    starts = np.linspace(0, len(y) - max_samples, n_windows).round().astype(int)
    return [y[s:s + max_samples] for s in starts]


class Qwen2AudioScorer:
    def __init__(self, model_path: str, dtype: str, max_sec: float, n_windows: int):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        td = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=td, device_map="auto", trust_remote_code=True
        ).eval()
        self.yes_ids = token_ids(self.tokenizer, YES_WORDS)
        self.no_ids = token_ids(self.tokenizer, NO_WORDS)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError(
                f"Could not find single-token Yes/No forms: yes={self.yes_ids}, no={self.no_ids}"
            )
        self.max_sec = float(max_sec)
        self.n_windows = int(n_windows)
        self._printed_shape = False
        print(f"yes_ids={self.yes_ids} no_ids={self.no_ids}", flush=True)

    def _build_prompt(self, pseudo_path: str, question: str) -> str:
        conversation = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": pseudo_path},
                {"type": "text", "text": question},
            ]},
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio_token = getattr(self.processor, "audio_token", "<|AUDIO|>")
        if audio_token not in prompt:


            bos = getattr(self.processor, "audio_bos_token", "<|audio_bos|>")
            eos = getattr(self.processor, "audio_eos_token", "<|audio_eos|>")
            prompt = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nAudio 1: {bos}{audio_token}{eos}\n"
                f"{question}<|im_end|>\n<|im_start|>assistant\n"
            )
        if prompt.count(audio_token) != 1:
            raise RuntimeError(
                f"Expected exactly one audio token in prompt, found {prompt.count(audio_token)}. "
                f"Prompt prefix={prompt[:300]!r}"
            )
        return prompt

    def _prepare_inputs(self, prompt: str, window: np.ndarray):


        attempts = []
        for key in ("audio", "audios"):
            kwargs = {
                "text": prompt,
                key: [window],
                "sampling_rate": SR,
                "return_tensors": "pt",
                "padding": True,
            }
            try:
                inputs = self.processor(**kwargs)
            except Exception as exc:
                attempts.append(f"{key}: {type(exc).__name__}: {exc}")
                continue
            keys = set(inputs.keys())
            if "input_features" in keys and "feature_attention_mask" in keys:
                return inputs, key
            attempts.append(f"{key}: returned keys={sorted(keys)}")
        raise RuntimeError(
            "Qwen2-Audio processor did not create audio tensors. "
            "The model would otherwise score text only. Attempts: " + " | ".join(attempts)
        )

    @torch.no_grad()
    def score_array(self, y: np.ndarray, pseudo_path: str, question: str) -> Tuple[float, int]:
        if len(y) < SR // 4:
            raise ValueError(f"Audio is too short: {pseudo_path} ({len(y) / SR:.3f}s)")
        values: List[float] = []
        for window in audio_windows(y, self.max_sec, self.n_windows):
            prompt = self._build_prompt(pseudo_path, question)
            inputs, audio_kw = self._prepare_inputs(prompt, window)
            if not self._printed_shape:
                shapes = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
                audio_token = getattr(self.processor, "audio_token", "<|AUDIO|>")
                print(
                    f"processor audio keyword={audio_kw} audio_token_count={prompt.count(audio_token)} "
                    f"tensor shapes={shapes}", flush=True
                )
                self._printed_shape = True
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            logits = self.model(**inputs, return_dict=True).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(lp[self.yes_ids], dim=0)
            no = torch.logsumexp(lp[self.no_ids], dim=0)
            values.append(float((yes - no).item()))
        return float(np.mean(values)), len(values)

    def score(self, wav_path: Path, question: str) -> Tuple[float, int]:
        import librosa
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        return self.score_array(y, str(wav_path), question)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ref_npz", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--n_windows", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=53.0)
    ap.add_argument("--selfcheck", type=int, default=0)
    ap.add_argument("--min_median_std", type=float, default=0.30)
    args = ap.parse_args()

    from tqdm import tqdm

    ids, y = load_ref(args.ref_npz, args.threshold)
    folders, diagnostics = resolve_folders(args.data_root, ids)
    missing = [d for d in diagnostics if d["status"] == "missing"]
    if missing:
        raise RuntimeError(f"Missing frozen subject folders: {[d['subject_id'] for d in missing[:20]]}")

    run_idx = stratified_probe(ids, y, args.selfcheck) if args.selfcheck > 0 else list(range(len(ids)))
    run_ids = [ids[i] for i in run_idx]
    items = ITEMS_EN if args.lang == "en" else ITEMS_ZH
    scorer = Qwen2AudioScorer(args.model, args.dtype, args.max_sec, args.n_windows)

    profiles = np.full((len(run_idx), 3, len(items)), np.nan, dtype=np.float32)
    windows_used = np.zeros((len(run_idx), 3), dtype=np.int16)
    paths = np.empty((len(run_idx), 3), dtype=object)
    errors: List[dict] = []
    for oi, src_i in enumerate(tqdm(run_idx, desc="Vocal-8")):
        sid = ids[src_i]
        folder = folders[norm_id(sid)]
        for ti, topic in enumerate(TOPICS):
            wav = find_audio(folder, topic)
            if wav is None:
                errors.append({"subject_id": sid, "topic": topic, "error": "audio_missing"})
                continue
            paths[oi, ti] = str(wav)
            for j, item in enumerate(items):
                question = (
                    f"Does the speaker's vocal delivery show {item}? Answer Yes or No."
                    if args.lang == "en" else
                    f"这段录音的发声方式是否表现出“{item}”？请只回答 Yes 或 No。"
                )
                try:
                    score, nw = scorer.score(wav, question)
                    profiles[oi, ti, j] = score
                    windows_used[oi, ti] = nw
                except Exception as exc:
                    errors.append({
                        "subject_id": sid, "topic": topic, "item": item, "error": repr(exc)
                    })
                    break
    if errors or np.any(~np.isfinite(profiles)):
        Path(args.output).with_suffix(".errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"Vocal-8 extraction is incomplete: errors={len(errors)}, "
            f"nonfinite={int(np.sum(~np.isfinite(profiles)))}"
        )

    flat = profiles.reshape(-1, profiles.shape[-1])
    item_std = flat.std(axis=0)
    median_std = float(np.median(item_std))
    unique_rows = int(len(np.unique(np.round(flat, 6), axis=0)))
    print("\n===== VOCAL-8 VARIATION CHECK =====", flush=True)
    for name, value in zip(items, item_std):
        print(f"{name:58s} std={value:.4f}", flush=True)
    print(f"median_item_std={median_std:.4f} unique_rows_1e-6={unique_rows}", flush=True)

    if args.selfcheck > 0:
        passed = median_std >= args.min_median_std and unique_rows > max(3, len(run_ids) // 2)
        print(f"自检结论：{'通过' if passed else '未通过'}", flush=True)
        print("Self-check mode does not save the requested output.", flush=True)
        if not passed:
            raise SystemExit(2)
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        subject_ids=np.asarray(run_ids),
        labels=y[np.asarray(run_idx)],
        profiles=profiles,
        feature_names=np.asarray(items),
        topics=np.asarray(TOPICS),
        audio_paths=paths,
        windows_used=windows_used,
        extraction=np.asarray(
            f"Qwen2-Audio next-token Yes/No log-odds; lang={args.lang}; "
            f"max_sec={args.max_sec}; n_windows={args.n_windows}; model={args.model}"
        ),
    )
    print(f"Saved: {out} profiles={profiles.shape}", flush=True)


if __name__ == "__main__":
    main()
