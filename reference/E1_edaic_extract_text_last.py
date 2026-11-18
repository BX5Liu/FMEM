#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from edaic_ceef7_common import load_manifest, participant_only_text, read_text, split_text_three, temporal_payload

INSTRUCTION = (
    "Instruct: Represent a spoken clinical interview response for depression-screening "
    "evidence retrieval, preserving emotional state, self-evaluation, energy, sleep, "
    "social withdrawal, and future-oriented language.\nQuery: "
)


def append_eos(enc, eos_id):
    if eos_id is None:
        return enc
    ids = enc["input_ids"]
    if ids.shape[1] and int(ids[0, -1]) == int(eos_id):
        return enc
    eos = torch.tensor([[eos_id]], dtype=ids.dtype)
    enc["input_ids"] = torch.cat([ids, eos], dim=1)
    enc["attention_mask"] = torch.cat(
        [enc["attention_mask"], torch.ones((1, 1), dtype=enc["attention_mask"].dtype)], dim=1
    )
    return enc


@torch.no_grad()
def embed(model, tok, text: str, max_len: int) -> np.ndarray:
    q = INSTRUCTION + (text.strip() or "No valid participant response.")
    enc = tok(q, return_tensors="pt", truncation=True, max_length=max_len - 1,
              add_special_tokens=True)
    enc = append_eos(enc, tok.eos_token_id)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, return_dict=True)
    h = out.last_hidden_state.float()
    idx = int(enc["attention_mask"][0].sum().item()) - 1
    v = h[0, idx]
    v = torch.nn.functional.normalize(v, p=2, dim=-1)
    return v.cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args()

    df = load_manifest(args.manifest)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    all_x = []
    segment_chars = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="E-DAIC Qwen3-Embedding final-EOS"):
        txt = participant_only_text(read_text(r["text_path"]))
        segs = split_text_three(txt)
        vecs = [embed(model, tok, seg, args.max_len) for seg in segs]
        all_x.append(np.stack(vecs))
        segment_chars.append([len(s) for s in segs])

    X = np.asarray(all_x, dtype=np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = temporal_payload(df.subject_id, df.label, X)
    payload.update({
        "split": df.split.to_numpy(),
        "interviewer_mode": df.interviewer_mode.to_numpy(),
        "segment_chars": np.asarray(segment_chars, dtype=int),
        "model_path": np.asarray(str(args.model)),
        "instruction": np.asarray(INSTRUCTION),
        "pooling": np.asarray("final_eos_token"),
    })
    np.savez_compressed(out, **payload)
    print(f"Saved: {out}", flush=True)
    print(f"Shape={X.shape} labels={np.bincount(df.label, minlength=2).tolist()}", flush=True)
    print("Representation: instruction-enabled final [EOS] hidden state", flush=True)


if __name__ == "__main__":
    main()
