#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from edaic_ceef7_common import load_manifest, participant_only_text, read_text, split_text_three, temporal_payload

ITEMS = [
    "depressed or persistently low mood",
    "reduced interest or pleasure",
    "hopelessness or negative expectations about the future",
    "self-blame, worthlessness, or strongly negative self-evaluation",
    "fatigue or low energy",
    "anxiety, tension, or restlessness",
    "social withdrawal or avoidance of interaction",
    "difficulty concentrating or organizing thoughts",
    "negative future orientation",
    "limited elaboration or low willingness to continue the response",
    "mismatch between expressed emotion and narrated content",
    "limited perceived social support",
]

SYSTEM = (
    "You label observable evidence in an interview response; you do not diagnose. "
    "Use only the supplied participant response. Do not infer unmentioned symptoms, "
    "demographics, questionnaire scores, or interviewer behavior. Reply with exactly Yes or No."
)


def one_token_variants(tok, variants: List[str]) -> List[int]:
    ids = []
    for s in variants:
        z = tok.encode(s, add_special_tokens=False)
        if len(z) == 1:
            ids.append(int(z[0]))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(f"No one-token variant found among: {variants}")
    return ids


def prompt(tok, answer: str, item: str) -> str:
    user = (
        "Participant response:\n"
        f"{answer}\n\n"
        f"Does this response contain observable evidence of {item}? "
        "Reply with exactly Yes or No."
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score(model, tok, answer: str, item: str, yes_ids: List[int], no_ids: List[int], max_len: int) -> float:
    s = prompt(tok, answer, item)
    enc = tok(s, return_tensors="pt", truncation=True, max_length=max_len,
              add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    logits = model(**enc, return_dict=True).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    ly = torch.logsumexp(lp[torch.as_tensor(yes_ids, device=lp.device)], dim=0)
    ln = torch.logsumexp(lp[torch.as_tensor(no_ids, device=lp.device)], dim=0)
    return float((ly - ln).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--selfcheck", type=int, default=0,
                    help="If >0, process only a deterministic stratified subset and save a probe file.")
    args = ap.parse_args()

    df = load_manifest(args.manifest)
    if args.selfcheck > 0:
        picked = []
        for _, g in df.groupby(["split", "label", "interviewer_mode"], sort=True):
            picked.extend(g.head(max(1, args.selfcheck // max(1, df.groupby(["split", "label", "interviewer_mode"]).ngroups))).index.tolist())
        if len(picked) < args.selfcheck:
            picked.extend([i for i in df.index if i not in set(picked)][:args.selfcheck-len(picked)])
        df = df.loc[picked[:args.selfcheck]].reset_index(drop=True)
        print(f"Self-check subjects: {df.subject_id.tolist()}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    yes_ids = one_token_variants(tok, ["Yes", " Yes", "yes", " yes", "YES"])
    no_ids = one_token_variants(tok, ["No", " No", "no", " no", "NO"])
    print(f"Yes token candidates={yes_ids}; No token candidates={no_ids}", flush=True)

    profiles = np.zeros((len(df), 3, len(ITEMS)), dtype=np.float32)
    for i, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc="E-DAIC observable LLM evidence")):
        segs = split_text_three(participant_only_text(read_text(r["text_path"])))
        for si, ans in enumerate(segs):
            for ji, item in enumerate(ITEMS):
                profiles[i, si, ji] = score(model, tok, ans or "No valid participant response.",
                                                    item, yes_ids, no_ids, args.max_len)

    flat_std = profiles.reshape(-1, profiles.shape[-1]).std(axis=0)
    print("Item standard deviations (log-odds):", flush=True)
    for item, sd in zip(ITEMS, flat_std):
        print(f"  {item:68s} {sd:8.4f}", flush=True)
    print(f"Median item std={np.median(flat_std):.4f}", flush=True)
    if float(np.median(flat_std)) < 0.10:
        print("[WARN] Evidence scores are nearly flat; inspect the chat template/token IDs before using them.", flush=True)

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = temporal_payload(df.subject_id, df.label, profiles, ITEMS)
    payload.update({
        "split": df.split.to_numpy(),
        "interviewer_mode": df.interviewer_mode.to_numpy(),
        "rubric": np.asarray("observable_12_english"),
        "model_path": np.asarray(str(args.model)),
        "readout": np.asarray("logsumexp_yes_variants_minus_logsumexp_no_variants"),
    })
    np.savez_compressed(out, **payload)
    print(f"Saved: {out} profiles={profiles.shape}", flush=True)


if __name__ == "__main__":
    main()
