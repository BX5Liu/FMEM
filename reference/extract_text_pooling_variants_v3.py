#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cef_common import TOPICS, norm_id

INSTRUCTION = (
    'Instruct: Represent a spoken interview answer for depression-screening '
    'evidence retrieval, preserving emotional state, self-evaluation, energy, '
    'social withdrawal, and future-oriented language.\nQuery: '
)


def read_text(path: Path) -> str:
    for enc in ('utf-8', 'gb18030', 'gbk'):
        try:
            return path.read_text(encoding=enc, errors='ignore').strip()
        except Exception:
            pass
    return ''


def folder_index(root: Path) -> Dict[str, Path]:
    exact: Dict[str, Path] = {}
    buckets: Dict[str, List[Path]] = {}
    for p in root.rglob('positive.txt'):
        folder = p.parent
        if not all((folder / f'{t}.txt').exists() for t in TOPICS):
            continue
        exact[folder.name.strip()] = folder
        buckets.setdefault(norm_id(folder.name), []).append(folder)
    out = dict(exact)
    for k, vals in buckets.items():
        vals = list(dict.fromkeys(vals))
        if len(vals) == 1:
            out[k] = vals[0]
    return out


def append_eos(enc: Dict[str, torch.Tensor], eos_id: int | None) -> Dict[str, torch.Tensor]:
    if eos_id is None:
        return enc
    ids = enc['input_ids']
    if int(ids[0, -1]) == int(eos_id):
        return enc
    eos = torch.tensor([[eos_id]], dtype=ids.dtype)
    enc['input_ids'] = torch.cat([ids, eos], dim=1)
    enc['attention_mask'] = torch.cat(
        [enc['attention_mask'], torch.ones((1, 1), dtype=enc['attention_mask'].dtype)], dim=1)
    return enc


@torch.no_grad()
def embed_all(model, tok, text: str, instruct: bool, max_len: int) -> Dict[str, np.ndarray]:
    text = text or '无有效回答'
    if instruct:
        text = INSTRUCTION + text
    enc = tok(text, return_tensors='pt', truncation=True,
              max_length=max_len - 1, add_special_tokens=True)
    enc = append_eos(enc, tok.eos_token_id)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError('Model did not return hidden_states')
    n_layers = len(hs)
    lo = max(1, int(n_layers * 0.50))
    hi = max(lo + 1, int(n_layers * 0.75) + 1)
    last = hs[-1].float()
    mid = torch.stack([hs[i].float() for i in range(lo, min(hi, n_layers))], dim=0).mean(0)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    last_idx = int(enc['attention_mask'][0].sum().item()) - 1

    def mean_pool(h):
        return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    vectors = {
        'last': last[:, last_idx, :],
        'mean_last': mean_pool(last),
        'mean_mid': mean_pool(mid),
        'last_mid': mid[:, last_idx, :],
    }
    return {
        k: torch.nn.functional.normalize(v, p=2, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        for k, v in vectors.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--ref_npz', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--instruct', action='store_true')
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--dtype', default='bfloat16')
    args = ap.parse_args()

    ref = np.load(args.ref_npz, allow_pickle=True)
    ids = [str(x).strip() for x in ref['positive_ids']]
    labels = np.asarray(ref['positive_labels']) if 'positive_labels' in ref else None
    idx = folder_index(Path(args.data_root))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype, device_map='auto'
    ).eval()

    pools = ('last', 'mean_last', 'mean_mid', 'last_mid')
    store = {p: {t: [] for t in TOPICS} for p in pools}
    for sid in tqdm(ids, desc='text pooling variants'):
        folder = idx.get(sid) or idx.get(norm_id(sid))
        if folder is None:
            raise FileNotFoundError(f'Cannot resolve subject folder: {sid}')
        for topic in TOPICS:
            vecs = embed_all(model, tok, read_text(folder / f'{topic}.txt'),
                             args.instruct, args.max_len)
            for p in pools:
                store[p][topic].append(vecs[p])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in pools:
        payload = {}
        for topic in TOPICS:
            payload[f'{topic}_ids'] = np.asarray(ids)
            payload[f'{topic}_features'] = np.stack(store[p][topic]).astype(np.float32)
        if labels is not None:
            payload['positive_labels'] = labels
        path = outdir / f'eatd_text_{p}.npz'
        np.savez_compressed(path, **payload)
        print('Saved', path, payload['positive_features'].shape)


if __name__ == '__main__':
    main()
