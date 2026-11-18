#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cef_common import TOPICS, norm_id

RUBRICS: Dict[str, List[str]] = {


    "observable": [
        "情绪低落", "兴趣或愉悦下降", "无望感", "自责或低自我价值",
        "疲劳或低能量", "焦虑紧张", "社交退缩", "注意力或思维困难",
        "未来消极取向", "回答展开不足", "情绪与叙述不一致", "社会支持不足",
    ],
    "symptom": [
        "情绪低落或心情压抑", "对活动失去兴趣或乐趣", "对未来感到没有希望",
        "自我责备或觉得自己没有价值", "疲乏、没有精力", "睡眠出现问题",
        "焦虑、紧张或坐立不安", "回避社交、不想与人来往",
        "注意力难以集中", "行动或说话变慢", "对未来持消极看法",
        "过度关注自己的问题", "回避谈论困难的事", "回答简短、不愿展开",
        "叙述内容与情绪表达不一致", "缺少来自他人的支持",
    ],

    "sds": [
        "情绪低落、沮丧", "早晨心情最差", "容易哭泣或想哭", "夜里睡不好",
        "食欲下降", "对异性或亲密关系兴趣下降", "体重变化或身体不适",
        "便秘等消化不适", "心跳加快或心慌", "容易疲劳", "头脑不如以前清楚",
        "做事比以前费力", "坐立不安、静不下来", "对未来不抱希望",
        "比以前更容易烦躁", "做决定变得困难", "觉得自己没有用处",
        "生活缺乏充实感", "觉得别人没有自己会更好", "对以前喜欢的事失去兴趣",
    ],
}

SYSTEM = ("你是访谈文本的证据标注员，不是诊断者。你只根据给出的这一段回答本身，"
          "判断其中是否体现了指定的线索。不要推测、不要使用编号或量表信息。"
          "只回答一个字：是 或 否。")


def read_text(path: Path) -> str:
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc, errors="ignore").strip()
        except Exception:
            continue
    return ""


def build_folder_index(root: Path) -> Dict[str, Path]:
    exact: Dict[str, Path] = {}
    buckets: Dict[str, List[Path]] = {}
    for pf in root.rglob("positive.txt"):
        folder = pf.parent
        if not all((folder / f"{t}.txt").exists() for t in TOPICS):
            continue
        exact[folder.name.strip()] = folder
        buckets.setdefault(norm_id(folder.name), []).append(folder)
    out = dict(exact)
    for k, v in buckets.items():
        uniq = list(dict.fromkeys(v))
        if len(uniq) == 1:
            out[k] = uniq[0]
    return out


def read_label(folder: Path) -> float:
    for name in ("new_label.txt", "label.txt"):
        p = folder / name
        if p.exists():
            try:
                return float(read_text(p).split()[0])
            except Exception:
                continue
    return float("nan")


def stratified_probe(ids: List[str], index: Dict[str, Path],
                     k: int, threshold: float) -> List[str]:

    buckets: Dict[tuple, List[str]] = {}
    for sid in ids:
        folder = index.get(sid) or index.get(norm_id(sid))
        if folder is None:
            continue
        lab = read_label(folder)
        if not np.isfinite(lab):
            continue
        coh = "v" if str(sid).strip().lower().startswith("v") else "t"
        buckets.setdefault((int(lab >= threshold), coh), []).append(sid)
    picked: List[str] = []
    keys = sorted(buckets)
    i = 0
    while len(picked) < k and any(buckets[kk] for kk in keys):
        kk = keys[i % len(keys)]
        if buckets[kk]:
            picked.append(buckets[kk].pop(0))
        i += 1
    return picked or ids[:k]


def single_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"'{s}' 不是单 token（{ids}），换个字或改模板")
    return ids[0]


def make_prompt(tok, answer: str, item: str) -> str:
    user = (f"下面是一位受访者对一个开放式问题的回答。\n\n"
            f"【回答】{answer}\n\n"
            f"这段回答是否体现了「{item}」？只回答 是 或 否。")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


@torch.no_grad()
def log_odds(model, tok, answer: str, item: str,
             yes_id: int, no_id: int, max_len: int = 1536) -> float:
    text = make_prompt(tok, answer, item)


    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_len, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    logits = model(**enc, return_dict=True).logits[0, -1, :].float()
    lp = torch.log_softmax(logits, dim=-1)
    return float(lp[yes_id].item() - lp[no_id].item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ref_npz", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rubric", default="observable", choices=sorted(RUBRICS))
    ap.add_argument("--base_evidence", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--threshold", type=float, default=53.0)
    ap.add_argument("--selfcheck", type=int, default=0,
                    help=">0 时只跑这么多受试者，打印分布诊断后退出")
    args = ap.parse_args()

    from tqdm import tqdm

    items = RUBRICS[args.rubric]
    ref = np.load(args.ref_npz, allow_pickle=True)
    ids = [str(x).strip() for x in ref["positive_ids"]]
    index = build_folder_index(Path(args.data_root))
    if args.selfcheck > 0:


        ids = stratified_probe(ids, index, args.selfcheck, args.threshold)
        print(f"自检样本({len(ids)}): {ids}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype), device_map="auto").eval()
    yes_id, no_id = single_token(tok, "是"), single_token(tok, "否")

    profiles = np.zeros((len(ids), 3, len(items)), dtype=np.float32)
    for i, sid in enumerate(tqdm(ids, desc=f"log-odds [{args.rubric}]")):
        folder = index.get(sid) or index.get(norm_id(sid))
        if folder is None:
            raise RuntimeError(f"找不到受试者: {sid}")
        for ti, t in enumerate(TOPICS):
            ans = read_text(folder / f"{t}.txt") or "（无有效回答）"
            for ii, item in enumerate(items):
                profiles[i, ti, ii] = log_odds(model, tok, ans, item,
                                               yes_id, no_id, args.max_len)

    std = profiles.reshape(-1, profiles.shape[-1]).std(axis=0)
    print(f"\n每个条目的跨样本标准差（log-odds 尺度）:", flush=True)
    for name, v in zip(items, std):
        mark = "  <-- 太平" if v < 0.10 else ""
        print(f"  {name:22s} {v:7.4f}{mark}", flush=True)
    print(f"\n中位标准差 = {float(np.median(std)):.4f}", flush=True)

    if args.selfcheck > 0:
        print("\n前 3 个受试者在负性回答上的前 5 个条目分数:", flush=True)
        for i in range(min(3, len(ids))):
            vals = " ".join(f"{v:+.2f}" for v in profiles[i, 2, :5])
            print(f"  {ids[i]:10s} {vals}", flush=True)
        ok = float(np.median(std)) > 0.30
        print(f"""
自检结论：{'通过，可以跑全量' if ok else '未通过'}
中位标准差 > 0.30 才说明模型的判断确实随输入变化。
如果没通过，按这个顺序调：
  1) 确认 chat template 生效（打印 make_prompt 的输出看有没有 <|im_start|>）
  2) 换问法：把「是否体现了X」改成更具体的行为描述
  3) 换 --rubric（symptom / sds 两套问法的敏感度可能差很多）
  4) 加 1~2 个 few-shot 示例锚定
  5) 如果都不行，就承认 LLM 直接打分这条路在本数据上不work，
     改用 G_targeted_text_features.py 的词典/句法特征——那条路不依赖模型的主观判断。
""", flush=True)
        return

    payload: Dict[str, np.ndarray] = {
        "subject_ids": np.asarray(ids),
        "profiles": profiles,
        "feature_names": np.asarray(items),
    }
    if args.base_evidence:
        base = np.load(args.base_evidence, allow_pickle=True)
        if "summary_features" in base:
            key = "subject_ids" if "subject_ids" in base else "ids"
            bid = [str(x).strip() for x in base[key]]
            pos = {s: j for j, s in enumerate(bid)}
            order = [pos.get(s, pos.get(norm_id(s), -1)) for s in ids]
            if min(order) < 0:
                raise RuntimeError("base_evidence 的 ID 与 ref_npz 对不上")
            payload["summary_features"] = np.asarray(
                base["summary_features"], dtype=np.float32)[order]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(f"\nSaved {args.output}  profiles={profiles.shape}", flush=True)


if __name__ == "__main__":
    main()
