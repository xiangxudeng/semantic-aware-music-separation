"""长尾攻坚第二步（思路 1）：用本地 Qwen2 自动生成标签的子概念。

手工给弱标签补同义说法已经证明有效（长尾 AUC 0.809 → 0.844），但手工写的只能覆盖我想到的那几个。
这里改成**自动生成**：对任意一个标签，让本地 Qwen2-1.5B 生成 5 条"音乐专家会怎么描述它"的短语，
再用 CLAP 文本编码器编码、与基础模板融合。

自动化的意义在于：用户拿来一个**没见过的标签**，系统也能现场展开，这才叫开放词汇。

生成结果会缓存到 results/llm_subconcepts.json，重复运行不再花时间。

用法：
    python scripts/41_longtail_subconcepts.py            # 缺缓存时自动生成
    python scripts/41_longtail_subconcepts.py --regenerate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES, WEAK_TAG_EXTRA, prompts_for_tag
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
CACHE = RESULTS / "llm_subconcepts.json"
LLM = "Qwen/Qwen2-1.5B-Instruct"
N_PER_TAG = 5

PROMPT = """You are helping build a music tagging system.
The tag is: "{tag}"

Write {n} short English phrases that a music expert would use to describe music labeled "{tag}".
Rules:
- each phrase is 3 to 10 words
- be concrete: mention genre, instruments, mood, era or region when relevant
- one phrase per line, no numbering, no quotes, no explanations
"""


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def generate(tags: list[str]) -> dict[str, list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载 {LLM}（CPU float32，首次约需 1 分钟）...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float32)
    model.eval()

    out: dict[str, list[str]] = {}
    t0 = time.time()
    for i, tag in enumerate(tags, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(tag=tag, n=N_PER_TAG)}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=110, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        lines = []
        for line in raw.splitlines():
            s = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"').strip()
            if 2 <= len(s.split()) <= 14 and s:
                lines.append(s)
        out[tag] = lines[:N_PER_TAG]
        print(f"  [{i:2d}/{len(tags)}] {tag:14} → {len(out[tag])} 条  "
              f"（{time.time() - t0:.0f}s） {out[tag][:2]}", flush=True)
    CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已缓存到 {CACHE}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
    if CACHE.exists() and not args.regenerate:
        subs = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"复用缓存 {CACHE}（{len(subs)} 个标签）")
    else:
        subs = generate(tags)

    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    labels = anno[tags].astype(int).to_numpy()
    emb = l2(np.load(RESULTS / "clap_official_emb_all25863_w3.npy").astype(np.float32))

    def idx_of(split: str) -> np.ndarray:
        ids = set(pd.read_csv(DATA / f"{split}_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
        return np.flatnonzero(anno["clip_id"].isin(ids).to_numpy())

    tr, te = idx_of("train"), idx_of("test")
    y_tr, y_te = labels[tr], labels[te]
    prevalence = y_tr.mean(0)
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_hand = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    # LLM 子概念：基础模板 + LLM 生成的短语，两种融合方式
    llm_mean, llm_max = [], []
    for tag in tags:
        base = enc.encode_text([t.format(tag) for t in BASE_TEMPLATES])
        sub = subs.get(tag, [])
        if sub:
            se = enc.encode_text(sub)
            llm_mean.append(l2(np.concatenate([base, se]).mean(0, keepdims=True))[0])
            # max 融合：对每条短语单独算相似度后取最大，等价于"命中任一子概念就算像"
            llm_max.append(se)
        else:
            llm_mean.append(l2(base.mean(0, keepdims=True))[0])
            llm_max.append(base)
    text_llm_mean = np.stack(llm_mean).astype(np.float32)

    # 逐标签的 max 融合需要按标签分组算，这里手工拼
    score_llm_max = np.zeros((len(te), len(tags)), dtype=np.float32)
    for i, tag in enumerate(tags):
        se = llm_max[i]
        score_llm_max[:, i] = (emb[te] @ se.T).max(1)

    configs = {
        "① 零样本（基础四模板）": emb[te] @ text_base.T,
        "② 手工弱标签扩展": emb[te] @ text_hand.T,
        "③ LLM 子概念（平均融合）": emb[te] @ text_llm_mean.T,
        "④ LLM 子概念（最大融合）": score_llm_max,
        "⑤ LLM 子概念 + 手工扩展（平均）": (emb[te] @ text_llm_mean.T + emb[te] @ text_hand.T) / 2,
    }

    rows = []
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for name, score in configs.items():
        df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))
        df.to_csv(RESULTS / f"longtail_subconcept_{len(rows)}.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "longtail_subconcepts.csv", index=False)
    base = rows[0]
    print("\n相对零样本：")
    for r in rows[1:]:
        rel = r["长尾_recall@prev"] / base["长尾_recall@prev"] - 1
        print(f"  {r['配置']:34}长尾AP {base['长尾_AP']:.4f}→{r['长尾_AP']:.4f}   "
              f"长尾召回 {base['长尾_recall@prev']:.4f}→{r['长尾_recall@prev']:.4f}（相对 {rel:+.1%}）")
    print(f"\n明细 -> {RESULTS / 'longtail_subconcepts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
