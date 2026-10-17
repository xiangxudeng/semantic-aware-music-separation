"""长尾攻坚第三步（思路 2）：多原型聚类，替代"每标签一个平均向量"。

单原型的问题：`eastern` 这种标签本身就是大杂烩（中国、印度、中东完全是不同的音乐），
把所有正样本平均成一个向量，等于把几种截然不同的声音抹平。

改成每个标签聚 k 个原型、打分取最相似的那个，理论上能刻画标签内部的多模态结构。
这里对比 k = 1 / 3 / 5 / 8，并叠加"弱标签语义扩展 + LLM 子概念"的文本侧。

注意口径：原型法需要每个标签的少量样本，属于 few-shot；没有样本的新标签可以退化回纯文本。

用法：
    python scripts/42_longtail_multiprototype.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES, prompts_for_tag
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
MAX_SAMPLES = 50  # 每个标签最多用多少个训练正样本做原型
K_LIST = (1, 3, 5, 8)
SEED = 0


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def kmeans(X: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """极简 k-means（样本量小、维度低，不需要 sklearn 的完整实现）。"""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(len(C)):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return l2(C)


def main() -> int:
    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
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

    subs = {}
    cache = RESULTS / "llm_subconcepts.json"
    if cache.exists():
        subs = json.loads(cache.read_text(encoding="utf-8"))

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_hand = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)
    text_llm = []
    for tag in tags:
        base = enc.encode_text([t.format(tag) for t in BASE_TEMPLATES])
        sub = subs.get(tag, [])
        vecs = np.concatenate([base, enc.encode_text(sub)]) if sub else base
        text_llm.append(l2(vecs.mean(0, keepdims=True))[0])
    text_llm = np.stack(text_llm).astype(np.float32)
    text_mix = l2((text_hand + text_llm) / 2)

    rng = np.random.default_rng(SEED)
    print(f"长尾 {len(tail_tags)} 类｜每标签最多用 {MAX_SAMPLES} 个训练正样本\n")

    rows = []
    print(HEADER)
    print("-" * len(HEADER))

    # 参照：纯文本
    for name, t in (("① 纯文本：基础四模板", text_base),
                    ("② 纯文本：手工扩展", text_hand),
                    ("③ 纯文本：手工+LLM 混合", text_mix)):
        df = evaluate_scores(emb[te] @ t.T, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))

    # 多原型
    scores_by_k = {}
    for k in K_LIST:
        score = np.zeros((len(te), len(tags)), dtype=np.float32)
        for i in range(len(tags)):
            pos = tr[y_tr[:, i] == 1]
            if len(pos) == 0:
                score[:, i] = emb[te] @ text_mix[i]
                continue
            pick = rng.choice(pos, size=min(MAX_SAMPLES, len(pos)), replace=False)
            protos = kmeans(emb[pick], k, seed=SEED)
            score[:, i] = (emb[te] @ protos.T).max(1)
        scores_by_k[k] = score
        name = f"④ 多原型 k={k}"
        df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))

    # 原型 + 文本融合
    for k in (1, 3, 5):
        for alpha in (0.5, 1.0):
            score = emb[te] @ text_mix.T + alpha * scores_by_k[k]
            name = f"⑤ 文本混合 + k={k} 原型 (α={alpha})"
            df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
            s = summarize(df, tail_tags)
            rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
            print(format_row(name, s))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "longtail_multiprototype.csv", index=False)
    best = out.loc[out["长尾_recall@prev"].idxmax()]
    base_recall = rows[0]["长尾_recall@prev"]
    print(f"\n长尾召回率最高：{best['配置']} → {best['长尾_recall@prev']:.4f}"
          f"（相对零样本 {best['长尾_recall@prev'] / base_recall - 1:+.1%}）")
    print(f"明细 -> {RESULTS / 'longtail_multiprototype.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
