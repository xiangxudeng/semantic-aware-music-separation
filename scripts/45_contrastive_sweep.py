"""对比学习为什么失败：是方法不对，还是学习率/轮数没调好？

44 号实验里，用真实描述做对比学习反而让指标全线下降。可能的解释有两种：
  (a) 方法本身不对——CLAP 的音频-文本空间已经对齐得很好，再拿几千条数据去"重新对齐"是破坏；
  (b) 只是训练配置不对——lr 太大或轮数太多，过拟合了那 3000 首曲子。

这个脚本用小网格（学习率 × 轮数）快速区分这两种解释：
如果每个配置都下降 → (a)；如果存在一个配置能超过基线 → (b)。

用法：
    python scripts/45_contrastive_sweep.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from importlib import import_module

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

contrast = import_module("44_longtail_caption_align")
DATA, LP, RESULTS = contrast.DATA, contrast.LP, contrast.RESULTS


def main() -> int:
    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    labels = anno[tags].astype(int).to_numpy()
    emb = contrast.l2(np.load(RESULTS / "clap_official_emb_all25863_w3.npy").astype(np.float32))
    row_of = {c: i for i, c in enumerate(anno["clip_id"])}

    test_ids = set(pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
    caps = contrast.load_captions()
    caps = caps[caps.clip_id.isin(row_of)]
    te = np.array(sorted(row_of[c] for c in test_ids))
    y_te = labels[te]
    tr = np.array(sorted({row_of[c] for c in caps.clip_id}))
    # 召回率的判定比例统一用 MTT **训练集** 的比例，不能跟着各方法自己的子集变，
    # 否则不同方法之间没法比（口径必须唯一）
    mtt_train = np.flatnonzero(
        anno["clip_id"].isin(
            set(pd.read_csv(DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
        ).to_numpy()
    )
    prevalence = labels[mtt_train].mean(0)
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]

    enc = ClapEncoder(windows=1)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    uniq = sorted(set(caps.caption))
    cap_vec = np.load(RESULTS / "lp_caption_emb.npy")
    vec_of = dict(zip(uniq, cap_vec))
    audio_idx = np.array([row_of[c] for c in caps.clip_id])
    X = torch.from_numpy(emb[audio_idx])
    S = torch.from_numpy(np.stack([vec_of[c] for c in caps.caption])).float()

    base_score = emb[te] @ text_base.T
    df0 = evaluate_scores(base_score, y_te, tags, prevalence=prevalence)
    s0 = summarize(df0, tail_tags)
    print(HEADER)
    print("-" * len(HEADER))
    print(format_row("① 零样本（基线）", s0))

    rows = [{"配置": "① 零样本（基线）", **{k: round(v, 4) for k, v in s0.items()}}]
    for lr in (1e-4, 3e-4, 1e-3):
        for epochs in (5, 20, 60):
            contrast.LR = lr
            model = contrast.train_adapter(X, S, epochs=epochs)
            with torch.no_grad():
                adapted = model(torch.from_numpy(emb)).numpy()
            df = evaluate_scores(adapted[te] @ text_base.T, y_te, tags, prevalence=prevalence)
            s = summarize(df, tail_tags)
            name = f"② lr={lr:g} epochs={epochs}"
            rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
            print(format_row(name, s), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "contrastive_sweep.csv", index=False)
    best = out.loc[out["长尾_AP"].idxmax()]
    print(f"\n长尾 AP 最高：{best['配置']} → {best['长尾_AP']:.4f}（基线 {s0['长尾_AP']:.4f}）")
    print("结论：" + ("存在优于基线的配置 → 是配置问题" if best["长尾_AP"] > s0["长尾_AP"]
                    else "全部低于基线 → 方法本身与目标不匹配，不是超参问题"))
    print(f"明细 -> {RESULTS / 'contrastive_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
