"""长尾攻坚第四步（思路 3）：句子级对比学习，并验证它是不是真开放词汇。

为什么要有这一步
----------------
前面那个"对齐适配器"虽然把长尾 AP 打到 0.2005，但留出测试证明它是闭集：
它的训练信号是"这条音频属于 50 类里的哪几类"，学到的只是这 50 条的决策边界。

这里换掉训练信号：把每条音频的标签组合成**一句话**
（例如 `music that is rock, guitar and male vocal`），用 CLAP 文本编码器编成目标向量，
再让适配器输出与这句话做**对比学习**（batch 内互相当负样本）。

理论上这样学到的是"音频 → 语言"的通用对齐，遇到训练时没见过的词也应该有效——
因为模型学的是"把声音映射到语言空间"，而不是"分辨这 50 类"。

验证方式
--------
随机留出 10 个标签完全不参与训练（连它们的词都不出现在句子里），
适配器只用剩下 40 类训练，然后在这 10 类上评测。跑 3 组划分。

用法：
    python scripts/43_longtail_contrastive.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
EPOCHS, BATCH, LR = 25, 256, 1e-3
N_HOLDOUT, SPLITS = 10, 3
SENT_TEMPLATES = [
    "music that is {}",
    "a track described as {}",
    "audio with the tags {}",
]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def l2_torch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Adapter(torch.nn.Module):
    """与 28_alignment_adapter.py 保持同构，方便对比。"""

    def __init__(self, dim: int = 512, hidden: int = 512) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, dim))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_torch(x + self.net(x))


def build_sentences(labels_onehot: np.ndarray, tag_names: list[str]) -> list[str]:
    """把每条音频的正标签拼成一句话；用 3 个模板轮流，避免所有句子长得一样。"""
    out = []
    for i, row in enumerate(labels_onehot):
        tags = [tag_names[j] for j in np.flatnonzero(row)]
        if not tags:
            out.append("")
            continue
        joined = ", ".join(tags[:4]) if len(tags) > 1 else tags[0]
        out.append(SENT_TEMPLATES[i % len(SENT_TEMPLATES)].format(joined))
    return out


def train_contrastive(X, S, mask_codes):
    torch.manual_seed(0)
    model = Adapter()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(X)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            a = model(X[idx])                      # (B, 512)
            logits = model.scale * a @ S[idx].T    # (B, B) 批内互为负样本
            # 标签组合完全相同的样本，不该互相当负样本，掩掉
            same = mask_codes[idx][:, None] == mask_codes[idx][None, :]
            logits = logits.masked_fill(same, -1e4)
            target = torch.arange(len(idx))
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


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
    text_all = ClapEncoder(windows=3).tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)

    X = torch.from_numpy(emb[tr])
    print(f"train {len(tr)} / test {len(te)}｜长尾 {len(tail_tags)} 类\n")

    rows, details = [], {}
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        hold = rng.choice(len(tags), size=N_HOLDOUT, replace=False)
        keep = np.array([i for i in range(len(tags)) if i not in set(hold)])
        keep_names = [tags[i] for i in keep]

        # 只用语料里"可见标签"拼句子：留出标签的词一个都不出现
        y_tr_keep = y_tr[:, keep]
        sent_texts = build_sentences(y_tr_keep, keep_names)
        uniq = sorted(set(s for s in sent_texts if s))
        code = {s: i for i, s in enumerate(uniq)}
        mask_codes = torch.tensor([code.get(s, -1) for s in sent_texts])
        t0 = time.time()
        # 句子向量由 CLAP 文本编码器给出（缓存一次即可）
        sent_cache = RESULTS / f"sent_emb_hold{seed}.npy"
        if sent_cache.exists():
            sent_emb = np.load(sent_cache)
        else:
            sent_emb = ClapEncoder(windows=1).encode_text(uniq).astype(np.float32)
            np.save(sent_cache, sent_emb)
        S = torch.from_numpy(sent_emb[[code.get(s, 0) for s in sent_texts]])
        print(f"  划分 {seed + 1}：可见标签 {len(keep)} 类，句子 {len(uniq)} 种，"
              f"文本编码 {time.time() - t0:.0f}s", flush=True)

        model = train_contrastive(X, S, mask_codes)
        with torch.no_grad():
            adapted = model(torch.from_numpy(emb)).numpy()

        # 在留出标签上评测
        zs_hold = (emb[te] @ text_all.T)[:, hold]
        ad_hold = (adapted[te] @ text_all.T)[:, hold]
        hold_names = [tags[i] for i in hold]
        hold_labels = y_te[:, hold]
        hold_prev = prevalence[hold]
        d_zs = evaluate_scores(zs_hold, hold_labels, hold_names, prevalence=hold_prev)
        d_ad = evaluate_scores(ad_hold, hold_labels, hold_names, prevalence=hold_prev)
        rows.append({
            "划分": seed + 1,
            "留出标签": ", ".join(hold_names),
            "零样本AP": round(d_zs.AP.mean(), 4), "对比学习AP": round(d_ad.AP.mean(), 4),
            "零样本AUC": round(d_zs.AUC.mean(), 4), "对比学习AUC": round(d_ad.AUC.mean(), 4),
            "零样本召回": round(d_zs["recall@prevalence"].mean(), 4),
            "对比学习召回": round(d_ad["recall@prevalence"].mean(), 4),
        })
        print(f"    留出标签：零样本 AP {d_zs.AP.mean():.4f} / AUC {d_zs.AUC.mean():.4f} "
              f"/ 召回 {d_zs['recall@prevalence'].mean():.4f}  →  "
              f"对比学习 AP {d_ad.AP.mean():.4f} / AUC {d_ad.AUC.mean():.4f} "
              f"/ 召回 {d_ad['recall@prevalence'].mean():.4f}", flush=True)
        details[seed] = (d_zs, d_ad)

        # 顺便在全部 50 类上评测这个适配器（含长尾）
        d_all = evaluate_scores(adapted[te] @ text_all.T, y_te, tags, prevalence=prevalence)
        s = summarize(d_all, tail_tags)
        print("    全 50 类：长尾 AP {:.4f} / AUC {:.4f} / 召回 {:.4f}".format(
            s["长尾_AP"], s["长尾_AUC"], s["长尾_recall@prev"]), flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df[["划分", "零样本AP", "对比学习AP", "零样本AUC", "对比学习AUC",
                     "零样本召回", "对比学习召回"]].to_string(index=False))
    print(f"\n平均：零样本 AP {df['零样本AP'].mean():.4f} / AUC {df['零样本AUC'].mean():.4f} "
          f"/ 召回 {df['零样本召回'].mean():.4f}")
    print(f"      对比学习 AP {df['对比学习AP'].mean():.4f} / AUC {df['对比学习AUC'].mean():.4f} "
          f"/ 召回 {df['对比学习召回'].mean():.4f}")
    ok = df["对比学习AP"].mean() > df["零样本AP"].mean()
    print(f"\n留出标签上是否提升：{'是 ✅ —— 学到的对齐能迁移到没见过的标签' if ok else '否 ❌ —— 仍然是记住训练过的标签'}")
    df.to_csv(RESULTS / "longtail_contrastive_holdout.csv", index=False)
    print(f"明细 -> {RESULTS / 'longtail_contrastive_holdout.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
