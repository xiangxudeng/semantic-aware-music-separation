"""验证适配器是不是真的"开放词汇"：留出标签不参与训练。

上一节的对齐适配器在固定 50 类上把长尾 AP 从 0.111 提到 0.2005，超过了闭集的线性探针。
但"映射到文本空间"这个架构只说明**理论上**支持新标签，必须实测：

做法：随机留出 10 个标签**完全不出现在训练里**（连它们的文本提示词都不用），
适配器只用剩下 40 类训练，然后在这 10 个没见过的标签上评测。
如果仍然比零样本有明显提升，说明学到的是"音频到文本空间的通用对齐"，而不是记住了这 50 类。

跑 3 组不同的随机划分，避免结论被某一次划分带偏。

用法：
    python scripts/29_adapter_holdout.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
EPOCHS, BATCH, LR = 30, 256, 1e-3
N_HOLDOUT = 10
SPLITS = 3


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def l2_torch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Adapter(torch.nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 512) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, dim))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_torch(x + self.net(x))


def train_adapter(X, Y, T, pos_weight):
    torch.manual_seed(0)
    model = Adapter()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(X)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            logits = model.scale * model(X[idx]) @ T.T
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, Y[idx], pos_weight=pos_weight
            )
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
    text_all = ClapEncoder(windows=3).tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)

    X = torch.from_numpy(emb[tr])
    Y_all = torch.from_numpy(y_tr.astype(np.float32))
    rows = []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        hold = rng.choice(len(tags), size=N_HOLDOUT, replace=False)
        keep = np.array([i for i in range(len(tags)) if i not in set(hold)])
        t0 = time.time()
        pos_weight = (len(tr) - y_tr[:, keep].sum(0)) / np.maximum(y_tr[:, keep].sum(0), 1)
        model = train_adapter(X, Y_all[:, keep], torch.from_numpy(text_all[keep]),
                              torch.from_numpy(pos_weight.astype(np.float32)))
        with torch.no_grad():
            adapted = model(torch.from_numpy(emb)).numpy()

        # 相似度矩阵先算全 50 类，再取留出标签对应的列
        zs = (emb[te] @ text_all.T)[:, hold]
        ad = (adapted[te] @ text_all.T)[:, hold]
        ap_z = np.array([average_precision_score(y_te[:, i], zs[:, j]) for j, i in enumerate(hold)])
        ap_a = np.array([average_precision_score(y_te[:, i], ad[:, j]) for j, i in enumerate(hold)])
        auc_z = np.array([roc_auc_score(y_te[:, i], zs[:, j]) for j, i in enumerate(hold)])
        auc_a = np.array([roc_auc_score(y_te[:, i], ad[:, j]) for j, i in enumerate(hold)])
        rows.append({
            "划分": seed + 1,
            "留出标签": ", ".join(tags[i] for i in hold),
            "零样本AP": round(ap_z.mean(), 4), "适配器AP": round(ap_a.mean(), 4),
            "零样本AUC": round(auc_z.mean(), 4), "适配器AUC": round(auc_a.mean(), 4),
            "训练秒": round(time.time() - t0, 1),
        })
        print(f"  划分 {seed + 1}：留出 {len(hold)} 类，"
              f"零样本 AP {ap_z.mean():.4f} / AUC {auc_z.mean():.4f}  →  "
              f"适配器 AP {ap_a.mean():.4f} / AUC {auc_a.mean():.4f}   "
              f"（{time.time() - t0:.0f} 秒）", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df[["划分", "零样本AP", "适配器AP", "零样本AUC", "适配器AUC", "训练秒"]].to_string(index=False))
    print(f"\n平均：零样本 AP {df['零样本AP'].mean():.4f} / AUC {df['零样本AUC'].mean():.4f}"
          f"  →  适配器 AP {df['适配器AP'].mean():.4f} / AUC {df['适配器AUC'].mean():.4f}")
    print("\n留出标签明细：")
    for r in rows:
        print(f"  划分 {r['划分']}：{r['留出标签']}")
    df.to_csv(RESULTS / "clap_adapter_holdout.csv", index=False)
    print(f"\n明细 -> {RESULTS / 'clap_adapter_holdout.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
