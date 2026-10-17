"""对齐适配器：长尾问题的"正解"能做到哪一步？（第 2 周结论性实验）

前面已经量出来的三档：

| 方案 | 长尾 AP | 长尾 AUC | 是否需要训练 |
|---|---|---|---|
| 零样本 | 0.111 | 0.809 | 否 |
| 弱标签语义扩展 + few-shot 原型 | 0.151 | 0.878 | 否（只用 10 样本/标签） |
| 线性探针（全量监督，闭集） | 0.193 | 0.912 | 是 |

原型校准是"检索式"的——每个标签只有一个平均向量，刻画不了标签内部的多样性
（比如 eastern 同时涵盖中国、印度、中东的音乐）。线性探针能学到多个判别方向，所以更强，
但它是**闭集**的（只输出那 50 类），不能直接当项目方案。

本脚本试中间的第三条路：**冻结 CLAP，只训练一个小的对齐适配器**，
把音频特征映射到文本空间。它保持开放词汇——新标签只要给出文本提示词就能打分，
不需要重新训练。目标：追平甚至超过线性探针。

用法：
    python scripts/28_alignment_adapter.py
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
from torch import nn

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES, prompts_for_tag

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
EPOCHS = 30
BATCH = 256
LR = 1e-3
SEED = 0


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def l2_torch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Adapter(nn.Module):
    """两层 MLP + 残差连接，把音频特征对齐到文本空间。"""

    def __init__(self, dim: int = 512, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_torch(x + self.net(x))


def metrics(scores: np.ndarray, labels: np.ndarray, tags: list[str]) -> dict:
    ap = np.array([average_precision_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    auc = np.array([roc_auc_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    tail = np.array([labels[:, i].sum() < 100 for i in range(len(tags))])
    return {
        "全部AP": ap.mean(), "长尾AP": ap[tail].mean(), "常见AP": ap[~tail].mean(),
        "全部AUC": auc.mean(), "长尾AUC": auc[tail].mean(), "常见AUC": auc[~tail].mean(),
    }


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_ext = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    X = torch.from_numpy(emb[tr])
    Y = torch.from_numpy(y_tr.astype(np.float32))
    T = torch.from_numpy(text_base)          # (50, 512)
    model = Adapter()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"开始训练对齐适配器：{len(tr)} 条训练样本，{EPOCHS} 轮，batch {BATCH}", flush=True)
    t0 = time.time()
    n = len(X)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        total = 0.0
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            xb, yb = X[idx], Y[idx]
            a = model(xb)                      # (B, 512)
            logits = model.scale * a @ T.T     # (B, 50)
            # 多标签：每个正标签都是一类正样本，用逐标签二分类的带 logit 调整损失
            pos_weight = (len(tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=torch.from_numpy(pos_weight.astype(np.float32))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  loss {total / n:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"训练完成，用时 {time.time() - t0:.0f} 秒", flush=True)

    model.eval()
    with torch.no_grad():
        adapted = model(torch.from_numpy(emb)).numpy()

    proto = np.zeros((len(tags), emb.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(len(tags)):
        pos = tr[y_tr[:, i] == 1]
        k = min(10, len(pos))
        if k:
            # pos 是全局下标（tr[mask]），所以直接索引 adapted 即可
            proto[i] = l2(adapted[rng.choice(pos, size=k, replace=False)].mean(0, keepdims=True))[0]

    combos = {
        "零样本（基础模板）": emb[te] @ text_base.T,
        "语义扩展 + 原型（第 2 周最好配置）": emb[te] @ text_ext.T + 1.0 * (emb[te] @ proto.T),
        "适配器（本次训练）": adapted[te] @ text_base.T,
        "适配器 + 语义扩展": adapted[te] @ text_ext.T,
    }
    rows = []
    print(f"\n{'配置':36}{'全部AP':>9}{'长尾AP':>9}{'常见AP':>9}{'长尾AUC':>10}")
    for name, score in combos.items():
        m = metrics(score, y_te, tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in m.items()}})
        print(f"{name:36}{m['全部AP']:>9.4f}{m['长尾AP']:>9.4f}{m['常见AP']:>9.4f}{m['长尾AUC']:>10.4f}")

    pd.DataFrame(rows).to_csv(RESULTS / "clap_adapter.csv", index=False)
    torch.save({"model": model.state_dict(), "args": {"epochs": EPOCHS, "lr": LR, "batch": BATCH}},
               RESULTS / "clap_adapter.th")
    print(f"\n明细 -> {RESULTS / 'clap_adapter.csv'}；权重 -> {RESULTS / 'clap_adapter.th'}")
    print("参照：线性探针（全量监督、闭集）长尾 AP 0.1929 / AUC 0.9120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
