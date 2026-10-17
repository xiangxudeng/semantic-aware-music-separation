"""无监督的特征后处理能不能救长尾？（第 2 周补充诊断）

上一节的线性探针说明：长尾标签的信息**本来就在 CLAP 特征里**，零样本只是没用上。
但线性探针用了标签，属于有监督方案，不是项目的答案（项目要开放词汇）。

于是自然的下一个问题：**不引入任何标签**，只对特征做一个全局线性变换，能不能把长尾捞回来？

测试四种经典做法（全部只用 train 划分的特征，不用标签）：

1. `none`：只做 L2 归一化（现状）
2. `center`：减去全体均值
3. `abtt-k`：再去掉前 k 个主方向（All-But-The-Top，Mu & Viswanath 2018）
4. `whiten`：PCA 白化

仍然保持零样本口径：变换在 train 特征上拟合，test 上评测，全程不碰标签。

用法：
    python scripts/24_embedding_postprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_tags
from src.semantic.clap_encoder import ClapEncoder

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_transforms(train: np.ndarray) -> dict[str, np.ndarray]:
    """只根据 train 特征（不用标签）算出各种变换矩阵。"""
    mean = train.mean(0, keepdims=True)
    centered = train - mean
    # 协方差的主方向；512 维不算大，直接做 SVD
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = (s**2) / max(1, len(train) - 1)
    out = {"mean": mean}
    for k in (1, 2, 4):
        out[f"abtt{k}"] = vt[:k]
    out["whiten"] = vt.T @ np.diag(1.0 / np.sqrt(var + 1e-8)) @ vt
    return out


def apply_transform(name: str, x: np.ndarray, t: dict[str, np.ndarray]) -> np.ndarray:
    if name == "none":
        return l2(x)
    if name == "center":
        return l2(x - t["mean"])
    if name.startswith("abtt"):
        k = name[4:]
        return l2(x - t["mean"] - (x - t["mean"]) @ t[f"abtt{k}"].T @ t[f"abtt{k}"])
    if name == "whiten":
        return l2((x - t["mean"]) @ t["whiten"])
    raise KeyError(name)


def score_all(scores: np.ndarray, labels: np.ndarray, tags: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "tag": tags,
        "AP": [average_precision_score(labels[:, i], scores[:, i]) for i in range(len(tags))],
        "AUC": [
            roc_auc_score(labels[:, i], scores[:, i]) if 0 < labels[:, i].sum() < len(labels) else np.nan
            for i in range(len(tags))
        ],
        "positives": labels.sum(0).astype(int),
    })


def main() -> int:
    tags = load_tags()
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    labels = anno[tags].astype(int).to_numpy()
    emb = np.load(RESULTS / "clap_official_emb_all25863_w3.npy").astype(np.float64)

    train_idx = np.flatnonzero(
        anno["clip_id"].isin(
            set(pd.read_csv(DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
        ).to_numpy()
    )
    test_idx = np.flatnonzero(
        anno["clip_id"].isin(
            set(pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
        ).to_numpy()
    )
    print(f"train {len(train_idx)} 条用于拟合变换，test {len(test_idx)} 条用于评测，全程不使用标签", flush=True)

    y_test = labels[test_idx]
    tail_tags = [t for i, t in enumerate(tags) if y_test[:, i].sum() < 100]

    enc = ClapEncoder(windows=3)
    text_emb = enc.tag_prompts(tags, templates=TEMPLATES).astype(np.float64)

    transforms = build_transforms(emb[train_idx])
    rows = []
    details = {}
    for name in ("none", "center", "abtt1", "abtt2", "abtt4", "whiten"):
        x = apply_transform(name, emb[test_idx], transforms)
        df = score_all(x @ text_emb.T, y_test, tags)
        df["config"] = name
        details[name] = df
        tail = df[df.tag.isin(tail_tags)]
        rows.append({
            "配置": name,
            "全部AP": round(df.AP.mean(), 4),
            "长尾AP": round(tail.AP.mean(), 4),
            "常见AP": round(df[~df.tag.isin(tail_tags)].AP.mean(), 4),
            "全部AUC": round(df.AUC.mean(), 4),
            "长尾AUC": round(tail.AUC.mean(), 4),
            "常见AUC": round(df[~df.tag.isin(tail_tags)].AUC.mean(), 4),
        })
        print(f"  {name:8} 全部AP {rows[-1]['全部AP']:.4f}  长尾AP {rows[-1]['长尾AP']:.4f}  "
              f"长尾AUC {rows[-1]['长尾AUC']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    out.to_csv(RESULTS / "clap_postprocess.csv", index=False)
    for name, df in details.items():
        df.to_csv(RESULTS / f"clap_postprocess_ap_{name}.csv", index=False)
    print(f"\n明细 -> {RESULTS / 'clap_postprocess.csv'}")

    best = out.loc[out["长尾AP"].idxmax()]
    print(f"\n长尾 AP 最好的配置：{best['配置']}（{best['长尾AP']:.4f}）")
    if best["配置"] != "none":
        print("长尾 14 类逐条对比：")
        zs = details["none"].set_index("tag")
        bs = details[best["配置"]].set_index("tag")
        for t in sorted(tail_tags, key=lambda x: bs.loc[x, "AP"], reverse=True):
            print(f"  {t:14} AP {zs.loc[t, 'AP']:.3f} → {bs.loc[t, 'AP']:.3f}   "
                  f"AUC {zs.loc[t, 'AUC']:.3f} → {bs.loc[t, 'AUC']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
