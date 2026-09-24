"""标签评测指标：AP / AUC / 召回率。

为什么单独做一个模块：大纲第 3 周的验收标准里写的是"**长尾标签召回率提升 ≥12%**"，
而召回率需要一个判定阈值，跟 AP / AUC 这种排序指标不是一回事。前面几周的实验全在算 AP / AUC，
这一块必须补齐，否则最后交不出那个数。

召回率的口径（两个都算，报告里都列）
------------------------------------
1. **recall@prevalence（主口径）**：对每个标签，把它分数最高的 p 比例当成"预测为正"，
   其中 p 取该标签在**训练集**里的正样本比例。这样预测数量与真实数量匹配，
   既不需要标定阈值，也不用测试集的真值（避免 oracle）。
2. **recall@PR（辅助）**：在 precision ≈ recall 的那个工作点上取召回率，
   用单个数字概括"精确率与召回率平衡时能捞回多少"。这是不平衡任务里的常用口径。

两个口径都给出，是为了避免"换个阈值数字就变"的扯皮。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_fraction(labels: np.ndarray, scores: np.ndarray, frac: float) -> float:
    """把分数最高的 frac 比例判为正，算召回率。"""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    k = max(1, int(round(frac * len(labels))))
    k = min(k, len(labels))
    top = np.argsort(-scores)[:k]
    hit = int(labels[top].sum())
    return hit / n_pos


def recall_at_pr_balance(labels: np.ndarray, scores: np.ndarray) -> float:
    """在 precision 与 recall 最接近的工作点上取召回率。"""
    if labels.sum() == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    i = int(np.argmin(np.abs(precision - recall)))
    return float(recall[i])


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    tags: list[str],
    prevalence: np.ndarray | None = None,
) -> pd.DataFrame:
    """逐标签算 AP / AUC / 两个召回率口径。

    prevalence：每个标签在**训练集**里的正样本比例，用于 recall@prevalence；
                不传时退化用评测集自身的比例（那就带 oracle 性质，仅作参考）。
    """
    rows = []
    for i, tag in enumerate(tags):
        y, s = labels[:, i], scores[:, i]
        frac = float(prevalence[i]) if prevalence is not None else float(y.mean())
        rows.append({
            "tag": tag,
            "AP": average_precision_score(y, s) if y.sum() else float("nan"),
            "AUC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan"),
            "recall@prevalence": recall_at_fraction(y, s, frac),
            "recall@PR": recall_at_pr_balance(y, s),
            "positives": int(y.sum()),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, tail_tags: list[str]) -> dict:
    """按 全部 / 长尾 / 常见 三组汇总。"""
    tail = df[df.tag.isin(tail_tags)]
    common = df[~df.tag.isin(tail_tags)]
    out = {}
    for name, sub in (("全部", df), ("长尾", tail), ("常见", common)):
        out[f"{name}_AP"] = float(sub.AP.mean())
        out[f"{name}_AUC"] = float(sub.AUC.mean())
        out[f"{name}_recall@prev"] = float(sub["recall@prevalence"].mean())
        out[f"{name}_recall@PR"] = float(sub["recall@PR"].mean())
    out["长尾类数"] = len(tail)
    return out


def format_row(name: str, s: dict) -> str:
    return (
        f"{name:34}{s['全部_AP']:>9.4f}{s['长尾_AP']:>9.4f}{s['长尾_AUC']:>10.4f}"
        f"{s['长尾_recall@prev']:>12.4f}{s['长尾_recall@PR']:>12.4f}"
    )


HEADER = (
    f"{'配置':34}{'全部AP':>9}{'长尾AP':>9}{'长尾AUC':>10}"
    f"{'长尾召回@比例':>12}{'长尾召回@PR':>12}"
)
