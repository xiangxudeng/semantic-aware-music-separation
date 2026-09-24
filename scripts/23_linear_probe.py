"""线性探针：长尾标签到底能不能靠监督信号救回来？（第 2 周补充诊断）

背景
----
导师关心的是"CLAP 长尾效果差"。第 2 周做完了归因（不是样本量问题），但长尾本身没解决：
零样本下长尾 14 类 AP 只有 0.112，其中 5 类连排序能力都接近随机。而"调提示词"这条路
对长尾几乎没有用（常见 36 类 +0.115，长尾 14 类只 +0.031）。

所以问题变成：**长尾是"CLAP 特征里就没有这些信息"，还是"零样本没把信息用好"？**

做法
----
用 MTT 官方 train 划分（18,706 条）训练一个多标签逻辑回归，在 test 划分上评测。
特征直接用已经算好的官方 CLAP 嵌入（25863 条，三个窗口平均），不用重新提取。

这不是项目的最终方案（项目要的是开放词汇），它的作用是给出**监督信号的上界参考**：
- 如果长尾 AP 大幅上升 → 说明特征里信息是有的，第 3 周的 LoRA 指令微调有理由做好
- 如果长尾 AP 仍然很低 → 说明问题在数据/标签本身，要调整预期

用法：
    python scripts/23_linear_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from src.data import load_tags

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"


def split_indices(anno: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for split in ("train", "val", "test"):
        ids = pd.read_csv(DATA / f"{split}_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str)
        out[split] = np.flatnonzero(anno["clip_id"].isin(set(ids)).to_numpy())
    return out


def evaluate(scores: np.ndarray, labels: np.ndarray, tags: list[str]) -> pd.DataFrame:
    rows = []
    for i, tag in enumerate(tags):
        y = labels[:, i]
        rows.append({
            "tag": tag,
            "AP": average_precision_score(y, scores[:, i]),
            "AUC": roc_auc_score(y, scores[:, i]) if 0 < y.sum() < len(y) else float("nan"),
            "positives": int(y.sum()),
        })
    return pd.DataFrame(rows).sort_values("AP", ascending=False)


def report(name: str, df: pd.DataFrame, tail_tags: list[str]) -> None:
    tail = df[df.tag.isin(tail_tags)]
    common = df[~df.tag.isin(tail_tags)]
    print(f"\n{name}")
    print(f"  全部 50 类   AP {df.AP.mean():.4f}   AUC {df.AUC.mean():.4f}")
    print(f"  长尾 {len(tail):2d} 类   AP {tail.AP.mean():.4f}   AUC {tail.AUC.mean():.4f}")
    print(f"  常见 {len(common):2d} 类   AP {common.AP.mean():.4f}   AUC {common.AUC.mean():.4f}")


def main() -> int:
    tags = load_tags()
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    emb = np.load(RESULTS / "clap_official_emb_all25863_w3.npy")
    assert len(emb) == len(anno), (emb.shape, len(anno))
    labels = anno[tags].astype(int).to_numpy()
    idx = split_indices(anno)
    print(f"全部 {len(anno)} 条 ｜ train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])}")

    # 长尾标签沿用第 2 周报告的口径：test 划分上正样本 < 100 的 14 类
    y_test = labels[idx["test"]]
    tail_tags = [t for i, t in enumerate(tags) if y_test[:, i].sum() < 100]
    print(f"长尾标签 {len(tail_tags)} 类：{', '.join(tail_tags)}")

    # 参照：零样本（直接用音频嵌入与文本嵌入的相似度）——这里用已保存的 AP 明细
    zs_path = RESULTS / "clap_zeroshot_ap_official_test5329_w3_t4.csv"
    if zs_path.exists():
        report("【参照】零样本（官方实现，三窗四模板）", pd.read_csv(zs_path), tail_tags)

    print("\n训练线性探针（MTT train 18,706 条，50 个二分类）...", flush=True)
    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="liblinear"),
        n_jobs=-1,
    )
    clf.fit(emb[idx["train"]], labels[idx["train"]])

    for split_name in ("val", "test"):
        s = clf.decision_function(emb[idx[split_name]])
        df = evaluate(s, labels[idx[split_name]], tags)
        report(f"【线性探针】在 {split_name} 划分上", df, tail_tags)
        if split_name == "test":
            df.to_csv(RESULTS / "clap_linear_probe_test.csv", index=False)
            print("\n  长尾 14 类逐条（线性探针 vs 零样本）：")
            zs = pd.read_csv(zs_path).set_index("tag") if zs_path.exists() else None
            sub = df[df.tag.isin(tail_tags)].sort_values("AUC")
            for r in sub.itertuples():
                zs_ap = f"{zs.loc[r.tag, 'AP']:.3f}" if zs is not None else "  -  "
                zs_auc = f"{zs.loc[r.tag, 'AUC']:.3f}" if zs is not None else "  -  "
                print(f"    {r.tag:14} 零样本 AP {zs_ap} / AUC {zs_auc}   →   线性探针 AP {r.AP:.3f} / AUC {r.AUC:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
