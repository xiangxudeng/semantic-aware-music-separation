"""长尾怎么解：few-shot 原型校准能走到哪一步？（第 2 周补充诊断）

已知
----
- 零样本长尾 AP 0.112 / AUC 0.808
- 线性探针（全量监督）长尾 AP 0.193 / AUC 0.912
- 信息确实在特征里，缺的是"把信息取出来"的手段

这个脚本回答：**如果只给每个标签很少的样本（1/5/10/50 个），能走到哪一步？**

方法
----
对每个标签 t，取 K 个训练集正样本的音频嵌入求平均，得到"音频原型" p_t，然后

    score(a, t) = cos(a, text_t) + alpha * cos(a, p_t)

alpha=0 即退回零样本。这个做法：
- 只用每个标签的极少量样本，建设成本几乎为零（特征已缓存）
- **仍然支持开放词汇**：没有样本的新标签可以把 alpha 设 0，只靠文本

用法：
    python scripts/26_fewshot_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.semantic.clap_encoder import ClapEncoder

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
]
TRIALS = 5  # K 很小时随机抽样的波动大，重复几次取平均


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def evaluate(scores: np.ndarray, labels: np.ndarray, tags: list[str], tail_tags: list[str]) -> dict:
    ap = np.array([average_precision_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    auc = np.array([roc_auc_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    is_tail = np.array([t in tail_tags for t in tags])
    return {
        "全部AP": ap.mean(), "长尾AP": ap[is_tail].mean(), "常见AP": ap[~is_tail].mean(),
        "全部AUC": auc.mean(), "长尾AUC": auc[is_tail].mean(), "常见AUC": auc[~is_tail].mean(),
    }


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
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]
    print(f"train {len(tr)} / test {len(te)} ｜ 长尾 {len(tail_tags)} 类", flush=True)

    text = ClapEncoder(windows=3).tag_prompts(tags, templates=TEMPLATES).astype(np.float32)
    text_sim = emb[te] @ text.T          # (n_test, 50)

    rows = []
    rng = np.random.default_rng(0)
    for K in (0, 1, 5, 10, 50):
        for alpha in ((0.0,) if K == 0 else (0.25, 0.5, 1.0, 2.0)):
            trials = 1 if K == 0 else TRIALS
            acc = []
            for _ in range(trials):
                proto = np.zeros((len(tags), emb.shape[1]), dtype=np.float32)
                for i in range(len(tags)):
                    pos = tr[y_tr[:, i] == 1]
                    n_pick = min(K, len(pos))
                    if n_pick <= 0:  # K=0 或该标签在训练集里没有正样本 → 原型保持全零
                        continue
                    pick = rng.choice(pos, size=n_pick, replace=False)
                    proto[i] = l2(emb[pick].mean(0, keepdims=True))[0]
                proto[~np.isfinite(proto)] = 0.0
                audio_sim = emb[te] @ proto.T
                acc.append(evaluate(text_sim + alpha * audio_sim, y_te, tags, tail_tags))
            row = {"K": K, "alpha": alpha}
            for key in acc[0]:
                row[key] = float(np.mean([a[key] for a in acc]))
            rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n" + df.round(4).to_string(index=False))
    df.to_csv(RESULTS / "clap_fewshot_prototype.csv", index=False)

    best = df.loc[df["长尾AP"].idxmax()]
    print(f"\n长尾 AP 最好的配置：K={int(best['K'])}，alpha={best['alpha']}"
          f"  →  长尾 AP {best['长尾AP']:.4f} / AUC {best['长尾AUC']:.4f}，"
          f"全部 AP {best['全部AP']:.4f}")
    zs = df[(df.K == 0)].iloc[0]
    print(f"对照：零样本 长尾 AP {zs['长尾AP']:.4f} / AUC {zs['长尾AUC']:.4f}；"
          f"线性探针（全量监督）长尾 AP 0.1929 / AUC 0.9120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
