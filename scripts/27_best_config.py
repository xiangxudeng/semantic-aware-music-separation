"""长尾怎么解：把已验证有效的两招合起来，做出当前最好配置（第 2 周结论性实验）。

已验证有效的两条（都不需要重新训练模型）：
1. **弱标签语义扩展提示词**：`eastern` 从 "a sound of eastern" 扩展成
   "traditional Chinese music / Indian classical music /..."（长尾 AUC 0.808 → 0.843）
2. **few-shot 原型校准**：每个标签取 K 个训练集正样本算音频原型，
   `score = cos(audio, text) + alpha * cos(audio, prototype)`

两者都保持开放词汇：没有样本的新标签把 alpha 设 0、只用文本即可。

本脚本给出四种组合的对照，作为第 3 周的起点。

用法：
    python scripts/27_best_config.py
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
from src.semantic.prompts import BASE_TEMPLATES, prompts_for_tag

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
K_SHOTS = 10  # 折中：每个标签 10 个样本已经能拿到大部分收益，建设成本仍然很低
ALPHA = 1.0


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def metrics(scores: np.ndarray, labels: np.ndarray, tags: list[str]) -> dict:
    ap = np.array([average_precision_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    auc = np.array([roc_auc_score(labels[:, i], scores[:, i]) for i in range(len(tags))])
    tail = np.array([labels[:, i].sum() < 100 for i in range(len(tags))])
    return {
        "全部AP": ap.mean(), "长尾AP": ap[tail].mean(), "常见AP": ap[~tail].mean(),
        "全部AUC": auc.mean(), "长尾AUC": auc[tail].mean(), "常见AUC": auc[~tail].mean(),
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

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_ext = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    rng = np.random.default_rng(0)
    proto = np.zeros((len(tags), emb.shape[1]), dtype=np.float32)
    for i in range(len(tags)):
        pos = tr[y_tr[:, i] == 1]
        n = min(K_SHOTS, len(pos))
        if n:
            proto[i] = l2(emb[rng.choice(pos, size=n, replace=False)].mean(0, keepdims=True))[0]

    text_sim_base = emb[te] @ text_base.T  # 基础模板的文本相似度
    text_sim_ext = emb[te] @ text_ext.T  # 语义扩展后的文本相似度
    proto_sim = emb[te] @ proto.T  # few-shot 音频原型相似度

    combos = [
        ("① 基础模板（第 2 周初口径）", text_sim_base),
        ("② 基础模板 + 原型校准", text_sim_base + ALPHA * proto_sim),
        ("③ 弱标签语义扩展", text_sim_ext),
        ("④ 语义扩展 + 原型校准（推荐）", text_sim_ext + ALPHA * proto_sim),
    ]
    rows = []
    print(f"配置（原型：每个标签 {K_SHOTS} 个训练样本，alpha={ALPHA}）")
    print(f"{'':34}{'全部AP':>9}{'长尾AP':>9}{'常见AP':>9}{'长尾AUC':>10}")
    for name, score in combos:
        m = metrics(score, y_te, tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in m.items()}})
        print(f"{name:34}{m['全部AP']:>9.4f}{m['长尾AP']:>9.4f}{m['常见AP']:>9.4f}{m['长尾AUC']:>10.4f}")

    pd.DataFrame(rows).to_csv(RESULTS / "clap_best_config.csv", index=False)
    print(f"\n明细 -> {RESULTS / 'clap_best_config.csv'}")
    print("\n参照：零样本长尾 AP 0.111 / AUC 0.809；线性探针（全量监督）长尾 AP 0.193 / AUC 0.912")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
