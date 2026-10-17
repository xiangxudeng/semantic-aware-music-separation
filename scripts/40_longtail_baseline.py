"""长尾攻坚第一步：把现有各配置统一到"AP / AUC / 召回率"三套口径上。

大纲第 3 周的验收标准是"长尾标签**召回率**提升 ≥12%"，而前几周一直在算 AP / AUC。
这个脚本先把已有配置的召回率补齐，确立基线，后面的改进才有得比。

用法：
    python scripts/40_longtail_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.semantic.clap_encoder import ClapEncoder
from src.semantic.prompts import BASE_TEMPLATES, prompts_for_tag
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
K_SHOTS = 10


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


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
    prevalence = y_tr.mean(0)  # 训练集里每个标签的正样本比例，召回率的判定依据
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]
    print(f"train {len(tr)} / test {len(te)}｜长尾 {len(tail_tags)} 类：{', '.join(tail_tags)}")
    print(f"长尾标签在训练集里的正样本比例：{np.mean([prevalence[tags.index(t)] for t in tail_tags]):.2%}\n")

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_ext = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    rng = np.random.default_rng(0)
    proto = np.zeros((len(tags), emb.shape[1]), dtype=np.float32)
    for i in range(len(tags)):
        pos = tr[y_tr[:, i] == 1]
        k = min(K_SHOTS, len(pos))
        if k:
            proto[i] = l2(emb[rng.choice(pos, size=k, replace=False)].mean(0, keepdims=True))[0]

    adapter = None
    ap_path = RESULTS / "clap_adapter.th"
    if ap_path.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from importlib import import_module

        Adapter = import_module("28_alignment_adapter").Adapter
        state = torch.load(ap_path, map_location="cpu", weights_only=False)
        adapter = Adapter()
        adapter.load_state_dict(state["model"])
        adapter.eval()

    configs = {
        "① 零样本（基础四模板）": emb[te] @ text_base.T,
        "② 手工弱标签语义扩展": emb[te] @ text_ext.T,
        "③ 语义扩展 + few-shot 原型": emb[te] @ text_ext.T + 1.0 * (emb[te] @ proto.T),
    }
    if adapter is not None:
        with torch.no_grad():
            adapted = adapter(torch.from_numpy(emb)).numpy()
        configs["④ 对齐适配器（闭集对照）"] = adapted[te] @ text_base.T

    rows = []
    print(HEADER)
    print("-" * len(HEADER))
    for name, score in configs.items():
        df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))
        df.to_csv(RESULTS / f"longtail_baseline_{len(rows)}.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "longtail_baseline.csv", index=False)

    base = rows[0]
    print(f"\n相对零样本的提升（大纲要求长尾召回率提升 ≥12%，相对提升则要 ×1.12）：")
    for r in rows[1:]:
        for key, label in (("长尾_recall@prev", "召回率@比例"), ("长尾_recall@PR", "召回率@PR"), ("长尾_AP", "长尾AP")):
            rel = r[key] / base[key] - 1
            flag = "达标" if label.startswith("召回率") and rel >= 0.12 else ("—" if not label.startswith("召回率") else "未达标")
            print(f"  {r['配置']:34}{label:12}{base[key]:.4f} → {r[key]:.4f}   相对 {rel:+.1%}  {flag}")
    print(f"\n明细 -> {RESULTS / 'longtail_baseline.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
