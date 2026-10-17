"""长尾攻坚收尾：调好参数的对比学习 + 开放词汇验证，并给出最终方案对照。

前面几步的结论
--------------
1. 纯文本（LLM 自动子概念 + 手工扩展）：长尾召回 +15.1%，完全不需要训练数据
2. few-shot 原型校准（每标签 10 个样本）：长尾召回 +45%，需要少量样本
3. 闭集适配器：长尾召回 +88%，但对没见过的标签无效
4. 描述对比学习：**换对超参后有效**（lr=1e-4 × 60 轮，长尾召回 +27.9%），
   关键是它能不能迁移到没见过的标签——本脚本要回答的就是这个

做法
----
在 3 组随机划分上，把留出标签的名字从描述文本里全部剔除，只用剩下的描述训练，
再在留出标签上评测。如果仍有提升，说明学到的是可迁移的"音频—语言"对齐。

用法：
    python scripts/46_longtail_final.py
"""

from __future__ import annotations

import re
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from src.semantic.prompts import BASE_TEMPLATES
from src.semantic.tagging_metrics import HEADER, evaluate_scores, format_row, summarize

contrast = import_module("44_longtail_caption_align")
DATA, RESULTS = contrast.DATA, contrast.RESULTS

LR, EPOCHS = 1e-4, 60      # 45 号扫描出来的最优配置
N_HOLDOUT, SPLITS = 10, 3


def main() -> int:
    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    labels = anno[tags].astype(int).to_numpy()
    emb = contrast.l2(np.load(RESULTS / "clap_official_emb_all25863_w3.npy").astype(np.float32))
    row_of = {c: i for i, c in enumerate(anno["clip_id"])}

    test_ids = set(pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
    mtt_train = np.flatnonzero(anno["clip_id"].isin(
        set(pd.read_csv(DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
    ).to_numpy())
    prevalence = labels[mtt_train].mean(0)   # 召回率的判定比例：统一用 MTT 训练集口径
    te = np.array(sorted(row_of[c] for c in test_ids))
    y_te = labels[te]
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]

    enc = contrast.ClapEncoder(windows=1)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    from src.semantic.prompts import prompts_for_tag

    text_hand = np.stack(
        [contrast.l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    caps = contrast.load_captions()
    caps = caps[caps.clip_id.isin(row_of)]
    uniq = sorted(set(caps.caption))
    cap_vec = np.load(RESULTS / "lp_caption_emb.npy")
    vec_of = dict(zip(uniq, cap_vec))

    def build(frame):
        idx = np.array([row_of[c] for c in frame.clip_id])
        S = torch.from_numpy(np.stack([vec_of[c] for c in frame.caption])).float()
        return torch.from_numpy(emb[idx]), S

    # ---------- 1. 全量描述训练 ----------
    contrast.LR = LR
    X, S = build(caps)
    print(f"全量描述训练：{len(X)} 对，lr={LR:g}，{EPOCHS} 轮", flush=True)
    model = contrast.train_adapter(X, S, epochs=EPOCHS)
    with torch.no_grad():
        adapted = model(torch.from_numpy(emb)).numpy()

    rows = []
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for name, score in {
        "① 零样本（基础四模板）": emb[te] @ text_base.T,
        "② 手工弱标签扩展": emb[te] @ text_hand.T,
        "③ 描述对比学习（调参后）": adapted[te] @ text_base.T,
        "④ 描述对比学习 + 手工扩展": adapted[te] @ text_hand.T,
    }.items():
        df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))
        df.to_csv(RESULTS / f"longtail_final_{len(rows)}.csv", index=False)

    # ---------- 2. 开放词汇验证 ----------
    print("\n开放词汇验证：把留出标签的名字从描述里全部剔除后再训练")
    hold_rows = []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        hold = rng.choice(len(tags), size=N_HOLDOUT, replace=False)
        words = [w for i in hold for w in re.split(r"[\s\-]+", tags[i].lower()) if len(w) > 2]
        keep = ~caps.caption.str.lower().apply(
            lambda s: any(re.search(rf"\b{re.escape(w)}", s) for w in words))
        sub = caps[keep]
        Xs, Ss = build(sub)
        m = contrast.train_adapter(Xs, Ss, epochs=EPOCHS)
        with torch.no_grad():
            ad = m(torch.from_numpy(emb)).numpy()
        names = [tags[i] for i in hold]
        d_zs = evaluate_scores((emb[te] @ text_base.T)[:, hold], y_te[:, hold], names, prevalence=prevalence[hold])
        d_ad = evaluate_scores((ad[te] @ text_base.T)[:, hold], y_te[:, hold], names, prevalence=prevalence[hold])
        hold_rows.append({
            "划分": seed + 1, "可用描述": len(sub),
            "零样本AP": round(d_zs.AP.mean(), 4), "适配器AP": round(d_ad.AP.mean(), 4),
            "零样本AUC": round(d_zs.AUC.mean(), 4), "适配器AUC": round(d_ad.AUC.mean(), 4),
            "零样本召回": round(d_zs["recall@prevalence"].mean(), 4),
            "适配器召回": round(d_ad["recall@prevalence"].mean(), 4),
        })
        print(f"  划分 {seed + 1}：{len(sub)} 条描述 | 零样本 AP {d_zs.AP.mean():.4f} / AUC {d_zs.AUC.mean():.4f}"
              f"  →  适配器 AP {d_ad.AP.mean():.4f} / AUC {d_ad.AUC.mean():.4f}", flush=True)

    hd = pd.DataFrame(hold_rows)
    hd.to_csv(RESULTS / "longtail_final_holdout.csv", index=False)
    print("\n留出标签平均："
          f"零样本 AP {hd['零样本AP'].mean():.4f} / AUC {hd['零样本AUC'].mean():.4f}  →  "
          f"适配器 AP {hd['适配器AP'].mean():.4f} / AUC {hd['适配器AUC'].mean():.4f}")
    ok = hd["适配器AP"].mean() > hd["零样本AP"].mean()
    print("开放词汇结论：" + ("✅ 有提升 —— 对齐可迁移到没见过的标签" if ok else "❌ 没有提升 —— 仍然是闭集"))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "longtail_final.csv", index=False)
    print(f"\n明细 -> {RESULTS / 'longtail_final.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
