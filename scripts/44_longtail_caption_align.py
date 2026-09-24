"""长尾攻坚第五步：用**真实音乐描述**做对比学习对齐（思路 3 的正确做法）。

上一版为什么失败
----------------
我先用模板把标签拼成句子（"audio with the tags rock, guitar"）当训练目标，结果全线变差。
原因大概率是：这类生造句子对 CLAP 的文本编码器来说是**分布外**的输入，
它的文本向量根本不在音频向量附近，拿它当目标等于把适配器往错的方向拉。

这一版换成真人（或 LLM 基于标签写的）自然语言描述，来自 LP-MusicCaps-MTT：
3,300 条 MTT 片段 × 每条 4 句描述。实测这些片段与 MTT 官方划分完全对齐，
且**没有一条落在测试集里**，可以放心当训练数据。

目标
----
1. 看真实描述能不能把长尾指标拉起来（对比零样本与前面的方案）
2. 用"从描述里剔除留出标签的词"的方式做开放词汇验证

用法：
    python scripts/44_longtail_caption_align.py
"""

from __future__ import annotations

import glob
import re
import sys
import time
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
LP = ROOT / "data" / "lpmusiccaps"
RESULTS = ROOT / "results"
EPOCHS, BATCH, LR = 150, 256, 1e-3
N_HOLDOUT, SPLITS = 10, 3


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


def load_captions() -> pd.DataFrame:
    rows = []
    # 只取 LP 的 train 划分：它的 test-*.parquet 正好就是 MTT 的测试集片段，混进来会污染评测
    for f in sorted(glob.glob(str(LP / "train-*.parquet"))):
        d = pd.read_parquet(f, columns=["track_id", "texts"])
        for tid, texts in zip(d["track_id"], d["texts"]):
            for t in texts:
                rows.append({"clip_id": str(tid), "caption": str(t)})
    return pd.DataFrame(rows)


def train_adapter(X, S, epochs=EPOCHS):
    torch.manual_seed(0)
    model = Adapter()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            if len(idx) < 4:
                continue
            a = model(X[idx])
            logits = model.scale * a @ S[idx].T
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx)))
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
    row_of = {cid: i for i, cid in enumerate(anno["clip_id"])}

    test_ids = set(pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
    train_ids = set(pd.read_csv(DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))

    caps = load_captions()
    caps = caps[caps.clip_id.isin(row_of)]
    assert not (set(caps.clip_id) & test_ids), "描述数据里混进了测试集片段"
    print(f"描述数据 {len(caps)} 条（{caps.clip_id.nunique()} 首曲子），全部来自训练/验证划分")

    tr = np.array(sorted({row_of[c] for c in caps.clip_id}))
    te = np.array(sorted(row_of[c] for c in test_ids))
    y_te = labels[te]
    # 召回率的判定比例统一用 MTT 训练集的比例（口径唯一，方法之间才可比）
    mtt_train = np.flatnonzero(anno["clip_id"].isin(train_ids).to_numpy())
    prevalence = labels[mtt_train].mean(0)
    tail_tags = [t for i, t in enumerate(tags) if y_te[:, i].sum() < 100]
    print(f"训练音频 {len(tr)} 条｜评测 {len(te)} 条｜长尾 {len(tail_tags)} 类\n")

    enc = ClapEncoder(windows=1)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES).astype(np.float32)
    text_hand = np.stack(
        [l2(enc.encode_text(prompts_for_tag(t)).mean(0, keepdims=True))[0] for t in tags]
    ).astype(np.float32)

    # 描述向量（缓存，只算一次）
    cache = RESULTS / "lp_caption_emb.npy"
    uniq = sorted(set(caps.caption))
    if cache.exists():
        cap_vec = np.load(cache)
        assert len(cap_vec) == len(uniq), (len(cap_vec), len(uniq))
    else:
        t0 = time.time()
        cap_vec = enc.encode_text(uniq).astype(np.float32)
        np.save(cache, cap_vec)
        print(f"描述文本编码 {len(uniq)} 条，用时 {time.time() - t0:.0f}s")
    vec_of = dict(zip(uniq, cap_vec))

    def build(cap_frame):
        audio_idx = np.array([row_of[c] for c in cap_frame.clip_id])
        S = torch.from_numpy(np.stack([vec_of[c] for c in cap_frame.caption])).float()
        return torch.from_numpy(emb[audio_idx]), S

    X, S = build(caps)
    print(f"\n开始对比学习：{len(X)} 对（音频, 描述），{EPOCHS} 轮，batch {BATCH}", flush=True)
    t0 = time.time()
    model = train_adapter(X, S)
    print(f"训练完成，用时 {time.time() - t0:.0f} 秒", flush=True)
    with torch.no_grad():
        adapted = model(torch.from_numpy(emb)).numpy()

    rows = []
    print()
    print(HEADER)
    print("-" * len(HEADER))
    configs = {
        "① 零样本（基础四模板）": emb[te] @ text_base.T,
        "② 手工弱标签扩展": emb[te] @ text_hand.T,
        "③ 描述对比学习适配器": adapted[te] @ text_base.T,
        "④ 描述对比学习 + 手工扩展": adapted[te] @ text_hand.T,
    }
    for name, score in configs.items():
        df = evaluate_scores(score, y_te, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)
        rows.append({"配置": name, **{k: round(v, 4) for k, v in s.items()}})
        print(format_row(name, s))
        df.to_csv(RESULTS / f"longtail_caption_{len(rows)}.csv", index=False)

    # 开放词汇验证：从描述里剔除留出标签的词，再训一遍
    print("\n开放词汇验证（把留出标签的词从描述里全部剔除后再训练）")
    holdout_rows = []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        hold = rng.choice(len(tags), size=N_HOLDOUT, replace=False)
        words = [w for i in hold for w in re.split(r"[\s\-]+", tags[i].lower()) if len(w) > 2]
        keep_mask = ~caps.caption.str.lower().apply(
            lambda s: any(re.search(rf"\b{re.escape(w)}", s) for w in words)
        )
        sub = caps[keep_mask]
        if len(sub) < 500:
            print(f"  划分 {seed + 1}：剔除后只剩 {len(sub)} 条，跳过")
            continue
        Xs, Ss = build(sub)
        m = train_adapter(Xs, Ss, epochs=80)
        with torch.no_grad():
            ad = m(torch.from_numpy(emb)).numpy()
        zs = (emb[te] @ text_base.T)[:, hold]
        adc = (ad[te] @ text_base.T)[:, hold]
        names = [tags[i] for i in hold]
        d_zs = evaluate_scores(zs, y_te[:, hold], names, prevalence=prevalence[hold])
        d_ad = evaluate_scores(adc, y_te[:, hold], names, prevalence=prevalence[hold])
        holdout_rows.append({
            "划分": seed + 1, "可用描述": len(sub),
            "零样本AP": round(d_zs.AP.mean(), 4), "适配器AP": round(d_ad.AP.mean(), 4),
            "零样本AUC": round(d_zs.AUC.mean(), 4), "适配器AUC": round(d_ad.AUC.mean(), 4),
        })
        print(f"  划分 {seed + 1}：剔除后 {len(sub)} 条描述｜留出 10 类 "
              f"零样本 AP {d_zs.AP.mean():.4f} / AUC {d_zs.AUC.mean():.4f}  →  "
              f"适配器 AP {d_ad.AP.mean():.4f} / AUC {d_ad.AUC.mean():.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "longtail_caption_align.csv", index=False)
    if holdout_rows:
        hd = pd.DataFrame(holdout_rows)
        hd.to_csv(RESULTS / "longtail_caption_holdout.csv", index=False)
        print("\n留出标签平均：" + f"零样本 AP {hd['零样本AP'].mean():.4f} → 适配器 {hd['适配器AP'].mean():.4f}；"
              f"AUC {hd['零样本AUC'].mean():.4f} → {hd['适配器AUC'].mean():.4f}")
        ok = hd["适配器AP"].mean() > hd["零样本AP"].mean()
        print("开放词汇结论：" + ("✅ 有提升，学到的是可迁移的语言对齐" if ok else "❌ 没有提升"))
    print(f"\n明细 -> {RESULTS / 'longtail_caption_align.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
