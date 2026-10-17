"""针对弱标签做语义扩展的提示词，能不能救长尾？（第 2 周补充诊断）

前面已经排除两条路：换措辞的多模板集成（长尾只 +0.031）与无监督特征后处理（全部变差）。
最后一个不引入标签的办法是**换说法本身**——给弱标签补上同义或更具体的描述，
比如 `eastern` 从 "This is a sound of eastern" 扩展成 "traditional Chinese music"、
"Indian classical music" 等。

做法：对 7 个最弱的标签手工写扩展提示词，其余 43 类沿用四模板；
用缓存好的 test 嵌入直接算，不重新提音频特征。

用法：
    python scripts/25_weak_tag_prompts.py
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
BASE_TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
]

# 只给最弱的几类加同义/更具体的说法；其余标签保持原样
WEAK_TAG_EXTRA = {
    "instrumental": ["instrumental music", "music with no singing", "a track performed only by instruments"],
    "modern": ["modern music", "contemporary music", "current-day popular music"],
    "eastern": ["traditional Chinese music", "Indian classical music", "Asian traditional music",
                "music with Eastern traditional instruments"],
    "baroque": ["baroque music", "Baroque era classical music", "17th century classical music"],
    "folk": ["folk music", "traditional folk songs", "acoustic folk music"],
    "upbeat": ["upbeat music", "cheerful and lively music", "happy fast tempo music"],
    "bass": ["deep bass", "low bass frequencies", "heavy bassline", "bass-heavy music"],
    "no vocals": ["music without vocals", "an instrumental track with no singing"],
}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def evaluate(scores: np.ndarray, labels: np.ndarray, tags: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "tag": tags,
        "AP": [average_precision_score(labels[:, i], scores[:, i]) for i in range(len(tags))],
        "AUC": [roc_auc_score(labels[:, i], scores[:, i]) for i in range(len(tags))],
        "positives": labels.sum(0).astype(int),
    })


def build_text(enc: ClapEncoder, tags: list[str]) -> np.ndarray:
    """逐个标签拼提示词：弱标签用基础模板 + 补充说法，其余只用基础模板。"""
    per_tag = []
    for tag in tags:
        prompts = [t.format(tag) for t in BASE_TEMPLATES]
        prompts += WEAK_TAG_EXTRA.get(tag, [])
        emb = enc.encode_text(prompts)
        per_tag.append(l2(emb.mean(0, keepdims=True))[0])
    return np.stack(per_tag)


def main() -> int:
    tags = load_tags()
    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    ids = set(pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
    test = anno[anno.clip_id.isin(ids)].reset_index(drop=True)
    labels = test[tags].astype(int).to_numpy()
    emb = np.load(RESULTS / "clap_official_emb_test5329_w3.npy")

    enc = ClapEncoder(windows=3)
    text_base = enc.tag_prompts(tags, templates=BASE_TEMPLATES)
    text_new = build_text(enc, tags)

    tail_tags = [t for i, t in enumerate(tags) if labels[:, i].sum() < 100]
    base = evaluate(emb @ text_base.T, labels, tags)
    new = evaluate(emb @ text_new.T, labels, tags)

    print(f"\n{'配置':22}{'全部AP':>9}{'长尾AP':>9}{'常见AP':>9}{'长尾AUC':>10}")
    for name, df in (("四模板（现状）", base), ("弱标签语义扩展", new)):
        tail, common = df[df.tag.isin(tail_tags)], df[~df.tag.isin(tail_tags)]
        print(f"{name:22}{df.AP.mean():>9.4f}{tail.AP.mean():>9.4f}{common.AP.mean():>9.4f}{tail.AUC.mean():>10.4f}")

    print("\n改动的 8 个标签：")
    b, n = base.set_index("tag"), new.set_index("tag")
    for tag in WEAK_TAG_EXTRA:
        if tag not in b.index:
            continue
        print(f"  {tag:14} AP {b.loc[tag, 'AP']:.3f} → {n.loc[tag, 'AP']:.3f}   "
              f"AUC {b.loc[tag, 'AUC']:.3f} → {n.loc[tag, 'AUC']:.3f}")

    pd.concat([base.assign(config="base"), new.assign(config="weak_prompts")]).to_csv(
        RESULTS / "clap_weak_tag_prompts.csv", index=False
    )
    print(f"\n明细 -> {RESULTS / 'clap_weak_tag_prompts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
