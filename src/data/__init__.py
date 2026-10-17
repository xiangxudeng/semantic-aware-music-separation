"""统一数据集加载接口。

把之前散在各个脚本里的数据读取逻辑收成一个模块，第 2 至 4 周直接调用即可。

用法（在项目根目录下运行）：
    from src.data import load_musdb, load_mtt, load_tags

    db   = load_musdb()              # MUSDB18-HQ 测试集，返回 musdb.DB
    df   = load_mtt(split="test")    # MTT 测试集，含 50 类标签与音频绝对路径
    tags = load_tags()               # 50 类标签清单

自检：
    python -m src.data

目录说明：第 2 周把原来的 `src/data.py` 改成了包 `src/data/`，
以便按计划把数据增强放在同一个包下（`src/data/augment.py`）。
对外接口没有变化。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# 数据集目录名 -> 是否为 musdb 的 wav 布局
MUSDB_SETS = {
    "musdb18hq_wav": True,   # MUSDB18-HQ 还原版，正式结果用这个
    "musdb18": False,        # 官方 .stem.mp4 版，跑代码逻辑时用这个更快
}


def load_tags() -> list[str]:
    """第 1 周冻结的 50 类标签清单。"""
    path = RESULTS / "mtt_50tags.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少标签清单 {path}，先跑 scripts/04_build_taglist.py")
    return pd.read_csv(path)["tag"].tolist()


def load_mtt(split: str = "test", limit: int | None = None) -> pd.DataFrame:
    """MTT 片段表，列为 clip_id / split / audio_path / mp3_path / 50 类标签。"""
    if split not in ("train", "val", "test"):
        raise ValueError("split 只能是 train / val / test")

    anno = DATA / "mtt" / "annotations_final.csv"
    split_file = DATA / "mtt" / f"{split}_gt_mtt.tsv"
    if not anno.exists() or not split_file.exists():
        raise FileNotFoundError(
            "MTT 未就绪，先跑 scripts/01_download_data.py mtt 与 05_fetch_mtt_audio.py"
        )

    tags = load_tags()
    df = pd.read_csv(anno, sep="\t")
    df["clip_id"] = df["clip_id"].astype(str)
    ids = set(pd.read_csv(split_file, sep="\t", header=None, usecols=[0])[0].astype(str))
    df = df[df["clip_id"].isin(ids)].reset_index(drop=True)
    df["split"] = split
    df["audio_path"] = df["mp3_path"].map(lambda p: DATA / "mtt" / p)
    if limit:
        df = df.head(limit)
    return df[["clip_id", "split", "audio_path", "mp3_path", *tags]]


def load_musdb(name: str = "musdb18hq_wav", subset: str = "test"):
    """返回 musdb.DB。name 可选 musdb18hq_wav（HQ 正式）或 musdb18（MP4 快速调试）。"""
    if name not in MUSDB_SETS:
        raise ValueError(f"name 只能是 {list(MUSDB_SETS)}")
    root = DATA / name
    if not root.exists():
        raise FileNotFoundError(f"没找到 {root}")

    import musdb

    return musdb.DB(root=str(root), is_wav=MUSDB_SETS[name], subsets=subset)


def load_musiccaps() -> pd.DataFrame | None:
    """MusicCaps 只有文字标注没有音频；未下载时返回 None。"""
    path = RESULTS / "musiccaps_index.csv"
    return pd.read_csv(path) if path.exists() else None


def summary() -> int:
    """自检：打印各数据集的可用状态。"""
    print(f"项目根目录 {ROOT}\n")

    try:
        tags = load_tags()
        print(f"标签清单：{len(tags)} 类 —— {', '.join(tags[:6])} …")
    except FileNotFoundError as exc:
        print(f"标签清单：不可用（{exc}）")

    for split in ("train", "val", "test"):
        try:
            df = load_mtt(split)
            hit = sum(p.exists() for p in df["audio_path"].head(200))
            print(f"MTT {split:5}：{len(df):6} 条，抽查前 200 条音频存在 {hit} 条")
        except FileNotFoundError:
            print(f"MTT {split:5}：未就绪")

    for name in MUSDB_SETS:
        try:
            counts = {s: len(load_musdb(name, subset=s).tracks) for s in ("train", "test")}
            print(f"{name:14}：train {counts['train']} 首 / test {counts['test']} 首")
        except FileNotFoundError:
            print(f"{name:14}：未就绪")

    caps = load_musiccaps()
    print(f"MusicCaps   ：{'未下载' if caps is None else str(len(caps)) + ' 条标注'}")
    return 0
