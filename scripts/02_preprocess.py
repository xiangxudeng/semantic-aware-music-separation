"""第 1 周数据预处理：统一索引 + 冒烟测试片段。

做三件事（缺哪个数据集就跳过哪个，不报错）：
  1) MUSDB18 → results/musdb18_index.csv，并抽取 3 条 30 秒片段到 data/clips/ 用于 CPU 冒烟测试
  2) MTT     → results/mtt_index.csv（clip_id / split / mp3_path / 标签名）
              results/mtt_tag_counts.csv（标签频次，用来确定"50 类标签清单"）
  3) MusicCaps → results/musiccaps_index.csv（纯文字标注）

用法：python scripts/02_preprocess.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CLIP_SECONDS = 30
N_CLIPS = 3


def musdb18() -> None:
    src = DATA / "musdb18"
    if not src.exists():
        print("[MUSDB18] 未下载，跳过")
        return

    rows = [
        {"track": p.stem.replace(".stem", ""), "subset": p.parent.name,
         "path": str(p), "size_mb": round(p.stat().st_size / 2**20, 1)}
        for p in sorted(src.glob("*/*.stem.mp4"))
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "musdb18_index.csv", index=False)
    counts = df["subset"].value_counts().to_dict()
    print(f"[MUSDB18] {len(df)} 首 {counts} -> musdb18_index.csv")
    if counts.get("train", 0) != 100 or counts.get("test", 0) != 50:
        print("  提示：还不是完整 150 首（训练集可能没下），索引按现有文件生成，不影响第 1 周基线")

    # 冒烟测试片段：CPU 上先用 30 秒，别拿整首歌试
    import musdb

    clips_dir = DATA / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    db = musdb.DB(root=str(src), subsets="test", is_wav=False)
    for track in db.tracks[:N_CLIPS]:
        out = clips_dir / f"{track.name.replace('/', '_')}_30s.wav"
        if out.exists():
            continue
        import soundfile as sf

        clip = track.audio[: int(track.rate * CLIP_SECONDS)]
        sf.write(str(out), clip, int(track.rate))
        print(f"  {out.name} ({clip.shape[0] / track.rate:.0f}s @ {track.rate} Hz)")
    print(f"  片段 -> {clips_dir}")


def mtt() -> None:
    src = DATA / "mtt"
    anno = src / "annotations_final.csv"
    if not anno.exists():
        print("[MTT] 未下载，跳过")
        return

    mp3_dir = src / "mp3"
    if not mp3_dir.exists() and (src / "mp3.zip").exists():
        print("[MTT] 解压 mp3.zip（约 3 GB，几分钟）...")
        with zipfile.ZipFile(src / "mp3.zip") as zf:
            zf.extractall(src)

    raw = pd.read_csv(anno, sep="\t")
    tag_cols = [c for c in raw.columns if c not in ("clip_id", "mp3_path")]
    print(f"[MTT] {len(raw)} 条片段，{len(tag_cols)} 个标签")

    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        f = src / f"{split}_gt_mtt.tsv"
        if f.exists():
            ids = pd.read_csv(f, sep="\t", header=None, usecols=[0])[0].astype(str)
            split_of.update({i: split for i in ids})

    labels = raw[tag_cols].astype(int)
    out = pd.DataFrame(
        {
            "clip_id": raw["clip_id"].astype(str),
            "split": raw["clip_id"].astype(str).map(split_of).fillna("unknown"),
            "mp3_path": raw["mp3_path"],
            "n_tags": labels.sum(axis=1),
            "tags": labels.apply(
                lambda r: ",".join([t for t, v in r.items() if v == 1]), axis=1
            ),
        }
    )
    out.to_csv(RESULTS / "mtt_index.csv", index=False)
    counts = (
        labels.sum().sort_values(ascending=False).rename("count").rename_axis("tag").reset_index()
    )
    counts.to_csv(RESULTS / "mtt_tag_counts.csv", index=False)
    print(f"  划分 {out['split'].value_counts().to_dict()} -> mtt_index.csv / mtt_tag_counts.csv")
    print(f"  前 10 个高频标签：{counts['tag'].head(10).tolist()}")


def musiccaps() -> None:
    src = DATA / "musiccaps" / "musiccaps-public.csv"
    if not src.exists():
        print("[MusicCaps] 未下载，跳过")
        return
    df = pd.read_csv(src)
    keep = [c for c in
            ["ytid", "start_s", "end_s", "caption", "aspect_list",
             "is_balanced_subset", "is_audioset_eval"] if c in df.columns]
    df[keep].to_csv(RESULTS / "musiccaps_index.csv", index=False)
    print(f"[MusicCaps] {len(df)} 条标注，字段 {keep} -> musiccaps_index.csv")
    print("  注意：本仓只有文字标注，音频来自 YouTube，国内通常下载不到。")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    musdb18()
    mtt()
    musiccaps()
    print(f"\n完成，索引在 {RESULTS}")


if __name__ == "__main__":
    main()
