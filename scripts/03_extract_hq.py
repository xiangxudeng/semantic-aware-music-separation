"""把 MUSDB18-HQ 的 parquet（内部是 FLAC）还原成 musdb 能直接读的 wav 分轨目录。

输出结构（musdb 的 wav 模式就是这个布局）：
    data/musdb18hq_wav/<split>/<track>/vocals.wav
                                   /drums.wav
                                   /bass.wav
                                   /other.wav
                                   /mixture.wav   ← 四条轨相加，MUSDB18 的定义

注意拆分口径：本数据集把官方 train 拆成了 train + validation，
所以还原时 validation 要并回 train，否则训练集只有 78 首。

用法：
    python scripts/03_extract_hq.py test          # 只还原测试集（第 1 周基线够用）
    python scripts/03_extract_hq.py train
    python scripts/03_extract_hq.py all
"""

from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "musdb18hq" / "data"
DST = ROOT / "data" / "musdb18hq_wav"
STEMS = ("vocals", "drums", "bass", "other")


def decode(blob: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return audio, sr


def convert(files: list[Path]) -> int:
    """同一个 split 的所有分片要一起处理：一首歌的分轨会散落在不同分片里。"""
    groups: dict[tuple[str, str], dict[str, bytes]] = defaultdict(dict)
    for i, parquet_file in enumerate(files, 1):
        table = pq.read_table(parquet_file, columns=["audio", "path", "instrument"])
        paths = table.column("path").to_pylist()
        instruments = table.column("instrument").to_pylist()
        blobs = [row["bytes"] for row in table.column("audio").to_pylist()]
        for path, inst, blob in zip(paths, instruments, blobs):
            parts = Path(path).parts       # musdb18hq/test/Skelpolu - Resurrection/drums.flac
            split = "test" if parts[-3] == "test" else "train"
            groups[(split, parts[-2])][inst] = blob
        print(f"  读取分片 [{i}/{len(files)}] {parquet_file.name}，累计 {len(groups)} 首")
        del table

    written = 0
    for (split, track), stems in sorted(groups.items()):
        missing = [s for s in STEMS if s not in stems]
        if missing:
            print(f"  !! {track} 缺少 {missing}，跳过")
            continue
        out_dir = DST / split / track
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = []
        for name in STEMS:
            audio, sr = decode(stems[name])
            sf.write(out_dir / f"{name}.wav", audio, sr, subtype="PCM_16")
            arrays.append(audio)
        sf.write(out_dir / "mixture.wav", sum(arrays), sr, subtype="PCM_16")
        written += 1
    return written


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "test"
    if which not in ("test", "train", "all"):
        print(f"用法：{argv[0]} [test|train|all]")
        return 2

    if not SRC.exists():
        print(f"没找到 {SRC}，先跑 01_download_data.py musdb18hq")
        return 1

    if which == "test":
        files = sorted(SRC.glob("test-*.parquet"))
    elif which == "train":
        files = sorted(SRC.glob("train-*.parquet")) + sorted(SRC.glob("validation-*.parquet"))
    else:
        files = sorted(SRC.glob("*.parquet"))

    print(f"待还原 {len(files)} 个 parquet -> {DST}")
    total = convert(files)
    print(f"完成，共 {total} 首")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
