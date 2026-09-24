"""第 3 周第 1 步：给 MUSDB 的 150 首曲子算 CLAP 语义向量，作为条件分离的输入。

为什么条件来自音频本身
----------------------
计划里写的是"按风格/乐器标签构建条件样本"，但 **MUSDB18 没有任何风格或乐器标签**，
它只有四条分轨。所以条件只能从音频本身取：用 CLAP 对**混合信号**提取 512 维语义向量。

这在工程上是成立的（推理时混合信号就在手上），但要说清一个局限：
条件与输入同源，模型有可能学会"忽略条件"。所以第 52 号实验的对照里必须包含
"无条件微调"这一组，否则分不清涨跌来自条件信息还是来自多训练了几步。

产出
----
- results/musdb_semantic.npy     (150, 512) 语义向量，L2 归一化
- results/musdb_semantic_index.csv  行序与 npy 一一对应（曲目名 / 划分 / 路径）

用法：
    python scripts/51_build_musdb_semantic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.semantic.clap_encoder import ClapEncoder
from src.utils.run_log import run_record

DATA = ROOT / "data" / "musdb18hq_wav"
RESULTS = ROOT / "results"


def main() -> int:
    with run_record("w3_build_musdb_semantic", config={"windows": 3, "clip_seconds": 10}) as run:
        tracks = []
        for split in ("train", "test"):
            d = DATA / split
            for t in sorted(p for p in d.iterdir() if p.is_dir()):
                tracks.append({"track": t.name, "split": split, "path": str(t / "mixture.wav")})
        print(f"MUSDB 曲目 {len(tracks)} 首（train {sum(t['split'] == 'train' for t in tracks)}）")

        enc = ClapEncoder(device="cpu", windows=3)
        vecs = []
        for i, t in enumerate(tracks, 1):
            v = enc.encode_file(t["path"])
            vecs.append(v)
            if i % 25 == 0 or i == 1:
                print(f"  {i}/{len(tracks)}  {enc.stats.seconds_per_clip:.3f}s/片段  "
                      f"已用 {enc.stats.wall_seconds / 60:.1f} 分钟", flush=True)
        emb = np.stack(vecs).astype(np.float32)

        np.save(RESULTS / "musdb_semantic.npy", emb)
        pd.DataFrame(tracks).to_csv(RESULTS / "musdb_semantic_index.csv", index=False)

        norms = np.linalg.norm(emb, axis=1)
        print(f"\n语义向量 {emb.shape}，范数 {norms.min():.4f}~{norms.max():.4f}")
        print(enc.report())
        # 向量必须彼此不同，否则条件等于常量、失去意义
        inter = emb @ emb.T
        off = inter[~np.eye(len(emb), dtype=bool)]
        print(f"两两余弦：均值 {off.mean():.3f}，最大 {off.max():.3f}（越大说明曲子越像）")
        run.artifact(RESULTS / "musdb_semantic.npy", RESULTS / "musdb_semantic_index.csv")
        run.note(f"150 首曲目语义向量就绪；平均每片段 {enc.stats.seconds_per_clip:.3f}s；两两余弦均值 {off.mean():.3f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
