"""第 1 周双赛道基线之一：Demucs 预训练模型在 MUSDB18-HQ 测试集上的分离 SDR。

用官方评测库 museval 算 BSS Eval v4 指标，取每首曲目每帧的中位数，
再对曲目取中位数——这是 MUSDB 排行榜的标准口径。

用法：
    python scripts/10_demucs_baseline.py --model htdemucs --limit 3
    python scripts/10_demucs_baseline.py --model htdemucs_6s --limit 3
    python scripts/10_demucs_baseline.py --model htdemucs            # 全量 50 首，很慢

注意 htdemucs_6s 多出 guitar / piano 两轨，但 MUSDB18 没有这两轨的标注
（它们被算在 other 里），所以只评测双方都有的四轨。
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

# stempeg / musdb 依赖 ffmpeg 在 PATH 上；万一没配好，这里兜一次
_FFMPEG_FALLBACK = r"/usr/bin"
if shutil.which("ffmpeg") is None and Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")

import museval
import musdb
import numpy as np
import pandas as pd
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "musdb18hq_wav"
RESULTS = ROOT / "results"

EVAL_STEMS = ("vocals", "drums", "bass", "other")  # MUSDB18 有标注的四轨


def separate(model, track, device: str) -> dict[str, np.ndarray]:
    """跑一次分离，返回 {轨名: (channels, samples)}。

    demucs 4.0.1 没有 demucs.api，用官方 README 的底层接口：
    先按整体均值方差归一化，分离完再还原。
    """
    wav = torch.from_numpy(track.audio.T.astype(np.float32))
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(
            model, wav_norm[None], device=device, split=True, overlap=0.25, progress=False
        )[0]
    sources = sources * ref.std() + ref.mean()
    return {name: sources[i].numpy() for i, name in enumerate(model.sources)}


def evaluate_one(model, track, device: str, merge_other: bool = False) -> dict[str, float]:
    separated = separate(model, track, device)
    if merge_other:
        # 六轨模型的 guitar/piano 本来属于 MUSDB18 的 other，合并后才是公平对比
        merged = separated["other"] + separated["guitar"] + separated["piano"]
        separated = {k: v for k, v in separated.items() if k in EVAL_STEMS}
        separated["other"] = merged

    refs, ests = [], []
    for stem in EVAL_STEMS:
        ref = track.targets[stem].audio.T.astype(np.float32)   # (ch, samples)
        est = separated[stem]                                  # (ch, samples)
        n = min(ref.shape[1], est.shape[1])
        refs.append(ref[:, :n])
        ests.append(est[:, :n])

    # museval 要 (nsrc, nsampl, nchan)
    refs = np.stack(refs).transpose(0, 2, 1)
    ests = np.stack(ests).transpose(0, 2, 1)

    # museval 默认按帧计算（window=2s, hop=1.5s），取中位数是 MUSDB 的标准口径
    sdr, _, _, _, _ = museval.metrics.bss_eval(refs, ests, compute_permutation=False)
    return {stem: float(np.nanmedian(sdr[i])) for i, stem in enumerate(EVAL_STEMS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="htdemucs", choices=["htdemucs", "htdemucs_6s"])
    ap.add_argument("--limit", type=int, default=3, help="评测前 N 首；0 表示全部")
    ap.add_argument(
        "--device",
        default="auto",
        help="auto 表示有 CUDA 用 cuda，否则用 cpu",
    )
    ap.add_argument(
        "--data",
        default="musdb18hq_wav",
        help="数据目录名：musdb18hq_wav（HQ）或 musdb18（MP4 官方格式）",
    )
    ap.add_argument(
        "--merge-other",
        action="store_true",
        help="六轨版专用：把 guitar+piano+other 合并后再与参考 other 比较，避免标签口径不公平",
    )
    args = ap.parse_args()
    if args.device == "auto":
        import torch as _torch

        args.device = "cuda" if _torch.cuda.is_available() else "cpu"

    data_dir = ROOT / "data" / args.data
    if not data_dir.exists():
        print(f"没找到 {data_dir}")
        return 1

    db = musdb.DB(root=str(data_dir), is_wav=args.data.endswith("_wav"), subsets="test")
    tracks = db.tracks if args.limit <= 0 else db.tracks[: args.limit]
    print(f"模型 {args.model} | 设备 {args.device} | 评测 {len(tracks)} / {len(db.tracks)} 首")

    model = get_model(args.model)
    model.eval()
    if args.merge_other and not {"guitar", "piano"} <= set(model.sources):
        print("--merge-other 只对六轨模型有意义，已忽略")
        args.merge_other = False

    rows = []
    t_start = time.time()
    for i, track in enumerate(tracks, 1):
        t0 = time.time()
        scores = evaluate_one(model, track, args.device, merge_other=args.merge_other)
        dt = time.time() - t0
        rows.append({"track": track.name, **scores})
        avg = np.mean(list(scores.values()))
        print(
            f"  [{i}/{len(tracks)}] {track.name[:34]:34} "
            f"人声 {scores['vocals']:5.2f} 鼓 {scores['drums']:5.2f} "
            f"贝斯 {scores['bass']:5.2f} 其他 {scores['other']:5.2f} | 均值 {avg:5.2f} | {dt:.0f}s"
        )
        if i == 1:
            per_track = time.time() - t_start
            print(f"  → 单首约 {per_track:.0f} 秒，全量 {len(db.tracks)} 首预计 "
                  f"{per_track * len(db.tracks) / 3600:.1f} 小时")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit > 0 else ""
    if args.merge_other:
        suffix += "_mergeother"
    out = RESULTS / f"demucs_{args.model}_sdr{suffix}.csv"
    df.to_csv(out, index=False)

    print("\n=== 汇总（对曲目取中位数，单位 dB，越高越好）===")
    summary = df[list(EVAL_STEMS)].median().round(3)
    for stem, val in summary.items():
        print(f"  {stem:7} {val:6.3f}")
    print(f"  平均     {summary.mean():6.3f}")
    print(f"\n明细 -> {out}")
    print(f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
