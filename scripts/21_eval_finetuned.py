"""第 2 周：微调后的 htdemucs 在 MUSDB18-HQ 测试集上的分离 SDR。

对照基线：未微调的 htdemucs，同一测试集 50 首，四轨平均 8.86 dB
（results/demucs_htdemucs_sdr.csv，由 10_demucs_baseline.py 产出）。

这个对照是干净的：htdemucs.yaml 里只挂了一个子模型（955717e8），
微调就是拿这个子模型继续训的，所以前后是同结构、同参数量的对比。

museval 是 CPU 单线程实现，单首约 187 秒，是整条链路的瓶颈。
本脚本按曲目做多进程并行：分离用 GPU（约 7 秒/首），评测用 CPU。

用法：
    python scripts/21_eval_finetuned.py --ckpt results/finetuned_ft1.th --limit 3 --workers 3
    python scripts/21_eval_finetuned.py --ckpt results/finetuned_ft1.th --workers 6   # 全量 50 首
    python scripts/21_eval_finetuned.py --ckpt "" --limit 2                            # 只测未微调基线
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# stempeg / musdb 需要 ffmpeg 在 PATH 上；万一没配好，这里兜一次
_FFMPEG_FALLBACK = r"/usr/bin"
if shutil.which("ffmpeg") is None and Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parents[1]
EVAL_STEMS = ("vocals", "drums", "bass", "other")

_G: dict = {}


def list_tracks(test_dir: Path) -> list[Path]:
    """MUSDB18-HQ 的目录布局天然可用：<曲目>/{mixture,vocals,drums,bass,other}.wav。"""
    return sorted(
        d
        for d in test_dir.iterdir()
        if d.is_dir() and (d / "mixture.wav").exists() and (d / "vocals.wav").exists()
    )


def build_model(ckpt: str, model_name: str):
    """微调权重存的是 htdemucs 包里的那个子模型，所以取 models[0] 再灌权重。"""
    import torch
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if ckpt:
        model = bag.models[0] if hasattr(bag, "models") else bag
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  !! 权重对不上：缺失 {len(missing)} 项，多余 {len(unexpected)} 项", flush=True)
        if isinstance(state, dict) and "args" in state:
            print(f"  微调配置：{state['args']}", flush=True)
    else:
        model = bag
    model.eval()
    return model


def out_path(args, ckpt: str) -> Path:
    tag = args.tag or ("finetuned" if ckpt else "base")
    suffix = f"_limit{args.limit}" if args.limit > 0 else ""
    return ROOT / "results" / f"demucs_{tag}_sdr{suffix}.csv"


def read_done(path: Path) -> tuple[list[dict], set[str]]:
    """读回已经完成的行，用于断点续跑。"""
    if not path.exists():
        return [], set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("track")]
    except Exception:  # noqa: BLE001
        return [], set()
    for r in rows:
        for k in list(r):
            if k != "track":
                r[k] = float(r[k])
    return rows, {r["track"] for r in rows}


def _init(ckpt: str, model_name: str, device: str, data_root: str):
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    _G["device"] = device
    _G["data_root"] = Path(data_root)
    _G["model"] = build_model(ckpt, model_name)


def _eval_one(track_dir_str: str) -> dict:
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    import museval

    track_dir = Path(track_dir_str)
    mix, _ = sf.read(str(track_dir / "mixture.wav"), dtype="float32", always_2d=True)
    wav = torch.from_numpy(mix.T)  # (ch, T)
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    t0 = time.time()
    with torch.no_grad():
        sources = apply_model(
            _G["model"], wav_norm[None], device=_G["device"], split=True, overlap=0.25, progress=False
        )[0]
    sources = sources * ref.std() + ref.mean()
    est = {name: sources[i].numpy() for i, name in enumerate(_G["model"].sources)}
    t_sep = time.time() - t0

    refs, ests = [], []
    for stem in EVAL_STEMS:
        r, _ = sf.read(str(track_dir / f"{stem}.wav"), dtype="float32", always_2d=True)
        e = est[stem].T
        n = min(len(r), len(e))
        refs.append(r[:n])
        ests.append(e[:n])

    t1 = time.time()
    sdr, _, _, _, _ = museval.metrics.bss_eval(
        np.stack(refs), np.stack(ests), compute_permutation=False
    )
    t_eval = time.time() - t1

    row = {"track": track_dir.name}
    row.update({s: float(np.nanmedian(sdr[i])) for i, s in enumerate(EVAL_STEMS)})
    row["sep_s"] = round(t_sep, 1)
    row["museval_s"] = round(t_eval, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/finetuned_ft1.th", help="微调权重；传空串表示测未微调基线")
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--data", default="musdb18hq_wav")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 首；0 表示全部 50 首")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数；0 表示 CPU 核数的一半，最多 8")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    ap.add_argument("--threads-per-worker", type=int, default=0)
    args = ap.parse_args()

    if args.ckpt.strip().lower() in ("", "none", "base", "0"):
        ckpt = ""  # 只评测未微调的基线
    else:
        ckpt = (
            str((ROOT / args.ckpt).resolve())
            if not Path(args.ckpt).is_absolute()
            else args.ckpt
        )
    if ckpt and not Path(ckpt).exists():
        print(f"没找到权重 {ckpt}")
        return 1

    data_dir = ROOT / "data" / args.data
    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"没找到测试集 {test_dir}")
        return 1

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        args.workers = args.workers or max(1, min(4, (os.cpu_count() or 4) // 4))
    else:
        args.workers = args.workers or 6

    if args.workers > 1:
        # 每个 worker 跑一首曲子；如果放任 BLAS 各自开满线程，几十个进程抢核会互相拖垮。
        # 实测两个 worker 不限制线程时，一首曲子的评测要 10 分钟以上（单进程只要 3 分钟）。
        per = str(args.threads_per_worker or max(1, (os.cpu_count() or 4) // args.workers))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = per
        os.environ["TORCH_THREADS"] = per
        print(f"每个 worker 限制 {per} 线程", flush=True)

    tracks = list_tracks(test_dir)
    if args.limit > 0:
        tracks = tracks[: args.limit]
    target = out_path(args, ckpt)
    rows, done = read_done(target)
    if done:
        tracks = [t for t in tracks if t.name not in done]
        print(f"断点续跑：已有 {len(done)} 首，剩 {len(tracks)} 首", flush=True)
    print(f"权重 {ckpt or '未微调基线'} | 设备 {args.device} | 并行 {args.workers} | 评测 {len(tracks)} 首", flush=True)
    if not tracks:
        print("没有要跑的曲目，直接汇总")

    t_start = time.time()
    if tracks and args.workers <= 1:
        _init(ckpt, args.model, args.device, str(data_dir))
        for i, t in enumerate(tracks, 1):
            row = _eval_one(str(t))
            rows.append(row)
            avg = np.mean([row[s] for s in EVAL_STEMS])
            print(
                f"  [{i}/{len(tracks)}] {row['track'][:30]:30} 均值 {avg:5.2f} dB "
                f"| 分离 {row['sep_s']}s 评测 {row['museval_s']}s",
                flush=True,
            )
            _write(rows, target)
    elif tracks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            args.workers, initializer=_init, initargs=(ckpt, args.model, args.device, str(data_dir))
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(_eval_one, [str(t) for t in tracks]), 1):
                rows.append(row)
                avg = np.mean([row[s] for s in EVAL_STEMS])
                print(
                    f"  [{i}/{len(tracks)}] {row['track'][:30]:30} 均值 {avg:5.2f} dB "
                    f"| 分离 {row['sep_s']}s 评测 {row['museval_s']}s",
                    flush=True,
                )
                _write(rows, target)

    df_summary = {s: float(np.median([r[s] for r in rows])) for s in EVAL_STEMS}
    print("\n=== 汇总（对曲目取中位数，单位 dB，越高越好）===")
    for s, v in df_summary.items():
        print(f"  {s:7} {v:6.3f}")
    overall = float(np.mean(list(df_summary.values())))
    print(f"  平均     {overall:6.3f}")
    if ckpt:
        base = {"vocals": 9.211, "drums": 10.096, "bass": 9.686, "other": 6.462}
        print("\n=== 相对未微调基线（8.86 dB）的变化 ===")
        for s in EVAL_STEMS:
            print(f"  {s:7} {df_summary[s] - base[s]:+6.3f}")
        print(f"  平均     {overall - 8.863:+6.3f}")
    print(f"\n总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    return 0


def _write(rows: list[dict], target: Path) -> None:
    """每首写完就落盘，中途断了不至于全丢。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["track"]))
    tmp.replace(target)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        raise SystemExit(0)
    raise SystemExit(main())
