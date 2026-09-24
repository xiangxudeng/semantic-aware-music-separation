"""Demucs 轻量微调（第 2 周）。

在 MUSDB18-HQ 训练集上对预训练 htdemucs 做微调。设计取舍：

- 随机片段采样，不做整曲训练：显存和时间都可控
- L1 波形损失，与 Demucs 官方一致
- 归一化方式与推理链路保持一致（按 mixture 的均值方差归一，输出再还原）
- 验证用 SI-SDR（快），museval 留给最终评测——实测 museval 一首曲子要三分钟
- 支持冻结编码器，只训解码器与瓶颈层，显存占用更低

用法：
    python scripts/20_finetune_demucs.py --epochs 3 --steps-per-epoch 200 --device cuda
    python scripts/20_finetune_demucs.py --freeze-encoder --lr 1e-4     # 更省显存的模式
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augment import AUGMENT_PRESETS, AugmentConfig, WaveformAugment  # noqa: E402

# stempeg / musdb 需要 ffmpeg 在 PATH 上；万一没配好，这里兜一次
_FFMPEG_FALLBACK = r"/usr/bin"
if shutil.which("ffmpeg") is None and Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    """尺度不变信噪比，作为训练期间的快速指标。"""
    ref = ref.reshape(-1).double()
    est = est.reshape(-1).double()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(10 * torch.log10((target**2).sum() / (((est - target) ** 2).sum() + 1e-12)))


_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def load_track(track, stems: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """直接读 wav 文件，绕过 musdb 的加载开销。

    实测 musdb 每首整曲要 9.5 秒，而纯 GPU 计算只要 1.2 秒——数据加载才是瓶颈。
    数据本来就是 <曲目>/mixture.wav 与 <曲目>/<stem>.wav 的标准布局，直接读即可。
    """
    import soundfile as sf

    # musdb 的 track.path 指向 mixture.wav 本身，取它所在目录作为基准
    base = Path(track.path).parent
    mix, _ = sf.read(str(base / "mixture.wav"), dtype="float32", always_2d=True)
    targets = torch.stack(
        [
            torch.from_numpy(
                sf.read(str(base / f"{s}.wav"), dtype="float32", always_2d=True)[0].T
            )
            for s in stems
        ]
    )
    return torch.from_numpy(mix.T), targets


def cached_track(index: int, track, stems: list[str], cache_tracks: int):
    """简单的 FIFO 缓存，避免每次都从磁盘重读整曲。"""
    if cache_tracks <= 0:
        return load_track(track, stems)
    if index not in _CACHE:
        if len(_CACHE) >= cache_tracks:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[index] = load_track(track, stems)
    return _CACHE[index]


def make_batch(
    tracks, stems: list[str], segment: int, batch: int, cache_tracks: int = 0, augment=None
):
    """从随机曲目里各取一个随机片段，拼成一个 batch（B, ch, T）与（B, S, ch, T）。"""
    ids = np.random.randint(0, len(tracks), size=batch)
    mixes, targets = [], []
    for i in ids:
        mix, target = cached_track(int(i), tracks[int(i)], stems, cache_tracks)
        n = mix.shape[1]
        start = 0 if n <= segment else int(np.random.randint(0, n - segment))
        mixes.append(mix[:, start : start + segment])
        targets.append(target[:, :, start : start + segment])
    mix_b, tgt_b = torch.stack(mixes), torch.stack(targets)  # (B, ch, T), (B, S, ch, T)
    if augment is not None:
        mix_b, tgt_b = augment.apply_batch(mix_b, tgt_b)
    return mix_b, tgt_b


def validate(model, tracks, stems: list[str], device: str, max_tracks: int) -> dict[str, float]:
    """每个 epoch 用 SI-SDR 快速看一眼有没有在学。

    museval 一首曲子要三分钟，不适合放在训练循环里；SI-SDR 只要几秒，
    足够判断"有没有变好"这个方向性问题。
    """
    import soundfile as sf
    from demucs.apply import apply_model

    was_training = model.training
    model.eval()
    rows = []
    for track in tracks[:max_tracks]:
        base = Path(track.path).parent
        mix, _ = sf.read(str(base / "mixture.wav"), dtype="float32", always_2d=True)
        wav = torch.from_numpy(mix.T)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            est = apply_model(
                model, ((wav - mean) / std)[None], device=device, split=True, overlap=0.25, progress=False
            )[0]
        est = est * std + mean
        tgt = torch.stack(
            [
                torch.from_numpy(
                    sf.read(str(base / f"{s}.wav"), dtype="float32", always_2d=True)[0].T
                )
                for s in stems
            ]
        )
        n = min(est.shape[-1], tgt.shape[-1])
        rows.append({s: si_sdr(tgt[i, :, :n], est[i, :, :n]) for i, s in enumerate(stems)})
    if was_training:
        model.train()
    return {s: float(np.median([r[s] for r in rows])) for s in stems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--data", default="musdb18hq_wav")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--segment", type=float, default=10.0, help="训练片段长度（秒）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-encoder", action="store_true", help="冻结编码器，只训其余部分")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="", help="输出文件名后缀，便于区分不同配置")
    ap.add_argument("--val-tracks", type=int, default=3, help="每轮验证用几首测试曲目")
    ap.add_argument("--limit-tracks", type=int, default=0, help="只用前 N 首训练，0 表示全部")
    ap.add_argument(
        "--cache-tracks",
        type=int,
        default=30,
        help="内存中缓存多少首整曲（每首约 370 MB）；0 表示不缓存",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accum-steps", type=int, default=1, help="梯度累积步数，等效放大 batch")
    ap.add_argument(
        "--augment",
        default="none",
        help="none / default / gain_only / no_gain / mask_only，或一个 json/yaml 配置路径",
    )
    args = ap.parse_args()

    if args.augment in ("none", "off", ""):
        augmenter = None
    elif args.augment in AUGMENT_PRESETS:
        augmenter = WaveformAugment(AUGMENT_PRESETS[args.augment])
    elif Path(args.augment).exists():
        augmenter = WaveformAugment(AugmentConfig.from_file(args.augment))
    else:
        print(f"--augment 不认识：{args.augment}")
        return 1

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import musdb

    data_root = ROOT / "data" / args.data
    if not data_root.exists():
        print(f"没找到 {data_root}")
        return 1

    db_train = musdb.DB(root=str(data_root), is_wav=True, subsets="train")
    tracks = db_train.tracks[: args.limit_tracks] if args.limit_tracks else db_train.tracks
    print(f"训练曲目 {len(tracks)} 首 | 设备 {args.device} | 片段 {args.segment}s | batch {args.batch_size}")

    model = get_model(args.model)
    # get_model 返回的是 BagOfModels，它的 forward 只服务于推理（内部带 no_grad）；
    # 训练要用内层的真实模型。单模型包里只有一层。
    if hasattr(model, "models"):
        print(f"解包 BagOfModels（含 {len(model.models)} 个子模型），取第一个用于训练")
        model = model.models[0]
    stems = list(model.sources)
    model.to(args.device).train()
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {trainable / 1e6:.1f} M / 总参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    if augmenter is not None:
        print(augmenter.describe(), flush=True)
    else:
        print("数据增强：关闭", flush=True)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")

    tag = args.tag or f"{args.model}_lr{args.lr}_bs{args.batch_size}" + ("_frozen" if args.freeze_encoder else "")
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    ckpt_path = results / f"finetuned_{tag}.th"
    log_path = results / f"finetune_{tag}.csv"
    val_path = results / f"finetune_val_{tag}.csv"

    # 验证用的测试曲目（只读，不参与训练）
    import musdb

    db_val = musdb.DB(root=str(data_root), is_wav=True, subsets="test")
    val_tracks = db_val.tracks[: args.val_tracks]
    base_scores = validate(model, val_tracks, stems, args.device, args.val_tracks)
    print(
        "训练前 SI-SDR 基线：" + "  ".join(f"{s} {v:.2f}" for s, v in base_scores.items()),
        flush=True,
    )
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *stems, "mean"])
        writer.writerow([0, *[f"{base_scores[s]:.4f}" for s in stems],
                         f"{np.mean(list(base_scores.values())):.4f}"])

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "lr", "elapsed_s"])

        step = 0
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            for i_step in range(args.steps_per_epoch):
                accum = max(1, args.accum_steps)
                optim.zero_grad(set_to_none=True)
                for _ in range(accum):
                    mix, target = make_batch(
                        tracks,
                        stems,
                        int(args.segment * 44100),
                        args.batch_size,
                        args.cache_tracks,
                        augmenter,
                    )
                    mix, target = mix.to(args.device), target.to(args.device)

                    # 与推理链路一致的归一化
                    ref = mix.mean(1, keepdim=True)
                    mean, std = ref.mean(), ref.std() + 1e-8
                    x = (mix - mean) / std

                    with torch.autocast("cuda", enabled=args.device == "cuda"):
                        out = model(x)
                        loss = torch.nn.functional.l1_loss(out, target / std) / accum

                    scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optim)
                scaler.update()

                step += 1
                writer.writerow([epoch, step, f"{loss.item():.5f}", f"{args.lr:.2e}", f"{time.time() - t_start:.0f}"])
                f.flush()
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step} loss {loss.item():.4f} ({time.time() - t_start:.0f}s)", flush=True)

            print(f"epoch {epoch} 完成，loss {loss.item():.4f}", flush=True)
            scores = validate(model, val_tracks, stems, args.device, args.val_tracks)
            print(
                f"  epoch {epoch} 验证 SI-SDR：" + "  ".join(f"{s} {v:.2f}" for s, v in scores.items())
                + f"  | 均值 {np.mean(list(scores.values())):.2f}"
                + f"（训练前 {np.mean(list(base_scores.values())):.2f}）",
                flush=True,
            )
            # 注意：这里不能复用上面的 f，那个是训练日志的文件对象
            with open(val_path, "a", newline="", encoding="utf-8") as val_f:
                vw = csv.writer(val_f)
                vw.writerow([epoch, *[f"{scores[s]:.4f}" for s in stems],
                             f"{np.mean(list(scores.values())):.4f}"])
            # 每轮存一次，中途崩了不至于全丢
            torch.save({"model": model.state_dict(), "args": vars(args), "stems": stems}, ckpt_path)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "stems": stems},
                results / f"finetuned_{tag}_ep{epoch}.th",
            )

    torch.save({"model": model.state_dict(), "args": vars(args), "stems": stems}, ckpt_path)
    print(f"\n权重 -> {ckpt_path}\n日志 -> {log_path}")
    if augmenter is not None:
        print(f"增强触发次数统计：{augmenter.applied}", flush=True)
    print(f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
