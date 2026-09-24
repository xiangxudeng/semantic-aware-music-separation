"""第 3 周第 2 步：语义条件分离的对照训练。

这一版回答的问题是：**把 CLAP 语义向量注入 Demucs，到底有没有用？**

五组对照（唯一变量是条件注入方式，其他全部相同）
------------------------------------------------
| 组名 | 条件 | 说明 |
|---|---|---|
| `none` | 无 | 对照组：同样的步数、同样的学习率，只是不注入条件 |
| `film_enc` | FiLM → encoder.3 | 乘性+加性调制，注入编码器末端（384 通道）|
| `film_dec` | FiLM → decoder.0 | 注入解码器起始（192 通道）|
| `film_both` | FiLM → 两处 | |
| `add_enc` | 纯加性 → encoder.3 | FiLM 的对照：只用 beta（h + Wc），没有乘性调制 |

为什么要设 `none` 组：第 2 周已经确认"无条件微调"本身不涨点（甚至可能掉），
如果不设这组，就无法区分"涨跌来自语义条件"还是"来自多训了几百步"。

条件是什么
----------
MUSDB 没有风格/乐器标签，所以条件取**混合信号自己的 CLAP 语义向量**（512 维，
由 scripts/51_build_musdb_semantic.py 预先算好）。语义向量按曲目查表，不参与梯度。

用法（云端 GPU）：
    python scripts/52_train_conditioned.py --cond-mode film --cond-targets encoder.3 --steps 300
    python scripts/52_train_conditioned.py --cond-mode none  --steps 300 --tag none
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.augment import AUGMENT_PRESETS, WaveformAugment
from src.separation.film import ConditionedModel
from src.utils.run_log import run_record

_FFMPEG_FALLBACK = r"/usr/bin"
if shutil.which("ffmpeg") is None and Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")

DATA = ROOT / "data" / "musdb18hq_wav"
RESULTS = ROOT / "results"
# 展示顺序（报告里按这个顺序列）
DISPLAY_STEMS = ("vocals", "drums", "bass", "other")
# 模型输出顺序：htdemucs 的 sources 是 ('drums','bass','other','vocals')，
# **拼参考音轨时必须用这个顺序**，否则训练时模型被要求输出错位的轨，
# 验证时每一轨也在跟错误的参考比（第 3 周踩过这个坑，SI-SDR 直接掉到 −25 dB）
MODEL_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref.reshape(-1).double()
    est = est.reshape(-1).double()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(10 * torch.log10((target**2).sum() / (((est - target) ** 2).sum() + 1e-12)))


_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def load_track(track_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """直读 wav：走 musdb 接口每首整曲要 9.5 秒，直读只要 1 秒出头。"""
    import soundfile as sf

    if str(track_dir) not in _CACHE:
        mix, _ = sf.read(str(track_dir / "mixture.wav"), dtype="float32", always_2d=True)
        tgt = torch.stack([
            torch.from_numpy(sf.read(str(track_dir / f"{s}.wav"), dtype="float32",
                                     always_2d=True)[0].T)
            for s in MODEL_STEMS
        ])
        _CACHE[str(track_dir)] = (torch.from_numpy(mix.T), tgt)
        if len(_CACHE) > 30:
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[str(track_dir)]


def make_batch(dirs, conds, segment: int, batch: int, augment):
    ids = np.random.randint(0, len(dirs), size=batch)
    mixes, targets, cs = [], [], []
    for i in ids:
        mix, tgt = load_track(dirs[i])
        n = mix.shape[1]
        start = 0 if n <= segment else int(np.random.randint(0, n - segment))
        mixes.append(mix[:, start : start + segment])
        targets.append(tgt[:, :, start : start + segment])
        cs.append(conds[i])
    m, t = torch.stack(mixes), torch.stack(targets)
    if augment is not None:
        m, t = augment.apply_batch(m, t)
    return m, t, torch.from_numpy(np.stack(cs))


def validate(model, samples, dirs, conds, device, use_cond) -> dict[str, float]:
    from demucs.apply import apply_model

    # 必须在 eval 模式下验证：train 模式下 HTDemucs 的 segment 处理与归一化都不同，
    # 实测会得到 −25 dB 这种明显错误的结果（同一份权重 eval 下是 +7~11 dB）
    was_training = model.training
    model.eval()
    rows = []
    for idx in samples:
        d = dirs[idx]
        mix, tgt = load_track(d)
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        if use_cond:
            model.set_cond(torch.from_numpy(conds[idx][None]))
        elif hasattr(model, "set_cond"):   # 无条件组的模型是裸 HTDemucs，没有这个方法
            model.set_cond(None)
        with torch.no_grad():
            est = apply_model(model, ((mix - mean) / std)[None], device=device,
                              split=True, overlap=0.25, progress=False)[0]
        est = est * std + mean
        n = min(est.shape[-1], tgt.shape[-1])
        est_by_name = {name: est[i] for i, name in enumerate(MODEL_STEMS)}
        rows.append({
            s: si_sdr(tgt[MODEL_STEMS.index(s), :, :n], est_by_name[s][:, :n]) for s in DISPLAY_STEMS
        })
    if was_training:
        model.train()
    return {s: float(np.median([r[s] for r in rows])) for s in DISPLAY_STEMS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-mode", default="none", choices=["none", "film", "add"])
    ap.add_argument("--cond-targets", default="encoder.3",
                    help="逗号分隔；可选 encoder.3 / decoder.0 / tencoder.3 / tdecoder.0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--segment", type=float, default=7.8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-tracks", type=int, default=3)
    ap.add_argument("--augment", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="冻结主干只训 FiLM：此时不训练就是 0 变化的天然基线，条件效果可单独量出")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("!! 要求 cuda 但不可用，退出（避免误在 CPU 上跑几小时）")
        return 1

    sem = np.load(RESULTS / "musdb_semantic.npy").astype(np.float32)
    index = pd.read_csv(RESULTS / "musdb_semantic_index.csv")
    vec_of = {r.track: sem[i] for i, r in enumerate(index.itertuples())}
    tr_dirs = [DATA / "train" / t for t in index[index.split == "train"].track]
    te_dirs = [DATA / "test" / t for t in index[index.split == "test"].track]
    tr_cond = np.stack([vec_of[d.name] for d in tr_dirs])
    te_cond = np.stack([vec_of[d.name] for d in te_dirs])
    val_samples = list(range(min(args.val_tracks, len(te_dirs))))

    tags = args.cond_targets if args.cond_mode != "none" else "none"
    tag = args.tag or f"{args.cond_mode}_{tags.replace('.', '').replace(',', '-')}"
    config = vars(args) | {"tag": tag, "train_tracks": len(tr_dirs), "cond_dim": int(sem.shape[1])}

    with run_record(f"w3_conditioned_{tag}", config=config) as run:
        augmenter = WaveformAugment(AUGMENT_PRESETS[args.augment]) if args.augment != "none" else None
        from demucs.pretrained import get_model

        bag = get_model("htdemucs")
        base = bag.models[0] if hasattr(bag, "models") else bag
        base.to(args.device).train()

        if args.cond_mode == "none":
            model = base
            targets = []
        else:
            layout = "channel_last" if args.cond_targets.startswith("tencoder") else "channel_first"
            targets = [(t.strip(), layout) for t in args.cond_targets.split(",") if t.strip()]
            # 形状探测用的假输入必须和主干在同一个设备上：
            # 主干已经 .to(device) 了，假输入还在 CPU 会直接报
            # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"
            sample = torch.zeros(1, 2, int(args.segment * 44100), device=args.device)
            model = ConditionedModel(base, cond_dim=int(sem.shape[1]), targets=targets,
                                     sample_input=sample, mode="both" if args.cond_mode == "film" else "beta")
            model.to(args.device)
        if args.freeze_backbone and targets:
            # 只训条件注入层：主干完全冻结，于是"不训练"就是 0 变化的天然基线，
            # 任何涨跌都只能归因于条件本身，不再和"主干微调"混在一起
            for p in model.parameters():
                p.requires_grad = False
            for film in model.films.values():
                for p in film.parameters():
                    p.requires_grad = True
            print("已冻结主干，只训练 FiLM", flush=True)
        params = [p for p in model.parameters() if p.requires_grad]
        n_film = sum(p.numel() for p in model.film_parameters()) if targets else 0
        print(f"条件方式 {args.cond_mode}｜注入点 {targets}｜FiLM 参数 {n_film / 1e3:.1f}K "
              f"｜可训练 {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
        run.note(f"条件 {args.cond_mode}/{targets}，FiLM 参数 {n_film / 1e3:.1f}K")

        before = validate(model, val_samples, te_dirs, te_cond, args.device, bool(targets))
        print("训练前 SI-SDR：" + "  ".join(f"{k} {v:.2f}" for k, v in before.items()), flush=True)

        opt = torch.optim.Adam(params, lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")
        log_path = RESULTS / f"cond_train_{tag}.csv"
        t0 = time.time()
        last = 0.0
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "elapsed_s"])
            for step in range(1, args.steps + 1):
                mix, tgt, cond = make_batch(tr_dirs, tr_cond, int(args.segment * 44100),
                                            args.batch_size, augmenter)
                mix, tgt = mix.to(args.device), tgt.to(args.device)
                ref = mix.mean(1, keepdim=True)
                mean, std = ref.mean(), ref.std() + 1e-8
                x = (mix - mean) / std
                if targets:
                    model.set_cond(cond.to(args.device))
                with torch.autocast("cuda", enabled=args.device == "cuda"):
                    out = model(x)
                    loss = torch.nn.functional.l1_loss(out, tgt / std)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                last = float(loss)
                w.writerow([step, f"{last:.5f}", f"{time.time() - t0:.0f}"])
                f.flush()
                if step % 50 == 0:
                    print(f"  step {step}/{args.steps} loss {last:.4f} ({time.time() - t0:.0f}s)", flush=True)

        after = validate(model, val_samples, te_dirs, te_cond, args.device, bool(targets))
        print("训练后 SI-SDR：" + "  ".join(f"{k} {v:.2f}" for k, v in after.items()), flush=True)
        delta = {k: after[k] - before[k] for k in DISPLAY_STEMS}
        b_mean, a_mean = float(np.mean(list(before.values()))), float(np.mean(list(after.values())))
        print(f"变化：" + "  ".join(f"{k} {v:+.2f}" for k, v in delta.items())
              + f"  | 均值 {b_mean:.2f} → {a_mean:.2f} ({a_mean - b_mean:+.3f})", flush=True)

        row = {"组": tag, "条件": args.cond_mode, "注入点": str(targets), "步数": args.steps,
               "训练前均值": round(b_mean, 3), "训练后均值": round(a_mean, 3),
               "变化": round(a_mean - b_mean, 3), "末步loss": round(last, 5),
               **{f"Δ{k}": round(delta[k], 3) for k in DISPLAY_STEMS},
               "FiLM参数K": round(n_film / 1e3, 1)}
        out_csv = RESULTS / "cond_separation_compare.csv"
        new = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

        ckpt = RESULTS / f"cond_model_{tag}.th"
        torch.save({"model": model.state_dict(), "args": vars(args), "stems": MODEL_STEMS}, ckpt)
        run.artifact(log_path, out_csv, ckpt)
        run.note(f"{tag}：训练前 {b_mean:.2f} → 训练后 {a_mean:.2f}（{a_mean - b_mean:+.3f} dB），"
                 f"其他轨 {delta['other']:+.2f}")
        print(f"\n汇总 -> {out_csv}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
