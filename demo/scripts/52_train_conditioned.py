"""四轨条件分离训练与逐曲目SI-SDR评测。

全曲级条件在当前预算下的作用由对照实验判断，不预设无信息归因。
训练seed、推理设置和冻结方式须与比较对象一致。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = Path(os.environ.get("AI_MUSIC_ASSETS", ROOT.parent / "ai_music")).resolve()

import numpy as np
import pandas as pd
import torch

from src.data.augment import AUGMENT_PRESETS, WaveformAugment
from src.separation.film import ConditionedModel
from src.utils.run_log import run_record

DATA = ASSETS / "data" / "musdb18hq_wav"
RESULTS = ASSETS / "results"
# 展示顺序（报告里按这个顺序列）
DISPLAY_STEMS = ("vocals", "drums", "bass", "other")
# 模型输出顺序：htdemucs 的 sources 是 ('drums','bass','other','vocals')，
# **拼参考音轨时必须用这个顺序**，否则训练时模型被要求输出错位的轨，
# 验证时每一轨也在跟错误的参考比（第 3 周踩过这个坑，SI-SDR 直接掉到 −25 dB）
MODEL_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")


def build_conditions(
    mode: str, vec_of: dict, tracks: list[str], text_vec: np.ndarray | None = None
) -> np.ndarray:
    """按 mode 构造条件矩阵 (N, 512)。

    - `track`：取每首曲子自己的向量，测试全曲表示的作用。
    - `cross`：取邻曲向量（错位一位），测试条件与曲目对应关系。
      这些对照不能单独证明语义信息无用或确定掉点的原因。
    - `text`：所有曲子共用同一段文本的 CLAP 嵌入，完全不来自音频。
    """
    own = np.stack([vec_of[t] for t in tracks]).astype(np.float32)
    if mode == "track":
        return own
    if mode == "cross":
        # 错位一位：谁也不会拿到自己的向量（N=1 时退化成自己，实际不会出现）
        return np.roll(own, 1, axis=0)
    if mode == "text":
        if text_vec is None:
            raise SystemExit("--cond-source text 需要先算出文本嵌入")
        return np.repeat(
            np.asarray(text_vec, dtype=np.float32)[None], len(tracks), axis=0
        )
    if mode == "zero":
        # 全零条件：信息量为零、且和输入完全无关的**常量**条件。
        # 可学习FiLM仍能产生常量仿射；这是参数化对照，不能单独确定语义因果。
        return np.zeros_like(own)
    raise SystemExit(f"未知条件源 {mode}")


def self_check() -> int:
    """不加载任何模型，只验三种条件源的构造（cross 必须真的换了一首曲子）。"""
    rng = np.random.default_rng(0)
    keys = [f"t{i}" for i in range(5)]
    vecs = {k: rng.normal(size=512).astype(np.float32) for k in keys}

    own = build_conditions("track", vecs, keys)
    assert own.shape == (5, 512)
    for i, t in enumerate(keys):
        assert np.allclose(own[i], vecs[t]), "track 模式应该取自己的向量"

    cross = build_conditions("cross", vecs, keys)
    for i, t in enumerate(keys):
        assert not np.allclose(cross[i], vecs[t]), (
            f"cross 模式第 {i} 首拿到了自己的向量"
        )
    assert np.allclose(cross[0], vecs[keys[-1]]), "cross 应该是错位一位"

    tv = rng.normal(size=512).astype(np.float32)
    txt = build_conditions("text", vecs, keys, tv)
    assert txt.shape == (5, 512) and np.allclose(txt, tv[None]), (
        "text 模式应该所有曲子共用同一向量"
    )
    zero = build_conditions("zero", vecs, keys)
    assert zero.shape == (5, 512) and np.count_nonzero(zero) == 0
    print("self-check ok：track / cross / text / zero 条件源构造正确")
    return 0


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref.reshape(-1).double()
    est = est.reshape(-1).double()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(
        10 * torch.log10((target**2).sum() / (((est - target) ** 2).sum() + 1e-12))
    )


_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def load_track(track_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """直读 wav：走 musdb 接口每首整曲要 9.5 秒，直读只要 1 秒出头。"""
    import soundfile as sf

    if str(track_dir) not in _CACHE:
        mix, _ = sf.read(
            str(track_dir / "mixture.wav"), dtype="float32", always_2d=True
        )
        tgt = torch.stack(
            [
                torch.from_numpy(
                    sf.read(
                        str(track_dir / f"{s}.wav"), dtype="float32", always_2d=True
                    )[0].T
                )
                for s in MODEL_STEMS
            ]
        )
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


def score_track(model, idx, dirs, conds, device, use_cond) -> dict[str, float]:
    """单首曲目的逐轨 SI-SDR。抽出来是为了能存"逐曲目"明细（置信区间要用）。"""
    from demucs.apply import apply_model

    mix, tgt = load_track(dirs[idx])
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    if use_cond:
        model.set_cond(torch.from_numpy(conds[idx][None]))
    elif hasattr(model, "set_cond"):  # 无条件组的模型是裸 HTDemucs，没有这个方法
        model.set_cond(None)
    # 同一曲目在各组使用相同随机时移；旧结果未固定Python随机种子。
    random.seed(idx)
    with torch.no_grad():
        est = apply_model(
            model,
            ((mix - mean) / std)[None],
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    est = est * std + mean
    n = min(est.shape[-1], tgt.shape[-1])
    est_by_name = {name: est[i] for i, name in enumerate(MODEL_STEMS)}
    return {
        s: si_sdr(tgt[MODEL_STEMS.index(s), :, :n], est_by_name[s][:, :n])
        for s in DISPLAY_STEMS
    }


def validate(model, samples, dirs, conds, device, use_cond) -> dict[str, float]:
    # 必须在 eval 模式下验证：train 模式下 HTDemucs 的 segment 处理与归一化都不同，
    # 实测会得到 −25 dB 这种明显错误的结果（同一份权重 eval 下是 +7~11 dB）
    was_training = model.training
    model.eval()
    rows = [score_track(model, idx, dirs, conds, device, use_cond) for idx in samples]
    if was_training:
        model.train()
    return {s: float(np.median([r[s] for r in rows])) for s in DISPLAY_STEMS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-mode", default="none", choices=["none", "film", "add"])
    ap.add_argument(
        "--cond-targets",
        default="encoder.3",
        help="逗号分隔；可选 encoder.3 / decoder.0 / tencoder.3 / tdecoder.0",
    )
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--segment", type=float, default=7.8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true", help="使用FP32训练；记录P40同机对照的实际精度")
    ap.add_argument("--val-tracks", type=int, default=3)
    ap.add_argument("--augment", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="只评测存档权重，不训练；会把逐曲目明细存下来供置信区间使用",
    )
    ap.add_argument("--ckpt", default="", help="--eval-only 时要加载的权重路径")
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结主干只训 FiLM：此时不训练就是 0 变化的天然基线，条件效果可单独量出",
    )
    ap.add_argument(
        "--cond-source",
        default="track",
        choices=["track", "cross", "text", "zero"],
        help="条件从哪来：track=混合信号自己的向量（第 3 周，负结果）/ "
        "cross=别的曲子的向量 / text=固定文本的 CLAP 嵌入",
    )
    ap.add_argument(
        "--cond-text",
        default="This is a piece of music",
        help="--cond-source text 时用的条件文本",
    )
    ap.add_argument("--cond-text-embedding", default="", help="相同CLAP与文本的预计算512维向量；省去云端重复加载编码器")
    ap.add_argument(
        "--self-check", action="store_true", help="只验条件构造逻辑，不加载模型"
    )
    args = ap.parse_args()

    if args.self_check:
        return self_check()

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
    # 条件来源：track = 第 3 周的做法；cross / text 是第 4 周用来定位根因的对照
    if args.cond_source == "text":
        # 类名是 ClapEncoder（不是 CLAPEncoder）。写错时 ImportError 会立刻抛出、
        # 队列里这一步直接跳过，只有日志留痕——9/30 04:55 的 C_text 就是这么丢的。
        if args.cond_text_embedding:
            text_vec = np.load(args.cond_text_embedding, allow_pickle=False)
            if text_vec.shape != (512,) or not np.isfinite(text_vec).all() or np.linalg.norm(text_vec) == 0:
                raise ValueError("文本条件向量必须是有限、非零的512维数组")
        else:
            from encoder import ClapEncoder
            text_vec = ClapEncoder(ASSETS, "cpu").text([args.cond_text])[0]
        print(f"文本条件「{args.cond_text}」-> 512 维向量", flush=True)
    else:
        text_vec = None
    tr_cond = build_conditions(
        args.cond_source, vec_of, [d.name for d in tr_dirs], text_vec
    )
    te_cond = build_conditions(
        args.cond_source, vec_of, [d.name for d in te_dirs], text_vec
    )
    print(
        f"条件来源 {args.cond_source}｜训练 {tr_cond.shape}｜测试 {te_cond.shape}",
        flush=True,
    )
    val_samples = list(range(min(args.val_tracks, len(te_dirs))))

    tags = args.cond_targets if args.cond_mode != "none" else "none"
    tag = args.tag or f"{args.cond_mode}_{tags.replace('.', '').replace(',', '-')}"
    config = vars(args) | {
        "tag": tag,
        "train_tracks": len(tr_dirs),
        "cond_dim": int(sem.shape[1]),
        "inference_shifts": 1,
        "inference_seed": "test_track_index",
        "training_rng_reset_after_model_init": True,
        "training_precision": "fp32" if args.no_amp or args.device != "cuda" else "cuda_autocast_fp16",
        "text_embedding_sha256": hashlib.sha256(text_vec.tobytes()).hexdigest() if text_vec is not None else None,
    }

    with run_record(f"w3_conditioned_{tag}", config=config) as run:
        augmenter = (
            WaveformAugment(AUGMENT_PRESETS[args.augment])
            if args.augment != "none"
            else None
        )
        from demucs.pretrained import get_model

        bag = get_model("htdemucs")
        base = bag.models[0] if hasattr(bag, "models") else bag
        base.to(args.device).train()

        if args.cond_mode == "none":
            model = base
            targets = []
        else:
            layout = (
                "channel_last"
                if args.cond_targets.startswith("tencoder")
                else "channel_first"
            )
            targets = [
                (t.strip(), layout) for t in args.cond_targets.split(",") if t.strip()
            ]
            # 形状探测用的假输入必须和主干在同一个设备上：
            # 主干已经 .to(device) 了，假输入还在 CPU 会直接报
            # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"
            sample = torch.zeros(1, 2, int(args.segment * 44100), device=args.device)
            model = ConditionedModel(
                base,
                cond_dim=int(sem.shape[1]),
                targets=targets,
                sample_input=sample,
                mode="both" if args.cond_mode == "film" else "beta",
            )
            model.to(args.device)
        if args.freeze_backbone and targets:
            # 只训条件注入层；仍需匹配零/文本条件，不能将变化全归因于语义。
            for p in model.parameters():
                p.requires_grad = False
            for film in model.films.values():
                for p in film.parameters():
                    p.requires_grad = True
            print("已冻结主干，只训练 FiLM", flush=True)
        params = [p for p in model.parameters() if p.requires_grad]
        n_film = sum(p.numel() for p in model.film_parameters()) if targets else 0
        print(
            f"条件方式 {args.cond_mode}｜注入点 {targets}｜FiLM 参数 {n_film / 1e3:.1f}K "
            f"｜可训练 {sum(p.numel() for p in params) / 1e6:.1f}M",
            flush=True,
        )
        run.note(f"条件 {args.cond_mode}/{targets}，FiLM 参数 {n_film / 1e3:.1f}K")

        log_path = RESULTS / f"cond_train_{tag}.csv"

        if args.eval_only:
            # 只评测：加载存档权重，在**全部** 50 首测试曲目上逐首打分并落盘。
            # 原来只用 3 首，样本太小（导师指出过），重评不需要重训。
            state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            rows = []
            for i, d in enumerate(te_dirs):
                r = score_track(model, i, te_dirs, te_cond, args.device, bool(targets))
                rows.append({"track": d.name, **r})
                if (i + 1) % 10 == 0:
                    print(f"  评测 {i + 1}/{len(te_dirs)}", flush=True)
            df = pd.DataFrame(rows)
            out = RESULTS / f"cond_eval_{tag}.csv"
            df.to_csv(out, index=False)
            # 注意必须是 list：pandas 会把元组当成一个列名，直接 KeyError
            med = df[list(DISPLAY_STEMS)].median()
            print(
                "50 首中位数："
                + "  ".join(f"{s} {med[s]:.3f}" for s in DISPLAY_STEMS)
                + f"  | 平均 {med.mean():.3f}"
            )
            run.artifact(out)
            run.note(f"{tag} 全 50 首重评：四轨平均 {med.mean():.3f} dB")
            return 0

        before = validate(
            model, val_samples, te_dirs, te_cond, args.device, bool(targets)
        )
        print(
            "训练前 SI-SDR：" + "  ".join(f"{k} {v:.2f}" for k, v in before.items()),
            flush=True,
        )
        # FiLM初始化会消耗torch随机数；重置后各组采样与增强使用相同起点。
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        opt = torch.optim.Adam(params, lr=args.lr)
        use_amp = args.device == "cuda" and not args.no_amp
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        t0 = time.time()
        last = 0.0
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "elapsed_s"])
            for step in range(1, args.steps + 1):
                mix, tgt, cond = make_batch(
                    tr_dirs,
                    tr_cond,
                    int(args.segment * 44100),
                    args.batch_size,
                    augmenter,
                )
                mix, tgt = mix.to(args.device), tgt.to(args.device)
                ref = mix.mean(1, keepdim=True)
                mean, std = ref.mean(), ref.std() + 1e-8
                x = (mix - mean) / std
                if targets:
                    model.set_cond(cond.to(args.device))
                with torch.autocast("cuda", enabled=use_amp):
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
                    print(
                        f"  step {step}/{args.steps} loss {last:.4f} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )

        after = validate(
            model, val_samples, te_dirs, te_cond, args.device, bool(targets)
        )
        print(
            "训练后 SI-SDR：" + "  ".join(f"{k} {v:.2f}" for k, v in after.items()),
            flush=True,
        )
        delta = {k: after[k] - before[k] for k in DISPLAY_STEMS}
        b_mean, a_mean = (
            float(np.mean(list(before.values()))),
            float(np.mean(list(after.values()))),
        )
        print(
            "变化："
            + "  ".join(f"{k} {v:+.2f}" for k, v in delta.items())
            + f"  | 均值 {b_mean:.2f} → {a_mean:.2f} ({a_mean - b_mean:+.3f})",
            flush=True,
        )

        row = {
            "组": tag,
            "条件": args.cond_mode,
            "注入点": str(targets),
            "步数": args.steps,
            "训练前均值": round(b_mean, 3),
            "训练后均值": round(a_mean, 3),
            "变化": round(a_mean - b_mean, 3),
            "末步loss": round(last, 5),
            **{f"Δ{k}": round(delta[k], 3) for k in DISPLAY_STEMS},
            "FiLM参数K": round(n_film / 1e3, 1),
        }
        out_csv = RESULTS / "cond_separation_compare.csv"
        new = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

        ckpt = RESULTS / f"cond_model_{tag}.th"
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "stems": MODEL_STEMS},
            ckpt,
        )
        run.artifact(log_path, out_csv, ckpt)
        run.note(
            f"{tag}：训练前 {b_mean:.2f} → 训练后 {a_mean:.2f}（{a_mean - b_mean:+.3f} dB），"
            f"其他轨 {delta['other']:+.2f}"
        )
        print(f"\n汇总 -> {out_csv}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
