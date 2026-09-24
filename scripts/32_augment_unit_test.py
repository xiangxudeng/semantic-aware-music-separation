"""数据增强模块的单元测试（第 2 周 P0）。

逐条验证：

1. 形状：单条与 batch、mixture 与 targets 的布局都不变
2. 五种增强各自确实生效
3. **mixture == Σ targets 这条物理约束在增强后依然成立**（这是分离任务最关键的检查）
4. 概率控制：p=0 不动、force=True 必动
5. 配置：JSON 读入 / 预设 / dict 往返一致
6. 可复现：同一个 seed 两次结果一致
7. 批量接口与逐条调用的结果一致
8. 真实 MUSDB 音频上跑一遍（读到就用，读不到就跳过）

用法：
    python scripts/32_augment_unit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data.augment import AUGMENT_PRESETS, AugmentConfig, WaveformAugment

RESULTS: list[tuple[bool, str]] = []
NAMES = ("gain", "polarity", "band_stop", "time_crop", "channel_swap")


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"    {detail}" if detail else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def make_pair(seed: int = 0, ch: int = 2, n: int = 44100, sources: int = 4):
    """造一对满足 mixture == Σ targets 的假数据。"""
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(sources, ch, n, generator=g)
    mix = targets.sum(0)
    return mix, targets


def sum_gap(mix, targets) -> float:
    n = min(mix.shape[-1], targets.shape[-1])
    return float((mix[..., :n] - targets[..., :n].sum(-3)).abs().max())


def one_augmentation(name: str):
    cfg = AugmentConfig(enabled=True, seed=1)
    for other in NAMES:
        if other == name:
            continue
        setattr(cfg, f"{other}_p", 0.0)
    if name == "gain":
        cfg.gain_db = (-6.0, 6.0)
    return WaveformAugment(cfg)


def main() -> int:
    print("数据增强模块单元测试", flush=True)
    mix, targets = make_pair()

    section("1. 形状")
    aug = WaveformAugment(AUGMENT_PRESETS["default"])
    m, t = aug(mix, targets)
    check(m.shape == mix.shape and t.shape == targets.shape,
          "单条 (ch,T)/(S,ch,T) 形状不变", f"{tuple(m.shape)} / {tuple(t.shape)}")
    bm = mix[None].repeat(3, 1, 1)
    bt = targets[None].repeat(3, 1, 1, 1)
    bm2, bt2 = aug.apply_batch(bm, bt)
    check(bm2.shape == bm.shape and bt2.shape == bt.shape,
          "batch (B,ch,T)/(B,S,ch,T) 形状不变", f"{tuple(bm2.shape)} / {tuple(bt2.shape)}")

    section("2. 五种增强各自生效")
    for name in NAMES:
        a = one_augmentation(name)
        m2, t2 = a(mix, targets, force=True)
        changed_m = not torch.allclose(m2, mix)
        changed_t = not torch.allclose(t2, targets)
        check(changed_m and changed_t, f"{name}：mixture 与 targets 都被改变",
              f"max|Δmix|={float((m2 - mix).abs().max()):.3e}")

    section("3. 物理约束 mixture == Σ targets 在增强后仍成立")
    check(sum_gap(mix, targets) < 1e-5, "增强前恒等式成立", f"max|Δ|={sum_gap(mix, targets):.2e}")
    for name in NAMES:
        a = one_augmentation(name)
        m2, t2 = a(mix, targets, force=True)
        gap = sum_gap(m2, t2)
        check(gap < 1e-3, f"{name}：增强后恒等式成立", f"max|Δ|={gap:.2e}")

    section("4. 概率控制")
    off = WaveformAugment(AUGMENT_PRESETS["off"])
    m3, t3 = off(mix, targets)
    check(torch.equal(m3, mix) and torch.equal(t3, targets), "enabled=false 时完全不动")
    never = WaveformAugment(
        AugmentConfig(gain_p=0, polarity_p=0, band_stop_p=0, time_crop_p=0, channel_swap_p=0)
    )
    m4, _ = never(mix, targets)
    check(torch.equal(m4, mix), "所有概率设 0 时完全不动")

    section("5. 配置")
    cfg_file = ROOT / "configs" / "augment_default.json"
    cfg = AugmentConfig.from_file(cfg_file)
    check(cfg.gain_db == (-6.0, 6.0) and cfg.gain_p == 0.5, "从 JSON 读配置成功", str(cfg.gain_db))
    check(AugmentConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict(), "dict 往返一致")
    check(len(AUGMENT_PRESETS) >= 5, "内置预设齐全", ", ".join(AUGMENT_PRESETS))
    try:
        import yaml  # noqa: F401

        has_yaml = True
    except ImportError:
        has_yaml = False
    print(f"    （pyyaml 可用：{has_yaml}，不可用时用 JSON 配置）", flush=True)

    section("6. 可复现")
    a1 = WaveformAugment(AugmentConfig(seed=7))
    a2 = WaveformAugment(AugmentConfig(seed=7))
    m5, t5 = a1(mix, targets)
    m6, t6 = a2(mix, targets)
    check(torch.equal(m5, m6) and torch.equal(t5, t6), "相同 seed 两次结果完全一致")
    a3 = WaveformAugment(AugmentConfig(seed=8))
    m7, _ = a3(mix, targets)
    check(not torch.allclose(m5, m7), "不同 seed 结果不同",
          f"max|Δ|={float((m5 - m7).abs().max()):.3e}")

    section("7. 批量与逐条一致")
    a4 = WaveformAugment(AugmentConfig(seed=3))
    bm3, bt3 = a4.apply_batch(bm, bt)
    a5 = WaveformAugment(AugmentConfig(seed=3))
    rows_m, rows_t = [], []
    for i in range(bm.shape[0]):
        mm, tt = a5(bm[i], bt[i])
        rows_m.append(mm)
        rows_t.append(tt)
    check(torch.allclose(bm3, torch.stack(rows_m)) and torch.allclose(bt3, torch.stack(rows_t)),
          "apply_batch == 逐条调用")

    section("8. 真实 MUSDB 音频")
    try:
        import soundfile as sf

        test_dir = ROOT / "data" / "musdb18hq_wav" / "test"
        track = sorted(d for d in test_dir.iterdir() if d.is_dir())[0]
        n = 44100 * 8
        mix_r, _ = sf.read(str(track / "mixture.wav"), dtype="float32", always_2d=True)
        stems = [
            sf.read(str(track / f"{s}.wav"), dtype="float32", always_2d=True)[0]
            for s in ("vocals", "drums", "bass", "other")
        ]
        mix_t = torch.from_numpy(mix_r[:n].T)
        tgt_t = torch.stack([torch.from_numpy(s[:n].T) for s in stems])
        gap0 = sum_gap(mix_t, tgt_t)
        check(gap0 < 1e-4, f"真实曲目（{track.name[:24]}，8 秒）mixture==Σstems",
              f"max|Δ|={gap0:.2e}")
        a6 = WaveformAugment(AUGMENT_PRESETS["default"])
        m8, t8 = a6(mix_t, tgt_t)
        check(m8.shape == mix_t.shape and t8.shape == tgt_t.shape, "真实音频增强后形状不变")
        gap1 = sum_gap(m8, t8)
        check(gap1 < 1e-3, "真实音频增强后恒等式仍成立", f"max|Δ|={gap1:.2e}")
        print("    " + a6.describe().replace("\n", "\n    "), flush=True)
        print(f"    本轮触发次数：{a6.applied}", flush=True)
    except FileNotFoundError as exc:
        print(f"    （跳过：{exc}）", flush=True)

    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{'=' * 60}")
    print(f"共 {len(RESULTS)} 项检查，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项")
    for label in failed:
        print(f"  失败：{label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
