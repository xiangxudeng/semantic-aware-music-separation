"""第 3 周开工自检：确认所有前置条件就绪，并把环境固化到运行记录里。

第 3 周要在"分离"和"语义"之间做真正的连接，涉及的东西比前两周都多
（Demucs + CLAP + FiLM + LoRA/量化），所以开工前先把前提逐条验一遍，
避免跑到一半才发现某个模块在 GPU 上不可用。

用法：
    python scripts/50_w3_kickoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.utils.run_log import data_fingerprint, run_record

CHECKS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    CHECKS.append((bool(ok), label, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"    {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    with run_record("w3_kickoff", config={}) as run:
        print("=== 1. 环境 ===", flush=True)
        gpu = f"，{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "（第 3 周的 LoRA 需要 GPU）"
        check(torch.__version__.startswith("2."), f"torch {torch.__version__}",
              f"CUDA 可用={torch.cuda.is_available()}{gpu}")

        print("\n=== 2. 第 2 周的模块 ===", flush=True)
        try:
            from src.separation.film import FiLM  # noqa: F401

            f = FiLM(cond_dim=512, num_features=8)
            x = torch.randn(2, 8, 16)
            check(torch.equal(f(x, torch.randn(2, 512)), x), "FiLM 零初始化=恒等映射")
        except Exception as exc:  # noqa: BLE001
            check(False, "FiLM 模块", repr(exc))
        try:
            from src.semantic.clap_encoder import ClapEncoder  # noqa: F401

            check(True, "CLAP 语义编码器可导入")
        except Exception as exc:  # noqa: BLE001
            check(False, "CLAP 语义编码器", repr(exc))
        try:
            from src.data.augment import AUGMENT_PRESETS, WaveformAugment  # noqa: F401

            check("default" in AUGMENT_PRESETS, "数据增强模块可导入")
        except Exception as exc:  # noqa: BLE001
            check(False, "数据增强模块", repr(exc))
        try:
            from src.semantic.tagging_metrics import evaluate_scores  # noqa: F401

            check(True, "标签指标模块（含召回率口径）可导入")
        except Exception as exc:  # noqa: BLE001
            check(False, "标签指标模块", repr(exc))

        print("\n=== 3. 数据 ===", flush=True)
        fp = data_fingerprint()
        for k, v in fp.items():
            print(f"    {k:26} {v}")
        check(fp.get("musdb_train_tracks", 0) == 100, "MUSDB 训练集 100 首", str(fp.get("musdb_train_tracks")))
        check(fp.get("musdb_test_tracks", 0) == 50, "MUSDB 测试集 50 首", str(fp.get("musdb_test_tracks")))
        check(fp.get("mtt_50tags_hash") is not None, "50 类标签清单就绪")

        print("\n=== 4. 条件分离前置 ===", flush=True)
        try:
            from demucs.pretrained import get_model

            from src.separation.film import ConditionedModel

            bag = get_model("htdemucs")
            model = bag.models[0] if hasattr(bag, "models") else bag
            model.eval()
            mix = torch.randn(1, 2, 44100 * 2)
            with torch.no_grad():
                base_out = model(mix)
            wrapped = ConditionedModel(
                model, cond_dim=512,
                targets=[("encoder.3", "channel_first"), ("decoder.0", "channel_first")],
                sample_input=mix,
            )
            wrapped.set_cond(torch.randn(1, 512))
            with torch.no_grad():
                cond_out = wrapped(mix)
            check(torch.allclose(base_out, cond_out, atol=1e-6),
                  "条件注入后前向与基线一致（零初始化）",
                  f"max|Δ|={float((base_out - cond_out).abs().max()):.2e}")
            check(cond_out.shape == base_out.shape, "条件前向输出形状正确", str(tuple(cond_out.shape)))
            run.note(f"FiLM 可注入点通道数：{wrapped.target_dims}")
            wrapped.remove_hooks()
        except Exception as exc:  # noqa: BLE001
            check(False, "条件分离前置检查", repr(exc))

        failed = [c for c in CHECKS if not c[0]]
        print(f"\n{'=' * 60}")
        print(f"共 {len(CHECKS)} 项，通过 {len(CHECKS) - len(failed)}，失败 {len(failed)}")
        run.note(f"开工自检 {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过；数据 {fp}")
        if failed:
            for _, label, detail in failed:
                print(f"  失败：{label} {detail}")
            return 1
        print("第 3 周前置条件全部就绪")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
