"""把 htdemucs 的真实结构导出来，作为架构拆解文档的数据来源（第 2 周 P1）。

文档里的每个数字都应该能追溯到某个脚本，这个脚本负责产出：

1. 每个子模块的参数量与占比
2. 前向过程中关键节点的张量形状（用假数据跑一遍、挂 forward hook 记录）
3. FiLM 可注入点的通道数

用法：
    python scripts/34_demucs_arch_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def human(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.1f} K"


def main() -> int:
    from demucs.pretrained import get_model

    bag = get_model("htdemucs")
    model = bag.models[0] if hasattr(bag, "models") else bag
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"模型：htdemucs（单模型，共 {human(total)} 参数）")
    print(f"音源数 {len(model.sources)}：{list(model.sources)}")
    print(f"采样率 {model.samplerate}，segment {float(model.segment):.1f} 秒")
    print()

    print("=== 顶层子模块参数量 ===")
    top = [(n, m) for n, m in model.named_children()]
    rows = []
    for name, mod in top:
        n = sum(p.numel() for p in mod.parameters())
        rows.append((name, type(mod).__name__, n, n / total * 100))
    for name, cls, n, pct in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:22} {cls:28} {human(n):>10}  {pct:5.1f}%")

    print("\n=== 分支层级结构 ===")
    for branch in ("encoder", "tencoder", "decoder", "tdecoder"):
        mod = getattr(model, branch, None)
        if mod is None:
            continue
        sub = list(mod.named_children()) if len(list(mod.named_children())) else [("", mod)]
        print(f"  {branch}：{len(sub)} 层")
        for i, (n, m) in enumerate(mod.named_children()):
            p = sum(x.numel() for x in m.parameters())
            print(f"    [{i}] {type(m).__name__:26} {human(p):>10}")

    # ---- 前向形状记录 ----
    print("\n=== 前向关键节点形状（假数据 1×2×88200，即 2 秒 44.1kHz 立体声）===")
    mix = torch.randn(1, 2, 88200)
    watch = ["encoder.0", "encoder.1", "encoder.2", "encoder.3",
             "tencoder.0", "tencoder.1", "tencoder.2", "tencoder.3",
             "crosstransformer", "decoder.0", "decoder.3", "tdecoder.3"]
    modules = dict(model.named_modules())
    shapes: dict[str, str] = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                shapes[name] = str(tuple(t.shape))
        return hook

    for name in watch:
        if name in modules:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
    with torch.no_grad():
        out = model(mix)
    for h in handles:
        h.remove()

    for name in watch:
        if name in shapes:
            print(f"  {name:22} -> {shapes[name]}")
    print(f"  {'最终输出':22} -> {tuple(out.shape)}   （B, 音源数, 通道, 采样点）")

    print("\n=== FiLM 可注入点（第 3 周用）===")
    for name, shape in shapes.items():
        parts = shape.strip("()").split(",")
        if not name.startswith(("encoder", "decoder")) or len(parts) != 4:
            continue
        print(f"  {name:22} 通道数 {parts[1].strip():>4}   张量 {shape}")

    print("\n=== 推理开销（本机 CPU，2 秒音频）===")
    import time

    with torch.no_grad():
        model(torch.randn(1, 2, 44100))
        t0 = time.time()
        for _ in range(3):
            model(torch.randn(1, 2, 44100))
        dt = (time.time() - t0) / 3
    print(f"  平均 {dt:.2f} 秒 / 2 秒片段（CPU，仅供量级参考；GPU 上实测见报告）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
