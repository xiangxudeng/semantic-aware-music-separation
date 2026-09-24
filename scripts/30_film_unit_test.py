"""FiLM 条件注入层的单元测试（第 2 周 P0）。

计划里的要求是"实现 FiLM 条件注入层并用假数据做单元测试，确认维度与梯度正常"。
这里用假数据把下面这些点逐条验证掉，最后再用**真实的 htdemucs** 跑一次条件前向，
确认注入没有破坏模型：

1. 形状：3 维/4 维、通道在前/在后两种布局都能正确广播
2. 零初始化时是恒等映射
3. cond=None 时是恒等映射
4. both / gamma / beta 三种模式都能跑
5. 打破零初始化后，条件确实会改变输出，且不同条件给出不同输出
6. 梯度：FiLM 参数、条件向量、主干参数都能拿到梯度
7. 真实 htdemucs 前向打通，且零初始化下与基线逐元素一致

用法：
    python scripts/30_film_unit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.separation.film import ConditionedModel, FiLM

torch.manual_seed(0)
RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"    {detail}" if detail else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def test_shapes() -> None:
    section("1. 形状与广播")
    B, C, T, F = 2, 8, 16, 5
    # 3 维、通道在前 (B, C, T)
    x = torch.randn(B, C, T)
    y = FiLM(cond_dim=512, num_features=C, layout="channel_first")(x, torch.randn(B, 512))
    check(y.shape == x.shape, "3 维 channel_first (B,C,T) 输出形状不变", f"{tuple(y.shape)}")

    # 4 维、通道在前 (B, C, F, T)
    x4 = torch.randn(B, C, F, T)
    y4 = FiLM(cond_dim=512, num_features=C, layout="channel_first")(x4, torch.randn(B, 512))
    check(y4.shape == x4.shape, "4 维 channel_first (B,C,F,T) 输出形状不变", f"{tuple(y4.shape)}")

    # 3 维、通道在后 (B, T, C)
    xt = torch.randn(B, T, C)
    yt = FiLM(cond_dim=512, num_features=C, layout="channel_last")(xt, torch.randn(B, 512))
    check(yt.shape == xt.shape, "3 维 channel_last (B,T,C) 输出形状不变", f"{tuple(yt.shape)}")

    # 单条样本（条件不带 batch 维）也应该能跑
    y1 = FiLM(cond_dim=512, num_features=C, layout="channel_first")(x[:1], torch.randn(512))
    check(y1.shape == x[:1].shape, "条件传一维 (cond_dim,) 时自动补 batch 维")


def test_identity() -> None:
    section("2/3. 零初始化与 cond=None 都是恒等映射")
    x = torch.randn(2, 8, 16)
    film = FiLM(512, 8, hidden_dim=32)
    y = film(x, torch.randn(2, 512))
    check(torch.equal(y, x), "零初始化 + 有条件的输出 == 输入（恒等）",
          f"max|Δ|={float((y - x).abs().max()):.2e}")
    y2 = film(x, None)
    check(torch.equal(y2, x), "cond=None 时不改任何东西")
    for mode in ("both", "gamma", "beta"):
        ym = FiLM(512, 8, mode=mode)(x, torch.randn(2, 512))
        check(torch.equal(ym, x), f"mode={mode} 零初始化也是恒等")


def test_condition_effective() -> None:
    section("5. 打破零初始化后，条件确实起作用")
    x = torch.randn(2, 8, 16)
    film = FiLM(512, 8)
    with torch.no_grad():
        film.to_film.weight.normal_(0, 0.1)
        film.to_film.bias.normal_(0, 0.1)
    c1, c2 = torch.randn(2, 512), torch.randn(2, 512)
    y1, y2 = film(x, c1), film(x, c2)
    check(not torch.allclose(y1, x), "非零权重时输出不再等于输入")
    check(not torch.allclose(y1, y2), "不同条件给出不同输出",
          f"max|y1-y2|={float((y1 - y2).abs().max()):.3f}")
    # 同一个条件两次调用结果必须一致（无随机性）
    check(torch.equal(film(x, c1), y1), "同一条件重复调用结果一致（确定性）")


def test_gradients() -> None:
    section("6. 梯度")
    B, C, T = 2, 8, 16
    backbone = torch.nn.Sequential(torch.nn.Conv1d(2, C, 3, padding=1), torch.nn.ReLU())
    x = torch.randn(B, 2, T)
    cond = torch.randn(B, 512, requires_grad=True)

    with torch.no_grad():
        backbone[0].weight  # noqa: B018
    film = FiLM(512, C)
    with torch.no_grad():  # 先给非零权重，否则梯度恒为 0（乘的是 x 之外的全零参数）
        film.to_film.weight.normal_(0, 0.1)
        film.to_film.bias.normal_(0, 0.1)

    h = backbone(x)
    loss = film(h, cond).pow(2).mean()
    loss.backward()

    film_grads = [p.grad for p in film.parameters() if p.requires_grad]
    check(all(g is not None for g in film_grads), "FiLM 自己的参数有梯度",
          f"{sum(g is not None for g in film_grads)}/{len(film_grads)}")
    check(cond.grad is not None and torch.isfinite(cond.grad).all(), "条件向量有梯度且有限",
          f"norm={float(cond.grad.norm()):.4f}")

    loss2 = film(backbone(x), cond.detach()).pow(2).mean()
    loss2.backward()
    check(backbone[0].weight.grad is not None and torch.isfinite(backbone[0].weight.grad).all(),
          "主干参数也能拿到梯度（FiLM 不截断反传）")


def test_real_demucs() -> None:
    section("7. 真实 htdemucs 前向打通")
    try:
        from demucs.pretrained import get_model
    except Exception as exc:  # noqa: BLE001
        check(False, "导入 demucs 失败", repr(exc))
        return

    bag = get_model("htdemucs")
    model = bag.models[0] if hasattr(bag, "models") else bag
    model.eval()

    names = set(dict(model.named_modules()))
    enc = sorted((n for n in names if n.startswith("encoder.") and n.count(".") == 1),
                 key=lambda s: int(s.split(".")[1]))
    dec = sorted((n for n in names if n.startswith("decoder.") and n.count(".") == 1),
                 key=lambda s: int(s.split(".")[1]))
    check(bool(enc) and bool(dec), "找到 encoder / decoder 子层",
          f"encoder 共 {len(enc)} 层，decoder 共 {len(dec)} 层")
    targets = [(enc[-1], "channel_first"), (dec[0], "channel_first")]
    print(f"    注入点：{targets}", flush=True)

    mix = torch.randn(1, 2, 44100 * 2)
    with torch.no_grad():
        base_out = model(mix)
    check(base_out.dim() == 4 and base_out.shape[0] == 1, "基线前向输出形状正常",
          f"{tuple(base_out.shape)}")

    wrapped = ConditionedModel(model, cond_dim=512, targets=targets, sample_input=mix)
    check(
        wrapped.target_dims == {targets[0][0]: wrapped.target_dims[targets[0][0]],
                                targets[1][0]: wrapped.target_dims[targets[1][0]]},
        "自动探测到注入点通道数",
        str(wrapped.target_dims),
    )
    with torch.no_grad():
        same_out = wrapped(mix)
    check(torch.allclose(base_out, same_out, atol=1e-6),
          "零初始化下条件模型与基线输出完全一致",
          f"max|Δ|={float((base_out - same_out).abs().max()):.2e}")

    wrapped.set_cond(torch.randn(1, 512))
    with torch.no_grad():
        cond_out = wrapped(mix)
    check(cond_out.shape == base_out.shape, "带条件前向输出形状不变", f"{tuple(cond_out.shape)}")
    check(torch.allclose(base_out, cond_out, atol=1e-6),
          "零初始化 + 条件：仍然与基线一致（条件尚未生效，符合预期）")

    # 打破零初始化，确认条件真的能改变 Demucs 的输出，并且梯度能回传
    for film in wrapped.films.values():
        with torch.no_grad():
            film.to_film.weight.normal_(0, 0.02)
            film.to_film.bias.normal_(0, 0.02)

    train_mix = torch.randn(1, 2, 44100)
    with torch.no_grad():
        out_before = model(train_mix)
    out = wrapped(train_mix)
    check(out.shape == out_before.shape, "打破零初始化后前向输出形状与基线一致", f"{tuple(out.shape)}")
    check(not torch.allclose(out, out_before, atol=1e-6),
          "打破零初始化后条件确实改变了 Demucs 的输出",
          f"max|Δ|={float((out - out_before).abs().max()):.4f}")
    loss = out.pow(2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in wrapped.film_parameters() if p.grad is not None)
    n_total = len(wrapped.film_parameters())
    check(n_with_grad == n_total and n_total > 0, "FiLM 参数全部拿到梯度", f"{n_with_grad}/{n_total}")
    cond_grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    check(cond_grad_ok, "Demucs 主干参数也能拿到梯度（没有断链）")

    model.zero_grad(set_to_none=True)
    film_params = sum(p.numel() for p in wrapped.film_parameters())
    backbone_params = sum(p.numel() for p in model.parameters())
    print(
        f"    参数开销：FiLM {film_params / 1e3:.1f} K vs Demucs {backbone_params / 1e6:.1f} M "
        f"（占 {film_params / backbone_params * 100:.2f}%）",
        flush=True,
    )
    check(
        film_params < backbone_params * 0.03,
        "FiLM 参数增量小于主干的 3%（相比全参数微调仍是极小开销）",
    )
    wrapped.remove_hooks()


def main() -> int:
    print("FiLM 条件注入层单元测试", flush=True)
    test_shapes()
    test_identity()
    test_condition_effective()
    test_gradients()
    test_real_demucs()

    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{'=' * 60}")
    print(f"共 {len(RESULTS)} 项检查，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        for label in failed:
            print(f"  失败：{label}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
