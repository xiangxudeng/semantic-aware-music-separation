"""FiLM 条件注入层（第 2 周 P0 交付物）。

FiLM = Feature-wise Linear Modulation：用一个条件向量 c 生成一组**逐通道**的
缩放 gamma 与平移 beta，作用在特征图上：

    h' = (1 + gamma) * h + beta

在本项目里的位置：第 3 周要做"语义条件分离"——把 CLAP 语义向量（512 维）
注入 Demucs，让它按风格/乐器条件去分离。注入方式的第一版就是 FiLM：
只加乘法和加法，不改 Demucs 的拓扑，参数增量很小。

两个关键设计
------------
1. **零初始化**：生成 gamma/beta 的最后一层权重与偏置全部置零，于是训练开始时
   (1 + gamma) = 1、beta = 0，整个模块是恒等映射——条件模型与基线模型的输出
   逐元素完全相同。如果不做零初始化，随机的 gamma/beta 会先把基线结果扰坏，
   实验里就分不清涨跌来自"条件信息"还是"初始化噪声"。
2. **条件可选**：`forward(x, cond=None)` 时原样返回，`set_cond()` 之后才生效。
   同一份模型既能跑基线也能跑条件版本，不用维护两套代码。

注入方式用的是 PyTorch 的 forward hook：拿到某个子模块的输出张量后就地做 FiLM。
好处是不需要改 Demucs 的 forward，也不用去替换它的 ModuleList。
"""

from __future__ import annotations

import torch
from torch import nn


def _pick_channels(x: torch.Tensor, layout: str) -> int:
    if layout == "channel_first":
        if x.dim() < 2:
            raise ValueError(f"channel_first 需要至少 2 维张量，收到 {tuple(x.shape)}")
        return x.shape[1]
    if x.dim() < 2:
        raise ValueError(f"channel_last 需要至少 2 维张量，收到 {tuple(x.shape)}")
    return x.shape[-1]


class FiLM(nn.Module):
    """把一个条件向量变成逐通道的 gamma/beta。

    Parameters
    ----------
    cond_dim : 条件向量维度（CLAP 是 512）
    num_features : 被调制特征图的通道数
    hidden_dim : 大于 0 时在中间加一层 SiLU 隐层
    mode : "both" 同时给 gamma 和 beta；"gamma" / "beta" 只给其中一个
    layout : "channel_first" 表示通道在 dim=1（卷积特征图 (B,C,F,T)），
             "channel_last" 表示通道在最后一维（Transformer 的 (B,T,C)）
    """

    def __init__(
        self,
        cond_dim: int,
        num_features: int,
        hidden_dim: int = 0,
        mode: str = "both",
        layout: str = "channel_first",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("both", "gamma", "beta"):
            raise ValueError(f"mode 只能是 both/gamma/beta，收到 {mode}")
        if layout not in ("channel_first", "channel_last"):
            raise ValueError(f"layout 只能是 channel_first/channel_last，收到 {layout}")
        self.cond_dim = cond_dim
        self.num_features = num_features
        self.mode = mode
        self.layout = layout

        self.norm = nn.LayerNorm(cond_dim) if use_layernorm else nn.Identity()
        out_dim = num_features * (2 if mode == "both" else 1)
        if hidden_dim > 0:
            self.to_film = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.to_film = nn.Linear(cond_dim, out_dim)

        # 零初始化：模块初始为恒等映射，见模块文档
        last = self.to_film[-1] if hidden_dim > 0 else self.to_film
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        self._cond: torch.Tensor | None = None

    def set_cond(self, cond: torch.Tensor | None) -> None:
        """挂上条件向量；传 None 表示清除，回到无条件（基线）行为。"""
        self._cond = cond

    def _broadcast(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """把 (B, C) 的调制参数扩展成能和 x 广播的形状。"""
        if self.layout == "channel_first":
            return p.reshape(p.shape[0], -1, *([1] * (x.dim() - 2)))
        return p.reshape(p.shape[0], *([1] * (x.dim() - 2)), -1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        cond = cond if cond is not None else self._cond
        if cond is None:
            return x
        if cond.dim() == 1:
            cond = cond[None]
        # 条件张量可能来自 CPU（例如验证时直接拿 numpy 转的），这里自动跟随特征图的设备，
        # 否则 LayerNorm 会报 "found at least two devices, cpu and cuda:0"
        if cond.device != x.device:
            cond = cond.to(x.device)
        params = self.to_film(self.norm(cond))
        if self.mode == "both":
            gamma, beta = params.chunk(2, dim=-1)
        elif self.mode == "gamma":
            gamma, beta = params, None
        else:
            gamma, beta = None, params

        if gamma is not None:
            g = self._broadcast(gamma, x)
            x = x * (1.0 + g.to(dtype=x.dtype))
        if beta is not None:
            b = self._broadcast(beta, x)
            x = x + b.to(dtype=x.dtype)
        return x

    def extra_repr(self) -> str:
        return (
            f"cond_dim={self.cond_dim}, num_features={self.num_features}, "
            f"mode={self.mode}, layout={self.layout}"
        )


def nth_output(output):
    """取子模块输出里的张量：有的模块返回 (tensor, extra)。"""
    if isinstance(output, tuple):
        return output[0]
    return output


def attach_film(module: nn.Module, film: FiLM):
    """给某个子模块挂一个 forward hook，对它输出的张量做 FiLM。

    返回 hook handle，调用 `.remove()` 即可撤销注入。
    """

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (film(output[0]), *output[1:])
        return film(output)

    return module.register_forward_hook(hook)


def infer_channel_dims(model: nn.Module, names: list[str], sample_input, layout: str) -> dict[str, int]:
    """跑一次假数据前向，把每个注入点的特征通道数读出来，省得手工查结构。

    Demucs 各层通道数藏在配置里、还会随模型版本变，直接读比写死可靠。
    """
    modules = dict(model.named_modules())
    missing = [n for n in names if n not in modules]
    if missing:
        raise KeyError(f"模型里没有这些子模块：{missing}")

    found: dict[str, int] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            found[name] = _pick_channels(nth_output(output), layout)

        return hook

    for name in names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(f"这些子模块在前向中没有被调用到：{missing}")
    return found


class ConditionedModel(nn.Module):
    """把 FiLM 挂到既有模型上的包装器。

    targets 的每一项是 (子模块名, layout)。通道数自动探测。
    调用 `set_cond(c)` 之后，模型前向就是条件版本；`set_cond(None)` 回到基线行为。
    """

    def __init__(
        self,
        backbone: nn.Module,
        cond_dim: int,
        targets: list[tuple[str, str]],
        sample_input,
        hidden_dim: int = 0,
        mode: str = "both",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cond_dim = cond_dim
        self.targets = list(targets)
        dims = infer_channel_dims(
            backbone, [n for n, _ in targets], sample_input, targets[0][1]
        )
        self.films = nn.ModuleDict()
        modules = dict(backbone.named_modules())
        self._handles = []
        for name, layout in targets:
            film = FiLM(cond_dim, dims[name], hidden_dim=hidden_dim, mode=mode, layout=layout)
            self.films[name.replace(".", "_")] = film
            self._handles.append(attach_film(modules[name], film))
        self.target_dims = dims

    def set_cond(self, cond: torch.Tensor | None) -> None:
        for film in self.films.values():
            film.set_cond(cond)

    def __getattr__(self, name: str):
        """把没定义的属性透传给主干。

        demucs 的 apply_model 会读 model.samplerate / model.sources 等属性，
        包装器必须把这些转给 backbone，否则推理直接报 AttributeError。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            backbone = self.__dict__.get("_modules", {}).get("backbone")
            if backbone is not None:
                return getattr(backbone, name)
            raise

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def film_parameters(self):
        return [p for film in self.films.values() for p in film.parameters()]

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
