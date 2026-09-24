"""音频数据增强模块（第 2 周 P0 交付物）。

五种增强，全部作用在**波形**上：

| 名称 | 做什么 | 直观作用 |
|---|---|---|
| `gain` 随机增益 | 整体乘一个随机增益 | 模型不该依赖绝对音量 |
| `polarity` 极性反转 | 整体乘 −1 | 反相不该改变分离结果 |
| `band_stop` 频带掩蔽 | 用理想带阻把某段频率置零 | 少一段频带也要能分 |
| `time_crop` 时间裁剪 | 随机保留一段、其余置零 | 局部缺失不该崩 |
| `channel_swap` 通道互换 | 左右声道对调 | 模型不该依赖声道位置 |

三条设计原则
------------
1. **成对处理**：增强同时作用在 mixture 与 targets 上。分离任务的目标是"从混合里
   还原各轨"，只改混合不改目标等于喂错标签。
2. **保持线性关系**：五种增强都是同一个线性/掩蔽算子作用在所有轨上，因此
   `mixture == Σ targets` 这条物理约束在增强后依然成立（单元测试里逐条验证）。
3. **可配置**：每一项都能单独开关、调参，可从 dict / JSON / YAML 载入，
   方便第 3 周做"哪种增强有用"的消融实验。

张量约定
--------
沿用 Demucs 训练脚本的布局，**倒数第二维是通道、最后一维是时间**：

    mixture: (ch, T)      或 (B, ch, T)
    targets: (S, ch, T)   或 (B, S, ch, T)

这样单条样本和 batch 走同一份代码。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

CH = -2  # 通道维
TIME = -1  # 时间维


@dataclass
class AugmentConfig:
    """每种增强的开关与参数。默认值偏保守，来自 Demucs 官方训练配置的习惯量级。"""

    gain_db: tuple[float, float] | None = (-6.0, 6.0)
    gain_p: float = 0.5

    polarity_p: float = 0.5

    band_stop_p: float = 0.3
    band_stop_min: float = 0.05  # 带宽占 Nyquist 的最小比例
    band_stop_max: float = 0.25

    time_crop_p: float = 0.3
    time_crop_min_keep: float = 0.5  # 至少保留多长

    channel_swap_p: float = 0.5

    seed: int | None = 0
    enabled: bool = True
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AugmentConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["gain_db"] = tuple(kwargs["gain_db"]) if kwargs.get("gain_db") else None
        extra = {k: v for k, v in data.items() if k not in known}
        obj = cls(**kwargs)
        obj._extra = extra
        return obj

    @classmethod
    def from_file(cls, path: str | Path) -> "AugmentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # noqa: BLE001
                raise RuntimeError("读 YAML 配置需要 pyyaml，或改用 .json") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data.get("augment", data) if isinstance(data, dict) else None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_extra", None)
        d["gain_db"] = list(d["gain_db"]) if d["gain_db"] else None
        return d


class WaveformAugment:
    """波形增强器。可调用对象，输入输出都是 (mixture, targets)。"""

    def __init__(self, config: AugmentConfig | dict | None = None) -> None:
        if isinstance(config, dict):
            config = AugmentConfig.from_dict(config)
        self.config = config or AugmentConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.applied: dict[str, int] = {}

    # ---------- 单个增强 ----------

    def _gain(self, mix, targets, p):
        if p <= 0 or self.config.gain_db is None or self.rng.random() >= p:
            return mix, targets
        lo, hi = self.config.gain_db
        db = float(self.rng.uniform(lo, hi))
        factor = 10.0 ** (db / 20.0)
        return mix * factor, targets * factor

    def _polarity(self, mix, targets, p):
        if p <= 0 or self.rng.random() >= p:
            return mix, targets
        return -mix, -targets

    def _band_stop(self, mix, targets, p):
        if p <= 0 or self.rng.random() >= p:
            return mix, targets
        n = mix.shape[TIME]
        if n < 64:
            return mix, targets
        half = n // 2 + 1
        lo_ratio = float(self.rng.uniform(self.config.band_stop_min, self.config.band_stop_max))
        width_ratio = float(
            self.rng.uniform(self.config.band_stop_min, max(self.config.band_stop_min, self.config.band_stop_max))
        )
        lo = int(lo_ratio * half)
        hi = min(half, lo + max(2, int(width_ratio * half)))

        def apply(x: torch.Tensor) -> torch.Tensor:
            spec = torch.fft.rfft(x, dim=TIME)
            spec[..., lo:hi] = 0
            return torch.fft.irfft(spec, n=n, dim=TIME)

        return apply(mix), apply(targets)

    def _time_crop(self, mix, targets, p):
        if p <= 0 or self.rng.random() >= p:
            return mix, targets
        n = mix.shape[TIME]
        if n < 64:
            return mix, targets
        keep = int(self.rng.uniform(self.config.time_crop_min_keep, 1.0) * n)
        keep = max(1, min(n, keep))
        start = int(self.rng.integers(0, n - keep + 1))
        mask = torch.zeros(n, dtype=mix.dtype, device=mix.device)
        mask[start : start + keep] = 1.0
        return mix * mask, targets * mask

    def _channel_swap(self, mix, targets, p):
        if p <= 0 or self.rng.random() >= p:
            return mix, targets
        if mix.shape[CH] < 2:
            return mix, targets
        return torch.flip(mix, dims=[CH]), torch.flip(targets, dims=[CH])

    # ---------- 对外接口 ----------

    def __call__(self, mix: torch.Tensor, targets: torch.Tensor, force: bool = False):
        """对一对 (mixture, targets) 做增强。

        force=True 时忽略各增强的概率，五种全部生效（单元测试用）。
        """
        if not self.config.enabled and not force:
            return mix, targets
        c = self.config
        steps = [
            ("gain", self._gain, c.gain_p),
            ("polarity", self._polarity, c.polarity_p),
            ("band_stop", self._band_stop, c.band_stop_p),
            ("time_crop", self._time_crop, c.time_crop_p),
            ("channel_swap", self._channel_swap, c.channel_swap_p),
        ]
        for name, fn, p in steps:
            before = mix
            mix, targets = fn(mix, targets, 1.0 if force else p)
            if mix is not before:
                self.applied[name] = self.applied.get(name, 0) + 1
        return mix, targets

    def apply_batch(self, mix: torch.Tensor, targets: torch.Tensor):
        """对 batch 逐条增强。mix (B,ch,T)，targets (B,S,ch,T)。"""
        mixes, tgts = [], []
        for i in range(mix.shape[0]):
            m, t = self(mix[i], targets[i])
            mixes.append(m)
            tgts.append(t)
        return torch.stack(mixes), torch.stack(tgts)

    def describe(self) -> str:
        c = self.config
        lines = [f"波形增强（enabled={c.enabled}, seed={c.seed}）"]
        lines.append(f"  随机增益      p={c.gain_p}  范围 {c.gain_db} dB" if c.gain_db else "  随机增益      关闭")
        lines.append(f"  极性反转      p={c.polarity_p}")
        lines.append(f"  频带掩蔽      p={c.band_stop_p}  带宽占比 {c.band_stop_min}~{c.band_stop_max}")
        lines.append(f"  时间裁剪      p={c.time_crop_p}  至少保留 {c.time_crop_min_keep}")
        lines.append(f"  通道互换      p={c.channel_swap_p}")
        return "\n".join(lines)

    def reset_stats(self) -> None:
        self.applied = {}


AUGMENT_PRESETS = {
    "default": AugmentConfig(),
    "off": AugmentConfig(enabled=False),
    "gain_only": AugmentConfig(polarity_p=0, band_stop_p=0, time_crop_p=0, channel_swap_p=0),
    "no_gain": AugmentConfig(gain_db=None, gain_p=0.0),
    "mask_only": AugmentConfig(gain_p=0, polarity_p=0, channel_swap_p=0),
}
