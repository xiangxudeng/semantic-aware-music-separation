"""CLAP 通用语义特征提取模块（第 2 周 P0 交付物）。

对外提供两种粒度的 512 维语义向量：

- **全曲级**：整首歌一个向量，用于"这首歌是什么风格"这类整体判断
- **片段级**：按固定窗口切出来的一系列向量，用于定位"哪一段里有钢琴solo"

实现用的是 **LAION 官方 laion_clap**（不是 HuggingFace 的移植版）。这一点很关键：
第 2 周做过对照，两者的嵌入根本不在一个空间（同一条音频余弦只有 0.29），
移植版在 MTT 上的零样本 mAP 是 0.2163，官方实现是 0.3136。项目统一走官方实现。

全曲级向量的取法
----------------
官方实现单次只看 10 秒（CLAP 的输入上限）。29 秒的歌如果只抽一个 10 秒窗口，
同一个文件多跑几次结果都不一样。这里改成**均匀取 N 个窗口再平均**：
比随机取更可复现，也覆盖得更全。实测这一步在零样本任务上值 +0.005 到 +0.01 个 mAP。

用法
----
    from src.semantic.clap_encoder import ClapEncoder

    enc = ClapEncoder(device="auto", windows=3)
    v = enc.encode_file("data/mtt/.../xxx.mp3")        # (512,)
    clips = enc.encode_clips("xxx.mp3", clip_seconds=10)  # (n, 512)
    t = enc.encode_text(["This is a sound of piano"])     # (1, 512)
    print(enc.report())                                   # 耗时统计
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_FFMPEG_FALLBACK = r"/usr/bin"
if Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")

SAMPLE_RATE = 48000  # CLAP 的训练采样率，不要改
EMBED_DIM = 512
CLIP_SECONDS = 10.0  # CLAP 单次输入上限

ROOT = Path(__file__).resolve().parents[2]
# laion_clap 在导入时会按**当前工作目录**去找 bert-base-uncased / roberta-base /
# facebook/bart-base 三个分词器，国内网络直接下载大概率失败。third_party 里放好了
# 这三个分词器的小文件，导入时临时切过去，用完马上切回来。
_THIRD_PARTY = ROOT / "third_party"


@contextmanager
def _local_tokenizers():
    if not (_THIRD_PARTY / "bert-base-uncased" / "vocab.txt").exists():
        yield  # 没有本地副本就走默认的联网下载
        return
    old = os.getcwd()
    os.chdir(_THIRD_PARTY)
    try:
        yield
    finally:
        os.chdir(old)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。分母加下限，避免静音片段算出 NaN。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass
class Stats:
    """耗时统计。第 3 周要评估"语义向量给分离用"的开销，这个必须留着。"""

    n_calls: int = 0
    n_clips: int = 0
    load_seconds: float = 0.0
    forward_seconds: float = 0.0
    wall_seconds: float = 0.0
    failures: int = 0
    bad_files: list[str] = field(default_factory=list)

    @property
    def seconds_per_clip(self) -> float:
        return self.forward_seconds / self.n_clips if self.n_clips else 0.0

    def as_dict(self) -> dict:
        return {
            "调用次数": self.n_calls,
            "编码片段数": self.n_clips,
            "读音频耗时(秒)": round(self.load_seconds, 2),
            "模型前向耗时(秒)": round(self.forward_seconds, 2),
            "总耗时(秒)": round(self.wall_seconds, 2),
            "平均每片段(秒)": round(self.seconds_per_clip, 3),
            "读不了的文件数": self.failures,
        }


class ClapEncoder:
    """CLAP 音频/文本编码器。

    Parameters
    ----------
    device : "auto" / "cuda" / "cpu"
    windows : 全曲级向量取几个 10 秒窗口做平均
    model_id : laion_clap 的权重编号；-1 表示用默认（非融合版 = 1）
    ckpt : 直接指定权重路径，一般不用
    """

    def __init__(
        self,
        device: str = "auto",
        windows: int = 3,
        model_id: int = 1,
        ckpt: str | None = None,
    ) -> None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.windows = max(1, int(windows))
        self.stats = Stats()

        with _local_tokenizers():
            import laion_clap

            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
            self.model.load_ckpt(ckpt=ckpt, model_id=model_id, verbose=False)
            self.model.eval()

    # ---------- 音频 ----------

    def _load(self, path: str | Path) -> np.ndarray | None:
        import librosa

        t0 = time.time()
        try:
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:  # noqa: BLE001
            self.stats.failures += 1
            self.stats.bad_files.append(str(path))
            print(f"    !! 读不了 {Path(path).name}: {type(exc).__name__}", flush=True)
            return None
        finally:
            self.stats.load_seconds += time.time() - t0
        return audio

    @staticmethod
    def _even_windows(audio: np.ndarray, n: int) -> list[np.ndarray]:
        """均匀取 n 个**长度正好 10 秒**的窗口（起点均匀铺开）。

        注意每个窗口必须 <= 10 秒：CLAP 内部对超长输入是"随机裁 10 秒"，
        一旦窗口超过上限，同一个文件编两次会得到不同结果。
        """
        clip_len = int(CLIP_SECONDS * SAMPLE_RATE)
        if len(audio) <= clip_len:
            return [audio]
        n = min(n, max(1, len(audio) // clip_len))
        starts = np.linspace(0, len(audio) - clip_len, n).astype(int)
        return [audio[s : s + clip_len] for s in starts]

    def _forward_clips(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        t0 = time.time()
        with torch.no_grad():
            emb = self.model.get_audio_embedding_from_data(x=clips, use_tensor=False)
        self.stats.forward_seconds += time.time() - t0
        self.stats.n_clips += len(clips)
        return l2_normalize(emb)

    def encode_file(self, path: str | Path) -> np.ndarray:
        """全曲级向量：(512,)。取 windows 个窗口的平均，再归一化。

        只解码需要的三个 10 秒窗口，不整曲解码——4 分钟的曲子整解要好几秒，
        而窗口化只解 30 秒，快一个数量级，结果与整解后切窗完全一致。
        """
        t0 = time.time()
        self.stats.n_calls += 1
        clips = self._windowed_load(path)
        if clips is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        emb = self._forward_clips(clips)
        self.stats.wall_seconds += time.time() - t0
        return l2_normalize(emb.mean(0))[0]

    def _windowed_load(self, path: str | Path) -> list[np.ndarray] | None:
        """按窗口位置定点解码；文件太短就整段读。"""
        import librosa
        import soundfile as sf

        t0 = time.time()
        try:
            info = sf.info(str(path))
            total = int(info.frames * SAMPLE_RATE / info.samplerate)
        except Exception as exc:  # noqa: BLE001
            self.stats.failures += 1
            self.stats.bad_files.append(str(path))
            print(f"    !! 读不了 {Path(path).name}: {type(exc).__name__}", flush=True)
            return None
        clip_len = int(CLIP_SECONDS * SAMPLE_RATE)
        try:
            if total <= clip_len:
                audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
                clips = [audio]
            else:
                n = min(self.windows, max(1, total // clip_len))
                starts = np.linspace(0, total - clip_len, n).astype(int)
                clips = [
                    librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                                 offset=float(s) / SAMPLE_RATE, duration=CLIP_SECONDS)[0]
                    for s in starts
                ]
        except Exception as exc:  # noqa: BLE001
            self.stats.failures += 1
            self.stats.bad_files.append(str(path))
            print(f"    !! 读不了 {Path(path).name}: {type(exc).__name__}", flush=True)
            return None
        finally:
            self.stats.load_seconds += time.time() - t0
        return clips

    def encode_files(self, paths: list[str | Path]) -> np.ndarray:
        """批量全曲级向量：(n, 512)。"""
        return np.stack([self.encode_file(p) for p in paths]) if paths else np.zeros((0, EMBED_DIM), np.float32)

    def encode_clips(self, path: str | Path, clip_seconds: float = CLIP_SECONDS,
                     hop_seconds: float | None = None) -> np.ndarray:
        """片段级向量：(n_clips, 512)。

        clip_seconds 控制窗口长度，hop_seconds 控制步长（默认不重叠）。
        """
        t0 = time.time()
        self.stats.n_calls += 1
        audio = self._load(path)
        if audio is None:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        clip_len = int(clip_seconds * SAMPLE_RATE)
        hop = int((hop_seconds if hop_seconds else clip_seconds) * SAMPLE_RATE)
        starts = list(range(0, len(audio), hop))  # 末尾不足一个窗口就补零，尾巴不丢
        clips = []
        for s in starts:
            chunk = audio[s : s + clip_len]
            if len(chunk) < clip_len:  # 末尾不足一个窗口就补零
                chunk = np.pad(chunk, (0, clip_len - len(chunk)))
            clips.append(chunk)
        emb = self._forward_clips(clips)
        self.stats.wall_seconds += time.time() - t0
        return emb

    # ---------- 文本 ----------

    def encode_text(self, texts: list[str] | str, batch_size: int = 256) -> np.ndarray:
        """文本向量：(n, 512)。

        分批送进模型：一次塞几万条会让中间激活把内存撑爆（实测 1.8 万条直接把进程干掉）。
        """
        if isinstance(texts, str):
            texts = [texts]
        t0 = time.time()
        self.stats.n_calls += 1
        if len(texts) <= batch_size:
            emb = self.model.get_text_embedding(texts, use_tensor=False)
        else:
            parts = [
                self.model.get_text_embedding(texts[i : i + batch_size], use_tensor=False)
                for i in range(0, len(texts), batch_size)
            ]
            emb = np.concatenate(parts, axis=0)
        self.stats.forward_seconds += time.time() - t0
        self.stats.wall_seconds += time.time() - t0
        return l2_normalize(emb)

    def tag_prompts(self, tags: list[str], templates: list[str] | None = None) -> np.ndarray:
        """把标签变成提示词再取平均，对应零样本评测里的"四模板集成"。"""
        templates = templates or ["This is a sound of {}"]
        per_tpl = [self.encode_text([t.format(tag) for tag in tags]) for t in templates]
        return l2_normalize(np.mean(per_tpl, axis=0))

    # ---------- 工具 ----------

    @staticmethod
    def similarity(audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        """余弦相似度矩阵：(n_audio, n_text)。两边都假定已归一化。"""
        return np.asarray(audio_emb) @ np.asarray(text_emb).T

    def report(self) -> str:
        lines = ["CLAP 编码器耗时统计（设备 {}，窗口数 {}）".format(self.device, self.windows)]
        for k, v in self.stats.as_dict().items():
            lines.append(f"  {k:16} {v}")
        if self.stats.bad_files:
            lines.append("  读不了的文件：" + ", ".join(Path(p).name for p in self.stats.bad_files))
        return "\n".join(lines)

    def reset_stats(self) -> None:
        self.stats = Stats()
