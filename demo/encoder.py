"""LAION 官方 CLAP 编码；统一 48 kHz、均匀十秒窗口与归一化。"""

import os
from pathlib import Path
import numpy as np
from music import normalize


class ClapEncoder:
    def __init__(self, assets, device):
        old = os.getcwd()
        third = assets / "third_party"
        try:
            if (third / "bert-base-uncased" / "vocab.txt").exists():
                os.chdir(third)
            import laion_clap

            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
            self.model.load_ckpt(model_id=1, verbose=False)
            self.model.eval()
        finally:
            os.chdir(old)

    def text(self, texts):
        import torch

        if not texts:
            raise ValueError("文本列表不能为空")
        with torch.inference_mode():
            value = self.model.get_text_embedding(texts, use_tensor=False)
        return normalize(value)

    def audio(self, path):
        import librosa
        import soundfile as sf
        import torch

        if not path or not Path(path).is_file():
            raise ValueError("请提供可读音频文件")
        info = sf.info(str(path))
        duration = info.duration
        if duration <= 0:
            raise ValueError("音频为空")
        n = min(3, max(1, int(duration // 10)))
        clips = [
            librosa.load(str(path), sr=48000, mono=True, offset=float(s), duration=10)[
                0
            ]
            for s in np.linspace(0, max(0, duration - 10), n)
        ]
        if any(len(c) == 0 or not np.isfinite(c).all() for c in clips):
            raise ValueError("音频解码失败")
        with torch.inference_mode():
            emb = self.model.get_audio_embedding_from_data(x=clips, use_tensor=False)
        return normalize(normalize(emb).mean(0))
