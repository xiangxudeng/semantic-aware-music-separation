"""跨模态与大模型相关模块：投影层、指令数据、LoRA 训练。"""

from .projector import AudioProjector, AudioLLM

__all__ = ["AudioProjector", "AudioLLM"]
