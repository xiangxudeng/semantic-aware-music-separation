"""把 CLAP 语义向量接进 Qwen2 的投影层。

结构（就是计划里的 `CLAP → 投影层 → Qwen2 解码器`）
----------------------------------------------------
```
音频 → CLAP 编码器 → 512 维语义向量
                      ↓ 投影层（两层 MLP + GELU）
                    N 个"音频 token"（每个 hidden 维）
                      ↓ 与文本 token 拼在一起
                    Qwen2 解码器 → 生成
```

为什么用"多个 token"而不是一个
------------------------------
一个 512 维向量压成 1 个 token，信息量太小，模型很难据此生成细节。
业界做法（LLaVA 系列）是把视觉/音频特征投影成若干个 token 再接进语言模型，
这里默认 8 个，可调。

关于可训练参数
--------------
投影层是随机初始化的（**不能零初始化**——零初始化会让音频 token 全是同一个向量，
等于没有信息）。它必须和 LoRA 一起训练；Qwen2 主干默认冻结。
"""

from __future__ import annotations

import torch
from torch import nn


class AudioProjector(nn.Module):
    """(B, in_dim) → (B, n_tokens, out_dim)"""

    def __init__(self, in_dim: int = 512, out_dim: int = 1536, n_tokens: int = 8,
                 hidden: int = 2048) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim * n_tokens),
        )
        # 注意：故意不做零初始化（见模块文档）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, n_windows, 512) → 先平均得到曲目级向量
            x = x.mean(1)
        return self.net(x).view(x.shape[0], self.n_tokens, self.out_dim)


class AudioLLM(nn.Module):
    """音频 token 前缀 + 文本的 Qwen2 包装。

    前向时把音频 token 拼在文本 embedding 前面，并相应地扩展 attention mask。
    """

    def __init__(self, llm: nn.Module, projector: AudioProjector) -> None:
        super().__init__()
        self.llm = llm
        self.projector = projector
        self.audio_token_id = None  # 仅用于记录，不占词表

    def forward(self, audio_emb: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **kw):
        audio_tokens = self.projector(audio_emb).to(dtype=self.llm.dtype)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([audio_tokens, text_emb], dim=1)

        bsz, n_audio = audio_emb.shape[0], audio_tokens.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = torch.cat(
            [torch.ones((bsz, n_audio), dtype=attention_mask.dtype, device=attention_mask.device),
             attention_mask], dim=1,
        )
        if labels is not None:
            # 音频 token 位置不计算损失
            pad = torch.full((bsz, n_audio), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([pad, labels], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels, **kw)

    @torch.no_grad()
    def generate_with_audio(self, audio_emb: torch.Tensor, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor | None = None, **kw):
        audio_tokens = self.projector(audio_emb).to(dtype=self.llm.dtype)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([audio_tokens, text_emb], dim=1)
        bsz, n_audio = audio_emb.shape[0], audio_tokens.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = torch.cat(
            [torch.ones((bsz, n_audio), dtype=attention_mask.dtype, device=attention_mask.device),
             attention_mask], dim=1,
        )
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=mask, **kw)

    def trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]
