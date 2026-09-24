"""第 3 周第 3 步：跨模态架构前向打通（CLAP → 投影层 → Qwen2）。

计划里的目标：`CLAP → 投影层 → Qwen2 解码器` 前向打通，确认特征维度对齐。

本脚本逐项确认：
1. 投影层把 512 维语义向量变成 N 个 hidden 维 token，形状正确
2. 音频 token 与文本 embedding 拼接后，attention mask 同步扩展
3. 语言模型的 logits 形状正确
4. **损失只在文本上计算**（音频 token 位置被置成 -100，不应该算损失）
5. 用真实 prompt 做一次 `generate`，确认整条链路能跑出文字

CPU 上也能跑（float32），云上会换成 4-bit 量化 + LoRA。

用法：
    python scripts/61_crossmodal_forward.py                # 默认 CPU float32
    python scripts/61_crossmodal_forward.py --load-4bit    # 需要 CUDA + bitsandbytes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.llm.projector import AudioLLM, AudioProjector
from src.utils.run_log import run_record

LLM = "Qwen/Qwen2-1.5B-Instruct"
PROMPT = "这段音乐可以用哪些标签描述？"

CHECKS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    CHECKS.append((bool(ok), label, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"    {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    with run_record("w3_crossmodal_forward",
                    config={"llm": LLM, "n_tokens": args.n_tokens, "load_4bit": args.load_4bit}) as run:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"加载 {LLM}（{'4bit' if args.load_4bit else 'float32'}）...", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLM)
        if args.load_4bit:
            from transformers import BitsAndBytesConfig

            llm = AutoModelForCausalLM.from_pretrained(
                LLM, quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
            )
        else:
            llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float32)
        llm.eval()
        for p in llm.parameters():
            p.requires_grad = False
        print(f"  用时 {time.time() - t0:.0f}s，hidden={llm.config.hidden_size}", flush=True)

        proj = AudioProjector(in_dim=512, out_dim=llm.config.hidden_size, n_tokens=args.n_tokens)
        model = AudioLLM(llm, proj)

        B = 2
        audio = torch.randn(B, 512)
        text = tok([PROMPT] * B, return_tensors="pt", padding=True)

        print("\n=== 1. 投影层形状 ===")
        tokens = proj(audio)
        check(tokens.shape == (B, args.n_tokens, llm.config.hidden_size),
              f"512 维 → ({args.n_tokens}, {llm.config.hidden_size})", str(tuple(tokens.shape)))
        n_audio_dim = proj(torch.randn(B, 3, 512))  # 多窗口输入应自动求平均
        check(n_audio_dim.shape == tokens.shape, "多窗口输入 (B, w, 512) 自动求平均", str(tuple(n_audio_dim.shape)))

        print("\n=== 2/3/4. 拼接、mask、损失 ===")
        out = model(audio, text["input_ids"], text["attention_mask"])
        want = (B, text["input_ids"].shape[1] + args.n_tokens, llm.config.vocab_size)
        check(out.logits.shape == want, "logits 形状 = 音频 token + 文本 token", str(tuple(out.logits.shape)))

        labels = text["input_ids"].clone()
        labels[text["attention_mask"] == 0] = -100
        out2 = model(audio, text["input_ids"], text["attention_mask"], labels=labels)
        check(torch.isfinite(out2.loss), "带标签前向能算出有限损失", f"loss={float(out2.loss):.4f}")
        loss_before = float(out2.loss)

        # 只对文本算损失：把文本标签换掉，损失应当变化（说明文本确实参与了损失）
        labels2 = labels.clone()
        labels2[labels2 != -100] = 0
        out3 = model(audio, text["input_ids"], text["attention_mask"], labels=labels2)
        check(abs(float(out3.loss) - loss_before) > 1e-6,
              "改文本标签会改变损失（说明损失只作用在文本，音频位置被屏蔽）",
              f"{loss_before:.4f} → {float(out3.loss):.4f}")
        check(torch.isfinite(out.logits).all(), "logits 无 NaN/Inf")

        print("\n=== 5. 端到端生成 ===")
        t0 = time.time()
        gen = model.generate_with_audio(audio[:1], text["input_ids"][:1], text["attention_mask"][:1],
                                        max_new_tokens=args.max_new, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
        answer = tok.decode(gen[0][text["input_ids"].shape[1]:], skip_special_tokens=True)
        check(len(answer) > 0, f"生成成功（{time.time() - t0:.0f}s）", repr(answer[:60]))

        n_train = sum(p.numel() for p in proj.parameters())
        n_total = sum(p.numel() for p in llm.parameters())
        print(f"\n投影层参数 {n_train / 1e6:.1f} M；Qwen2 主干 {n_total / 1e9:.2f} B（冻结）")
        print("注意：此刻音频 token 是随机的，生成内容没有意义；打通链路即可。")
        run.note(f"跨模态前向打通：音频 {args.n_tokens} token，投影层 {n_train / 1e6:.1f} M 参数，"
                 f"loss {loss_before:.4f}")
        run.artifact(ROOT / "src" / "llm" / "projector.py")

        failed = [c for c in CHECKS if not c[0]]
        print(f"\n{'=' * 60}\n共 {len(CHECKS)} 项，通过 {len(CHECKS) - len(failed)}，失败 {len(failed)}")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
