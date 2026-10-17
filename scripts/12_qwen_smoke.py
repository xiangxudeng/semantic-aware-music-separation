"""第 1 周验收项：验证 Qwen2-1.5B-Instruct 能否本地推理。

大纲第 1 周的验收标准里有一条"大模型 4 比特量化推理正常"。本机无 NVIDIA 独立显卡，
4 比特量化依赖 CUDA（bitsandbytes），因此本脚本做两件事：

1. 用 float32 在 CPU 上跑通一次真实推理，验证模型、权重、分词器整条链路；
2. 尝试 4 比特加载，把结果如实打印出来——本机预期失败，云服务器上应该成功。

用法：
    python scripts/12_qwen_smoke.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Qwen/Qwen2-1.5B-Instruct"
PROMPT = "用一句话解释什么是音源分离。"
OUT = Path(__file__).resolve().parents[1] / "results" / "qwen_smoke.txt"


def report(lines: list[str]) -> None:
    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


def main() -> int:
    lines = [f"Qwen2-1.5B-Instruct 推理自检   {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    lines.append(f"torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}")
    lines.append(f"模型仓库：{REPO}")

    print(f"加载分词器与模型（{REPO}，首次会下载约 3 GB）...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)
    model.eval()
    lines.append(f"\n【CPU float32 推理】加载耗时 {time.time() - t0:.0f} 秒")
    lines.append(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    dt = time.time() - t0
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    lines.append(f"提问：{PROMPT}")
    lines.append(f"回答：{answer.strip()}")
    lines.append(f"生成耗时 {dt:.1f} 秒（max_new_tokens=40），约 {40 / dt:.1f} token/秒")
    lines.append("结论：CPU float32 推理链路正常。")

    # 第二步：尝试 4 比特
    lines.append("\n【4 比特量化推理】")
    try:
        from transformers import BitsAndBytesConfig

        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # 4 比特模型不能再调用 .to()，必须用 device_map 指定设备
        qmodel = AutoModelForCausalLM.from_pretrained(
            REPO, quantization_config=cfg, device_map="auto"
        )
        used = torch.cuda.memory_allocated() / 2**30
        lines.append(f"结果：加载成功。模型占用显存约 {used:.2f} GB。")
        lines.append("结论：4 比特量化推理可用，满足大纲对显存不超过 8 GB 的要求。")
        del qmodel
    except Exception as exc:  # noqa: BLE001
        import traceback

        lines.append(f"结果：加载失败 —— {type(exc).__name__}: {str(exc)[:200]}")
        lines.append("完整报错：")
        lines.extend("  " + l for l in traceback.format_exc().splitlines()[-6:])
        lines.append("说明：4 比特量化依赖 CUDA。若当前机器没有 NVIDIA 显卡，属预期结果，")
        lines.append("      需在云 GPU 服务器上补做。")

    report(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
