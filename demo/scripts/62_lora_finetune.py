"""第 3 周第 5 步：LoRA 4-bit 指令微调。

训练什么
--------
只有两处参数在更新：
- **音频投影层**（26.2 M）：把 CLAP 的 512 维语义向量变成 8 个 Qwen2 输入 token
- **Qwen2 上的 LoRA 适配器**：低秩增量，不动主干

Qwen2 主干用 4 bit 量化加载并冻结（大纲要求显存 ≤8 GB）。

一个关键的性能设计
------------------
训练时**不需要读音频**：CLAP 是冻结的，所有指令涉及的 MTT 片段在第 2 周算零样本时
已经把语义向量算好并缓存（`results/clap_official_emb_all25863_w3.npy`，25863 × 512）。
所以这里只按 clip_id 查表，省掉了整个音频解码与 CLAP 前向，这是能不能在一小时内跑完的关键。

用法（云端 GPU）：
    python scripts/62_lora_finetune.py --epochs 2 --batch-size 8 --max-per-task 6000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = Path(os.environ.get("AI_MUSIC_ASSETS", ROOT.parent / "ai_music")).resolve()

import numpy as np
import pandas as pd
import torch

from src.llm.projector import AudioLLM, AudioProjector
from src.utils.run_log import run_record

DATA = ASSETS / "data" / "mtt"
RESULTS = ASSETS / "results"
LLM = "Qwen/Qwen2-1.5B-Instruct"


def load_embeddings() -> dict[str, np.ndarray]:
    """clip_id → CLAP 语义向量（复用第 2 周缓存，避免训练时读音频）。"""
    anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    # 优先读 fp16 版本（26 MB）：云端上传 53 MB 的 fp32 版本会中途断开，
    # 半精度对条件向量来说完全够用
    fp16 = RESULTS / "clap_official_emb_all25863_w3_fp16.npy"
    path = fp16 if fp16.exists() else RESULTS / "clap_official_emb_all25863_w3.npy"
    emb = np.load(path).astype(np.float32)
    return {cid: emb[i] for i, cid in enumerate(anno["clip_id"])}


def load_instructions(split: str, max_per_task: int) -> list[dict]:
    path = RESULTS / f"instruction_{split}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_per_task > 0:
        by_task: dict[str, list[dict]] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        kept = []
        for task, items in by_task.items():
            kept.extend(items[:max_per_task])
        rows = kept
    return rows


def rare_tag_oversample(
    rows: list[dict], factor: int, prevalence: pd.Series
) -> list[dict]:
    """类别感知的稀有标签加权（数据层实现）。

    大纲第 3 周要求"针对标签长尾问题，设计类别感知的损失加权策略"。
    最直接的等价做法是**提高含稀有标签样本的采样概率**：把提到长尾标签的样本复制若干份，
    效果上等于给它们的损失加了权重，而且不动训练循环。
    """
    if factor <= 0:
        return rows
    rare = set(prevalence[prevalence < 0.01].index)
    if not rare:
        return rows
    extra = []
    for r in rows:
        text = r["prompt"] + " " + r["answer"]
        if any(t in text for t in rare):
            extra.extend([r] * factor)
    print(
        f"  稀有标签加权：{len(extra)} 条被复制 {factor} 次（原 {len(rows)} 条 → {len(rows) + len(extra)} 条）"
    )
    return rows + extra


def build_batch(rows, tok, vec_of, device):
    """把一批指令拼成 (audio_emb, input_ids, attention_mask, labels)。"""
    audio, texts, labels = [], [], []
    for r in rows:
        vec = vec_of.get(r["clip_id"])
        if vec is None:
            continue
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = prefix + r["answer"] + tok.eos_token
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        # 只对回答部分算损失：提示词位置置 −100
        lab = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        audio.append(vec)
        texts.append(torch.tensor(f_ids))
        labels.append(torch.tensor(lab))
    if not audio:
        return None
    max_len = max(len(t) for t in texts)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(texts), max_len), dtype=torch.long)
    lab = torch.full((len(texts), max_len), -100, dtype=torch.long)
    for i, (t, l) in enumerate(zip(texts, labels)):
        input_ids[i, : len(t)] = t
        attn[i, : len(t)] = 1
        lab[i, : len(l)] = l
    return (
        torch.from_numpy(np.stack(audio)).to(device),
        input_ids.to(device),
        attn.to(device),
        lab.to(device),
    )


def build_optimizer(lora_params, proj_params, lr: float, proj_lr: float):
    """两组参数：LoRA 用 `lr`，投影层用 `proj_lr`；`proj_lr<=0` 时合成一组。

    为什么要分开：投影层是**从零学起**的（26.2 M 随机初始化），LoRA 是在预训练权重上做
    低秩增量——两者共用同一个学习率并不合理，第 3 周描述生成失败的归因里就有这一条。
    返回 (optimizer, 投影层那一组)，第二项给"两阶段训练"用来临时置零 LoRA 学习率。
    """
    lora_params, proj_params = list(lora_params), list(proj_params)
    if proj_lr <= 0:
        return torch.optim.AdamW(lora_params + proj_params, lr=lr), None
    groups = [{"params": lora_params, "lr": lr}, {"params": proj_params, "lr": proj_lr}]
    return torch.optim.AdamW(groups), groups[1]


def self_check() -> int:
    """不加载模型，只验参数分组与"第一阶段只训投影层"这个行为。"""
    a = torch.nn.Linear(4, 4)  # 冒充 LoRA 参数
    b = torch.nn.Linear(4, 4)  # 冒充投影层
    opt, grp = build_optimizer(a.parameters(), b.parameters(), 1e-4, 1e-3)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == 4, "同一个参数不能出现在两组里"
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 1e-3]
    assert grp is opt.param_groups[1], "投影层必须是第二组"

    opt2, grp2 = build_optimizer(a.parameters(), b.parameters(), 1e-4, 0.0)
    assert (
        grp2 is None
        and len(opt2.param_groups) == 1
        and opt2.param_groups[0]["lr"] == 1e-4
    )

    x = torch.randn(3, 4)
    a0, b0 = a.weight.detach().clone(), b.weight.detach().clone()
    opt.param_groups[0]["lr"] = 0.0  # 模拟"第一阶段只训投影层"
    (a(x).sum() + b(x).sum()).backward()
    opt.step()
    assert torch.equal(a.weight, a0), "第一阶段 LoRA 不该变"
    assert not torch.equal(b.weight, b0), "第一阶段投影层应该变"
    print("self-check ok：分组不重叠、lr 正确、第一阶段只动投影层")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument(
        "--max-per-task",
        type=int,
        default=6000,
        help="每个任务最多用多少条，0 表示不限制",
    )
    ap.add_argument("--max-steps", type=int, default=0, help="限制总步数，便于冒烟测试")
    ap.add_argument("--tag", default="lora1")
    ap.add_argument(
        "--rare-oversample",
        type=int,
        default=0,
        help="含长尾标签的样本复制几份，等价于类别感知的损失加权",
    )
    ap.add_argument("--smoke", action="store_true", help="极小规模冒烟：50 条、20 步")
    ap.add_argument(
        "--proj-lr",
        type=float,
        default=0.0,
        help="投影层单独的学习率；0 表示和 LoRA 共用 --lr",
    )
    ap.add_argument(
        "--stage1-steps",
        type=int,
        default=0,
        help="前 N 步只训投影层（LoRA 学习率临时置 0），先把音频接地再联合训；"
        "需要同时给 --proj-lr",
    )
    ap.add_argument(
        "--self-check", action="store_true", help="只验参数分组与两阶段行为，不加载模型"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子：LoRA 初始化与每轮样本打乱都用它，多跑几个种子就能给出方差",
    )
    args = ap.parse_args()

    if args.self_check:
        return self_check()
    if args.stage1_steps and args.proj_lr <= 0:
        print("!! --stage1-steps 需要同时给 --proj-lr，否则第一阶段会把 LoRA 也冻上")
        return 2
    if args.smoke:
        args.max_per_task, args.max_steps, args.epochs, args.batch_size = 50, 20, 1, 4

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"随机种子 {args.seed}", flush=True)

    with run_record(f"w3_lora_{args.tag}", config=vars(args)) as run:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("加载指令数据与语义向量缓存...", flush=True)
        vec_of = load_embeddings()
        rows = load_instructions("train", args.max_per_task)
        from collections import Counter

        print(
            f"  训练指令 {len(rows)} 条，任务分布 {dict(Counter(r['task'] for r in rows))}",
            flush=True,
        )
        if args.rare_oversample:
            anno_prev = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
            tr_ids = set(
                pd.read_csv(
                    DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0]
                )[0].astype(str)
            )
            sub = anno_prev[anno_prev["clip_id"].astype(str).isin(tr_ids)]
            tags50 = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
            prevalence = sub[tags50].astype(int).mean()
            rows = rare_tag_oversample(rows, args.rare_oversample, prevalence)
            print(f"  加权后训练指令 {len(rows)} 条", flush=True)

        print(f"加载 {LLM}（4bit）...", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLM)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            LLM,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        llm.config.use_cache = False
        for p in llm.parameters():
            p.requires_grad = False
        print(
            f"  用时 {time.time() - t0:.0f}s，显存 {torch.cuda.memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )

        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        llm = get_peft_model(llm, peft_cfg)
        llm.print_trainable_parameters()

        # 投影层要显式搬到主干所在的设备上：model.to(dtype) 只改精度不改设备，
        # 4bit 主干的参数在 GPU 上而新建的投影层在 CPU，直接前向会报设备不匹配。
        # 另外投影层保持 fp32 训练（更稳），输出由 AudioLLM 内部转成主干的精度。
        device = next(llm.parameters()).device
        proj = AudioProjector(
            in_dim=512, out_dim=llm.config.hidden_size, n_tokens=args.n_tokens
        ).to(device)
        model = AudioLLM(llm, proj)

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        print(f"可训练参数合计 {n_train / 1e6:.2f} M（投影层 + LoRA）", flush=True)
        run.note(
            f"可训练 {n_train / 1e6:.2f}M；指令 {len(rows)} 条；4bit 显存 {torch.cuda.memory_allocated() / 1e9:.2f}GB"
        )

        proj_params = list(proj.parameters())
        lora_params = [p for p in trainable if not any(p is q for q in proj_params)]
        opt, _ = build_optimizer(lora_params, proj_params, args.lr, args.proj_lr)
        if args.proj_lr > 0:
            print(f"两组学习率：LoRA {args.lr:g}｜投影层 {args.proj_lr:g}", flush=True)
        if args.stage1_steps:
            # 第一阶段：只训投影层。用"把 LoRA 组的学习率置 0"实现，等价于冻结，
            # 但不用重建优化器（重建会丢掉动量，0 风险更小）
            opt.param_groups[0]["lr"] = 0.0
            print(f"前 {args.stage1_steps} 步只训投影层，之后转入联合训练", flush=True)
        steps_per_epoch = max(1, math.ceil(len(rows) / args.batch_size))
        total = steps_per_epoch * args.epochs
        if args.max_steps:
            total = min(total, args.max_steps)
        print(
            f"计划 {total} 步（每轮 {steps_per_epoch} 步 × {args.epochs} 轮）",
            flush=True,
        )

        log_path = RESULTS / f"lora_train_{args.tag}.csv"
        step, t0 = 0, time.time()
        model.train()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step,loss,lr,elapsed_s\n")
            for epoch in range(1, args.epochs + 1):
                order = np.random.permutation(len(rows))
                for i in range(0, len(rows), args.batch_size):
                    if step >= total:
                        break
                    batch = [rows[j] for j in order[i : i + args.batch_size]]
                    built = build_batch(batch, tok, vec_of, device)
                    if built is None:
                        continue
                    audio, ids, attn, lab = built
                    out = model(audio, ids, attn, labels=lab)
                    loss = out.loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    step += 1
                    f.write(
                        f"{step},{float(loss):.4f},{opt.param_groups[0]['lr']:.1e},{time.time() - t0:.0f}\n"
                    )
                    f.flush()
                    if args.stage1_steps and step == args.stage1_steps:
                        opt.param_groups[0]["lr"] = args.lr
                        print(
                            f"  step {step}：转入联合训练（LoRA lr={args.lr:g}）",
                            flush=True,
                        )
                    if step % 20 == 0 or step == 1:
                        print(
                            f"  step {step}/{total} loss {float(loss):.4f} "
                            f"({time.time() - t0:.0f}s，{torch.cuda.memory_allocated() / 1e9:.2f}GB)",
                            flush=True,
                        )
                if step >= total:
                    break

        out_dir = RESULTS / f"lora_{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.llm.save_pretrained(out_dir)
        torch.save(proj.state_dict(), out_dir / "projector.pt")
        (out_dir / "train_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n适配器 -> {out_dir}\n日志 -> {log_path}")
        print(
            f"训练完成：{step} 步，用时 {(time.time() - t0) / 60:.1f} 分钟，"
            f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        run.artifact(log_path, out_dir)
        run.note(
            f"LoRA 训练 {step} 步，用时 {(time.time() - t0) / 60:.1f} 分钟，"
            f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
