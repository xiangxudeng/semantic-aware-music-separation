"""第 3 周第 5 步（续）：评测 LoRA 微调后的模型。

要出三个数
----------
1. **50 类标签 mAP**：大纲的验收口径之一（"微调后模型 50 类固定标签 mAP 相比零样本提升"）
2. **长尾 14 类的 mAP 与召回率**：按第 2 周统一的口径（召回用 MTT 训练集比例判定），
   对应大纲的"长尾标签召回率提升 ≥12%"
3. **描述生成 BLEU**：用 LP-MusicCaps 的测试集（300 首、每首 4 条参考描述）

标签 mAP 怎么从生成式模型里算出来
----------------------------------
生成式模型只吐文字，没有分数，直接算不了 AP。做法是把它变成判别任务：
对每个标签问一句"这段音乐里有 {标签} 吗？"，比较模型在**回答首 token** 上
「有」与「没有」的相对概率，作为该标签的分数。这样每首歌得到 50 个分数，
就能按标准口径算 AP / AUC / 召回。

成本：一首歌一次批量前向（50 个问句），全部 5329 首约几分钟。

用法：
    python scripts/63_eval_lora.py --adapter results/lora_lora1 --limit 0
    python scripts/63_eval_lora.py --adapter results/lora_lora1 --limit 500   # 快速看趋势
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.llm.projector import AudioLLM, AudioProjector
from src.semantic.tagging_metrics import evaluate_scores, summarize
from src.utils.run_log import run_record

DATA = ROOT / "data" / "mtt"
LP = ROOT / "data" / "lpmusiccaps"
RESULTS = ROOT / "results"


def load_model(adapter: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = json.loads((adapter / "train_args.json").read_text(encoding="utf-8"))
    # 分词器要从基座模型加载：save_pretrained 保存的是 LoRA 适配器，目录里只有 adapter_config.json，
    # 没有 tokenizer 文件，直接 from_pretrained(adapter) 会报 "does not appear to have config.json"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        torch_dtype=torch.float16, device_map="auto",
    )
    llm = PeftModel.from_pretrained(base, adapter)
    llm.eval()
    device = next(llm.parameters()).device
    proj = AudioProjector(in_dim=512, out_dim=llm.config.hidden_size,
                          n_tokens=args.get("n_tokens", 8)).to(device)
    proj.load_state_dict(torch.load(adapter / "projector.pt", map_location=device))
    model = AudioLLM(llm, proj)
    model.eval()
    return model, tok


@torch.no_grad()
def score_tags(model, tok, vec, tags, device, batch=25) -> np.ndarray:
    """返回 (n_tags,) 的分数：logit(有) − logit(没)。"""
    yes_id = tok("有", add_special_tokens=False)["input_ids"][0]
    no_id = tok("没", add_special_tokens=False)["input_ids"][0]
    out = []
    for i in range(0, len(tags), batch):
        chunk = tags[i : i + batch]
        seqs = [
            tok.apply_chat_template([{"role": "user", "content": f"这段音乐里有{t}吗？"}],
                                    tokenize=False, add_generation_prompt=True)
            for t in chunk
        ]
        enc = tok(seqs, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        audio = torch.from_numpy(np.stack([vec] * len(chunk))).to(device)
        logits = model(audio, enc["input_ids"], enc["attention_mask"]).logits
        last = enc["attention_mask"].sum(1) - 1 + model.projector.n_tokens
        step_logits = logits[torch.arange(len(chunk)), last]
        out.append((step_logits[:, yes_id] - step_logits[:, no_id]).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 首；0 表示全部")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    adapter = ROOT / args.adapter if not Path(args.adapter).is_absolute() else Path(args.adapter)
    tag = args.tag or adapter.name

    with run_record(f"w3_eval_lora_{tag}", config=vars(args)) as run:
        tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
        anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
        anno["clip_id"] = anno["clip_id"].astype(str)
        labels_all = anno[tags].astype(int).to_numpy()
        fp16 = RESULTS / "clap_official_emb_all25863_w3_fp16.npy"
        emb = np.load(fp16 if fp16.exists() else RESULTS / "clap_official_emb_all25863_w3.npy")
        emb = emb.astype(np.float32)

        test_ids = pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str)
        train_ids = set(pd.read_csv(DATA / "train_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str))
        te_idx = np.flatnonzero(anno["clip_id"].isin(set(test_ids)).to_numpy())
        tr_idx = np.flatnonzero(anno["clip_id"].isin(train_ids).to_numpy())
        # 长尾标签的定义必须基于**完整的测试集**，不能跟着 --limit 的子集变：
        # 只评 300 首时几乎所有标签的正样本都不足 100，会把全部标签都算成长尾
        # （实测长尾 mAP 与整体 mAP 完全相等，就是这么来的）
        full_te = labels_all[te_idx]
        tail_tags = [t for i, t in enumerate(tags) if full_te[:, i].sum() < 100]
        if args.limit:
            te_idx = te_idx[: args.limit]
        prevalence = labels_all[tr_idx].mean(0)   # 召回率判定比例：统一用训练集口径

        model, tok = load_model(adapter)
        device = next(model.parameters()).device
        print(f"待评 {len(te_idx)} 首 × {len(tags)} 类标签", flush=True)

        scores = []
        t0 = time.time()
        for i, idx in enumerate(te_idx, 1):
            scores.append(score_tags(model, tok, emb[idx], tags, device))
            if i % 50 == 0 or i == 1:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(te_idx)}  {rate:.2f}s/首  剩余 {rate * (len(te_idx) - i) / 60:.1f} 分钟",
                      flush=True)
        score = np.stack(scores)
        y = labels_all[te_idx]

        df = evaluate_scores(score, y, tags, prevalence=prevalence)
        s = summarize(df, tail_tags)

        zs_path = RESULTS / "clap_zeroshot_ap_official_test5329_w3_t4.csv"
        zs = pd.read_csv(zs_path).set_index("tag") if zs_path.exists() and not args.limit else None
        print(f"\n{'指标':22}{'LoRA 微调':>12}" + (f"{'零样本':>12}{'变化':>12}" if zs is not None else ""))
        for key, label in (("全部_AP", "50 类 mAP"), ("长尾_AP", "长尾 14 类 mAP"),
                           ("长尾_AUC", "长尾 AUC"), ("长尾_recall@prev", "长尾召回率")):
            row = f"{label:22}{s[key]:>12.4f}"
            if zs is not None:
                z = {"全部_AP": zs.AP.mean(), "长尾_AP": zs[zs.index.isin(tail_tags)].AP.mean(),
                     "长尾_AUC": zs[zs.index.isin(tail_tags)].AUC.mean()}.get(key)
                if z:
                    row += f"{z:>12.4f}{(s[key] / z - 1):>11.1%}"
            print(row)
        if zs is not None:
            rel = s["全部_AP"] / zs.AP.mean() - 1
            print(f"\n大纲口径：50 类 mAP 相对零样本提升 {rel:+.1%}"
                  f"（要求 ≥20%：{'达标' if rel >= 0.20 else '未达标'}）")
            rel_tail = s["长尾_recall@prev"] / (0.1408) - 1
            print(f"长尾召回率相对第 2 周零样本基线（0.1408）提升 {rel_tail:+.1%}"
                  f"（要求 ≥12%：{'达标' if rel_tail >= 0.12 else '未达标'}）")

        df.to_csv(RESULTS / f"lora_eval_{tag}.csv", index=False)
        run.artifact(RESULTS / f"lora_eval_{tag}.csv")
        run.note(f"LoRA 评测：50 类 mAP {s['全部_AP']:.4f}，长尾 AP {s['长尾_AP']:.4f}，"
                 f"长尾召回 {s['长尾_recall@prev']:.4f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
