"""诊断：模型到底有没有在看音频？

描述生成的 BLEU 比随机基线还低，最可能的解释是**模型忽略了音频 token**，
只是顺着 prompt 续写出"听起来像音乐描述"的话。

判定方法很直接：固定 prompt，换不同音频，看输出是否变化。
- 输出几乎一模一样 → 音频被忽略，caption 任务根本没有落地
- 输出随音频明显不同 → 音频起了作用，问题在别处（比如描述风格不匹配）

用法：
    python scripts/65_probe_audio_grounding.py --adapter results/lora_lora1
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from src.utils.run_log import run_record

DATA = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
PROMPT = "用一句话描述这段音乐，提到你能听出的风格、乐器和情绪。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    adapter = ROOT / args.adapter if not Path(args.adapter).is_absolute() else Path(args.adapter)
    with run_record("w3_probe_grounding", config=vars(args)) as run:
        evalmod = import_module("63_eval_lora")
        model, tok = evalmod.load_model(adapter)
        device = next(model.parameters()).device

        anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
        anno["clip_id"] = anno["clip_id"].astype(str)
        fp16 = RESULTS / "clap_official_emb_all25863_w3_fp16.npy"
        emb = np.load(fp16 if fp16.exists() else RESULTS / "clap_official_emb_all25863_w3.npy").astype(np.float32)
        test_ids = pd.read_csv(DATA / "test_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str)
        te = np.flatnonzero(anno["clip_id"].isin(set(test_ids)).to_numpy())[: args.n]
        tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
        lab = anno[tags].astype(int).to_numpy()

        prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                         tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        outs = []
        for i in te:
            audio = torch.from_numpy(emb[i][None]).to(device)
            with torch.no_grad():
                gen = model.generate_with_audio(audio, enc["input_ids"], enc["attention_mask"],
                                                max_new_tokens=40, do_sample=False,
                                                pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            positive = [t for j, t in enumerate(tags) if lab[i, j] == 1]
            outs.append(text)
            print(f"\n[{anno['clip_id'].iloc[i]}] 真实标签：{', '.join(positive[:8])}", flush=True)
            print(f"  生成：{text}", flush=True)

        uniq = len(set(outs))
        print(f"\n{len(outs)} 个不同音频 → {uniq} 种不同输出")
        if uniq <= max(1, len(outs) // 2):
            verdict = "音频基本被忽略（输出高度雷同）"
        else:
            verdict = "输出随音频变化，音频起了作用"
        print(f"判定：{verdict}")
        run.note(f"音频接地诊断：{len(outs)} 个音频 → {uniq} 种输出；{verdict}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
