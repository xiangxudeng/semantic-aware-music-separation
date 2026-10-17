"""第 3 周：描述生成的 BLEU 评测 + 人工抽检。

大纲的验收标准里有"文本生成 BLEU 值提升 ≥15%"。这里给两个数：

1. **LoRA 微调后**在 LP-MusicCaps 测试集（300 首 × 4 条参考描述）上的 BLEU-4
2. **随机基线**：从训练集里随机抽一条别人的描述当答案——这是"完全没看图随便说"的下限，
   用来判断 BLEU 是否真的有意义

同时打印 10 条"生成 vs 参考"的对照，供人工抽检（大纲还要求人工评测 100 条样本，
先做 10 条的快速抽检）。

用法（云端 GPU）：
    python scripts/64_eval_caption.py --adapter results/lora_lora1 --n-samples 300
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.run_log import run_record

DATA = ROOT / "data" / "mtt"
LP = ROOT / "data" / "lpmusiccaps"
RESULTS = ROOT / "results"
PROMPT = "用一句话描述这段音乐，提到你能听出的风格、乐器和情绪。"


def load_lp_test() -> dict[str, list[str]]:
    """LP 的 test 划分（300 首，正好落在 MTT 测试集里）→ clip_id: 参考描述列表。

    优先读预先导出的小 JSON（236 KB）：原始 parquet 两个多 GB（含音频），
    传到云服务器既慢又容易中途断开，而评测只需要参考描述文本。
    """
    small = RESULTS / "lp_test_captions.json"
    if small.exists():
        return json.loads(small.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(LP / "test-*.parquet"))):
        d = pd.read_parquet(f, columns=["track_id", "texts"])
        for tid, texts in zip(d["track_id"], d["texts"]):
            out[str(tid)] = [str(t) for t in texts]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    adapter = ROOT / args.adapter if not Path(args.adapter).is_absolute() else Path(args.adapter)
    tag = args.tag or adapter.name

    with run_record(f"w3_eval_caption_{tag}", config=vars(args)) as run:
        import sacrebleu

        sys.path.insert(0, str(ROOT / "scripts"))
        from importlib import import_module

        evalmod = import_module("63_eval_lora")
        model, tok = evalmod.load_model(adapter)
        device = next(model.parameters()).device

        anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
        anno["clip_id"] = anno["clip_id"].astype(str)
        row_of = {c: i for i, c in enumerate(anno["clip_id"])}
        fp16 = RESULTS / "clap_official_emb_all25863_w3_fp16.npy"
        emb = np.load(fp16 if fp16.exists() else RESULTS / "clap_official_emb_all25863_w3.npy")
        emb = emb.astype(np.float32)

        refs = load_lp_test()
        clip_ids = [c for c in refs if c in row_of]
        random.seed(0)
        clip_ids = random.sample(clip_ids, min(args.n_samples, len(clip_ids)))
        print(f"评测 {len(clip_ids)} 首，每首 4 条参考描述", flush=True)

        hyps, all_refs, samples = [], [], []
        for i, cid in enumerate(clip_ids, 1):
            prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            audio = torch.from_numpy(emb[row_of[cid]][None]).to(device)
            with torch.no_grad():
                gen = model.generate_with_audio(audio, enc["input_ids"], enc["attention_mask"],
                                                max_new_tokens=args.max_new, do_sample=False,
                                                pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            hyps.append(text)
            all_refs.append(refs[cid])
            if len(samples) < 10:
                samples.append((cid, text, refs[cid][0]))
            if i % 50 == 0 or i == 1:
                print(f"  {i}/{len(clip_ids)}  例：{text[:60]}", flush=True)

        bleu = sacrebleu.corpus_bleu(hyps, list(zip(*all_refs)))
        # 随机基线：拿别人的描述当答案，代表"完全没听"的下限
        pool = [r[0] for r in all_refs]
        rnd = [pool[(i + 37) % len(pool)] for i in range(len(hyps))]
        bleu_rnd = sacrebleu.corpus_bleu(rnd, list(zip(*all_refs)))

        print(f"\n=== BLEU-4 ===")
        print(f"  LoRA 微调    {bleu.score:.2f}   （1/2/3/4-gram: "
              f"{bleu.precisions[0]:.1f}/{bleu.precisions[1]:.1f}/{bleu.precisions[2]:.1f}/{bleu.precisions[3]:.1f}，"
              f"BP {bleu.bp:.3f}）")
        print(f"  随机基线     {bleu_rnd.score:.2f}")
        print(f"  相对随机基线 提升 {(bleu.score / max(bleu_rnd.score, 1e-6) - 1) * 100:.1f}%")

        print("\n=== 人工抽检（生成 vs 参考）===")
        for cid, hyp, ref in samples:
            print(f"  [{cid}] 生成：{hyp[:110]}")
            print(f"           参考：{ref[:110]}")

        out = RESULTS / f"caption_eval_{tag}.json"
        out.write_text(json.dumps({
            "bleu4": bleu.score, "bleu4_random": bleu_rnd.score,
            "n": len(hyps), "clip_ids": clip_ids, "hyps": hyps, "refs": all_refs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        run.artifact(out)
        run.note(f"描述生成 BLEU-4 {bleu.score:.2f}（随机基线 {bleu_rnd.score:.2f}，"
                 f"提升 {(bleu.score / max(bleu_rnd.score, 1e-6) - 1) * 100:.1f}%）")
        print(f"\n明细 -> {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
