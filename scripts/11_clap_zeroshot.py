"""第 1 周双赛道基线之二：CLAP 在 MTT 上的零样本标签 mAP。

做法：不训练，直接把 50 类标签写成文本提示，取 CLAP 的文本嵌入与音频嵌入算相似度，
按相似度排序后对每个标签算 AP，再对 50 类取平均得到 mAP。

文本提示沿用 LAION CLAP 的做法："This is a sound of {tag}"。

用法：
    python scripts/11_clap_zeroshot.py --limit 500      # 先用 500 条估个值
    python scripts/11_clap_zeroshot.py --limit 0        # 全量测试集

音频嵌入会缓存到 results/clap_emb_*.npy，重复运行不用重算。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import ClapModel, ClapProcessor

ROOT = Path(__file__).resolve().parents[1]
MTT = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
REPO = "laion/clap-htsat-unfused"
TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
    "A song with {}",
    "Music that is {}",
]
SR = 48000
CLIP_SAMPLES = 10 * SR  # CLAP 的输入上限是 10 秒


def load_audio(path: Path) -> np.ndarray | None:
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    !! 读不了 {path.name}: {type(exc).__name__}")
        return None
    return audio


def windows(audio: np.ndarray, n: int) -> list[np.ndarray]:
    """把整段音频切成 n 个不重叠窗口；n<=0 表示按 10 秒铺满。"""
    if n <= 0:
        n = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
    n = min(n, max(1, int(np.ceil(len(audio) / CLIP_SAMPLES))))
    edges = np.linspace(0, len(audio), n + 1).astype(int)
    return [audio[edges[i]:edges[i + 1]] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="用前 N 条测试片段；0 表示全部")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="用哪个划分：test 为官方测试集（对外可比），val 用于选配置，all 为全部 25863 条",
    )
    ap.add_argument("--model", default=REPO, help="CLAP 仓库名")
    ap.add_argument("--windows", type=int, default=1, help="每条音频取几个窗口取平均；0 表示按 10 秒铺满")
    ap.add_argument("--templates", type=int, default=1, help="使用前 N 个提示模板做集成")
    args = ap.parse_args()

    tags_file = RESULTS / "mtt_50tags.csv"
    if not tags_file.exists():
        print(f"没找到 {tags_file}，先跑 04_build_taglist.py")
        return 1
    tags = pd.read_csv(tags_file)["tag"].tolist()

    anno = pd.read_csv(MTT / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    if args.split == "all":
        test = anno.reset_index(drop=True)
    else:
        ids = pd.read_csv(
            MTT / f"{args.split}_gt_mtt.tsv", sep="\t", header=None, usecols=[0]
        )[0].astype(str)
        test = anno[anno["clip_id"].isin(set(ids))].reset_index(drop=True)
    if args.limit > 0:
        test = test.head(args.limit)
    print(f"MTT 划分 {args.split}：{len(test)} 条片段，标签 {len(tags)} 类")

    labels = test[tags].astype(int).to_numpy()
    print(f"每个标签正样本数：最少 {labels.sum(0).min()}，最多 {labels.sum(0).max()}")

    print(f"\n加载 CLAP（{args.model}）...")
    processor = ClapProcessor.from_pretrained(args.model)
    model = ClapModel.from_pretrained(args.model)
    model.eval()

    used = TEMPLATES[: max(1, args.templates)]
    text_batch = [tpl.format(tag) for tpl in used for tag in tags]
    with torch.no_grad():
        text_inputs = processor(text=text_batch, return_tensors="pt", padding=True)
        emb = model.get_text_features(**text_inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_emb = emb.view(len(used), len(tags), -1).mean(0)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    print(f"文本嵌入就绪 {tuple(text_emb.shape)}，模板数 {len(used)}")

    tag = args.model.split("/")[-1]
    cache = RESULTS / f"clap_emb_{tag}_{args.split}{len(test)}_w{args.windows}.npy"
    if cache.exists():
        audio_emb = np.load(cache)
        print(f"音频嵌入用缓存 {cache.name} {audio_emb.shape}")
    else:
        vectors = []
        t0 = time.time()
        for i, row in enumerate(test.itertuples(), 1):
            wav = load_audio(MTT / row.mp3_path)
            if wav is None:
                vectors.append(np.zeros(text_emb.shape[-1], dtype=np.float32))
                continue
            with torch.no_grad():
                parts = []
                for chunk in windows(wav, args.windows):
                    feats = processor(audios=chunk, sampling_rate=SR, return_tensors="pt")
                    e = model.get_audio_features(**feats)
                    parts.append(e / e.norm(dim=-1, keepdim=True))
                emb = torch.stack(parts).mean(0)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            vectors.append(emb.squeeze(0).numpy().astype(np.float32))
            if i % 50 == 0 or i == len(test):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(test)}  平均 {rate:.2f}s/条，预计剩余 {rate * (len(test) - i) / 60:.1f} 分钟")
        audio_emb = np.stack(vectors)
        np.save(cache, audio_emb)

    scores = audio_emb @ text_emb.numpy().T          # (n_clip, n_tag)
    aps = {
        tag: float(average_precision_score(labels[:, i], scores[:, i]))
        for i, tag in enumerate(tags)
    }
    mAP = float(np.mean(list(aps.values())))

    out = pd.DataFrame(
        sorted(((t, a, int(labels[:, i].sum())) for i, (t, a) in enumerate(aps.items())),
               key=lambda x: -x[1]),
        columns=["tag", "AP", "positives"],
    )
    suffix = f"_{tag}_{args.split}_w{args.windows}_t{len(used)}"
    suffix += f"_limit{args.limit}" if args.limit > 0 else ""
    out.to_csv(RESULTS / f"clap_zeroshot_ap{suffix}.csv", index=False)

    print(f"\n=== CLAP 零样本 50 类标签 mAP = {mAP:.4f} ===")
    print("（大纲第 1 周的口径是 mAP ≥ 0.22）\n")
    print("AP 最高的 10 类：")
    for r in out.head(10).itertuples():
        print(f"  {r.AP:.3f}  {r.tag:16} 正样本 {r.positives}")
    print("AP 最低的 10 类：")
    for r in out.tail(10).itertuples():
        print(f"  {r.AP:.3f}  {r.tag:16} 正样本 {r.positives}")
    print(f"\n明细 -> results/clap_zeroshot_ap{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
