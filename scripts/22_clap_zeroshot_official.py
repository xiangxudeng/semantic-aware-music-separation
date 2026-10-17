"""零样本标签 mAP（官方 LAION-CLAP 实现）。

为什么要有这个脚本：第 1 周用的是 HuggingFace `transformers` 里的移植版
（`laion/clap-htsat-unfused` + `ClapModel/ClapProcessor`）。第 2 周排查
"零样本绝对值偏低"时做了对照实验，发现两者产出的嵌入根本不在同一个空间：

    同一条音频，两个实现的嵌入余弦相似度        0.29
    同一个标签文本，两个实现的嵌入余弦相似度    0.33

在 500 条 MTT test 片段上（三窗口 + 四模板）：
    transformers 移植版   0.2163
    官方 laion_clap       0.3136

所以正式口径改用官方实现。本脚本与 11_clap_zeroshot.py 的参数含义一致，
方便直接对照。

安装：pip install laion-clap==1.1.7（会自动下 630k-audioset-best.pt 到包里）

用法：
    python scripts/22_clap_zeroshot_official.py --split val  --windows 3 --templates 4
    python scripts/22_clap_zeroshot_official.py --split test --windows 3 --templates 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# librosa 解 mp3 需要 ffmpeg 在 PATH 上；沙箱里可能被剥掉，兜一次
_FFMPEG_FALLBACK = r"/usr/bin"
if Path(_FFMPEG_FALLBACK, "ffmpeg.exe").exists():
    os.environ["PATH"] = _FFMPEG_FALLBACK + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parents[1]

# laion_clap 在 import 时会去下 bert-base-uncased（内部用）与 roberta-base（文本分词器）。
# 国内网络这一步经常失败，third_party 里放好了这两个分词器的小文件，优先本地命中。
_THIRD_PARTY = ROOT / "third_party"
if (_THIRD_PARTY / "bert-base-uncased" / "vocab.txt").exists():
    os.chdir(_THIRD_PARTY)

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MTT = ROOT / "data" / "mtt"
RESULTS = ROOT / "results"
TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
    "A song with {}",
    "Music that is {}",
]


def l2(x: np.ndarray) -> np.ndarray:
    # 读不了的文件用零向量占位，归一化时要兜住 0 除，否则会算出 NaN 把后面的指标全带崩
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def embed(model, files: list[str], chunk: int, dim: int = 512) -> np.ndarray:
    """分批算音频嵌入；遇到读不了的文件逐条降级，不让整轮任务挂掉。

    MTT 里有个别 mp3 是坏的（soundfile 与 audioread 都会报 EOFError），
    直接抛异常会让跑了几个小时的任务白费。
    """
    parts = []
    bad: list[str] = []
    for s in range(0, len(files), chunk):
        block = files[s : s + chunk]
        try:
            parts.append(model.get_audio_embedding_from_filelist(x=block, use_tensor=False))
        except Exception:  # noqa: BLE001
            rows = []
            for f in block:
                try:
                    rows.append(model.get_audio_embedding_from_filelist(x=[f], use_tensor=False)[0])
                except Exception as exc:  # noqa: BLE001
                    print(f"    !! 读不了 {Path(f).name}: {type(exc).__name__}", flush=True)
                    bad.append(Path(f).name)
                    rows.append(np.zeros(dim, dtype=np.float32))
            parts.append(np.stack(rows))
        done = min(s + chunk, len(files))
        print(f"    {done}/{len(files)}", flush=True)
    if bad:
        print(f"  共 {len(bad)} 条读不了，已用零向量占位：{bad}", flush=True)
    return np.concatenate(parts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条；0 表示该划分全部")
    ap.add_argument("--windows", type=int, default=3, help="每条音频取几个 10 秒窗口取平均")
    ap.add_argument("--templates", type=int, default=4, help="用前 N 个提示模板做集成")
    ap.add_argument("--chunk", type=int, default=100, help="一批喂多少条，太大在 CPU 上会顶爆内存")
    args = ap.parse_args()

    tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
    anno = pd.read_csv(MTT / "annotations_final.csv", sep="\t")
    anno["clip_id"] = anno["clip_id"].astype(str)
    if args.split == "all":
        clips = anno.reset_index(drop=True)
    else:
        ids = set(
            pd.read_csv(MTT / f"{args.split}_gt_mtt.tsv", sep="\t", header=None, usecols=[0])[0].astype(str)
        )
        clips = anno[anno.clip_id.isin(ids)].reset_index(drop=True)
    if args.limit > 0:
        clips = clips.head(args.limit)
    labels = clips[tags].astype(int).to_numpy()
    files = [str(MTT / p) for p in clips["mp3_path"]]
    print(f"MTT {args.split}：{len(files)} 条 × {len(tags)} 类标签", flush=True)

    import laion_clap

    print("加载官方模型（enable_fusion=False）...", flush=True)
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(verbose=False)

    cache = RESULTS / f"clap_official_emb_{args.split}{len(files)}_w{args.windows}.npy"
    if cache.exists():
        audio_emb = np.load(cache)
        print(f"音频嵌入用缓存 {cache.name} {audio_emb.shape}", flush=True)
    else:
        t0 = time.time()
        parts = []
        for w in range(args.windows):
            np.random.seed(w)  # 固定住每个窗口的随机裁剪位置，结果可复现
            print(f"  窗口 {w + 1}/{args.windows}", flush=True)
            one = embed(model, files, args.chunk)
            rate = (time.time() - t0) / ((w + 1) * len(files))
            left = rate * (args.windows - w - 1) * len(files) / 60
            print(f"  窗口 {w + 1} 完成，均 {rate:.2f}s/条，剩余 {left:.1f} 分钟", flush=True)
            parts.append(l2(one))
        audio_emb = l2(np.mean(parts, axis=0))
        np.save(cache, audio_emb)

    text_per_tpl = []
    for tpl in TEMPLATES[: args.templates]:
        text_per_tpl.append(l2(model.get_text_embedding([tpl.format(t) for t in tags], use_tensor=False)))
    text_emb = l2(np.mean(text_per_tpl, axis=0))

    scores = audio_emb @ text_emb.T
    aps = {
        tag: float(average_precision_score(labels[:, i], scores[:, i])) for i, tag in enumerate(tags)
    }
    aucs = {tag: float(roc_auc_score(labels[:, i], scores[:, i])) for i, tag in enumerate(tags)}
    mAP = float(np.mean(list(aps.values())))

    suffix = f"official_{args.split}{len(files)}_w{args.windows}_t{args.templates}"
    out = pd.DataFrame(
        sorted(
            (
                (t, aps[t], aucs[t], int(labels[:, i].sum()))
                for i, t in enumerate(tags)
            ),
            key=lambda x: -x[1],
        ),
        columns=["tag", "AP", "AUC", "positives"],
    )
    out.to_csv(RESULTS / f"clap_zeroshot_ap_{suffix}.csv", index=False)

    print(f"\n=== 官方实现 CLAP 零样本 50 类标签 mAP = {mAP:.4f}（平均 AUC {np.mean(list(aucs.values())):.4f}）===")
    print("AP 最高的 8 类：")
    for r in out.head(8).itertuples():
        print(f"  {r.AP:.3f}  {r.tag:16} 正样本 {r.positives}")
    print("AP 最低的 8 类：")
    for r in out.tail(8).itertuples():
        print(f"  {r.AP:.3f}  {r.tag:16} 正样本 {r.positives}")
    print(f"\n明细 -> results/clap_zeroshot_ap_{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
