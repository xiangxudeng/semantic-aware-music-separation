"""CLAP 语义特征提取模块的自检（第 2 周 P0 验收项之一："特征提取可用"）。

跑一遍下面这些，确认模块能直接用：

1. 全曲级向量：形状 (512,)、已归一化
2. 可复现：同一个文件编两次结果完全一致
3. 片段级向量：按 10 秒窗口切，条数与时长对得上
4. 文本向量：50 类标签的提示词
5. 定性检查：拿"钢琴"标注的音轨去查，相似度最高的标签里应该有 piano
6. 耗时统计：读音频 / 模型前向 / 平均每片段

用法：
    python scripts/31_clap_encoder_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data import load_mtt, load_tags
from src.semantic.clap_encoder import EMBED_DIM, ClapEncoder

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"    {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    print("CLAP 语义特征提取模块自检", flush=True)
    t_all = time.time()

    tags = load_tags()
    mtt = load_mtt(split="test")
    print(f"标签 {len(tags)} 类，MTT 测试片段 {len(mtt)} 条", flush=True)

    print("\n=== 加载编码器 ===", flush=True)
    t0 = time.time()
    enc = ClapEncoder(device="auto", windows=3)
    print(f"  加载耗时 {time.time() - t0:.1f} 秒（设备 {enc.device}）", flush=True)

    # ---------- 1. 全曲级 ----------
    print("\n=== 1. 全曲级向量 ===", flush=True)
    sample = mtt.head(3)
    paths = [str(ROOT / "data" / "mtt" / p) for p in sample["mp3_path"]]
    vecs = enc.encode_files(paths)
    check(vecs.shape == (3, EMBED_DIM), f"3 条音频 → 形状 (3, {EMBED_DIM})", str(vecs.shape))
    norms = np.linalg.norm(vecs, axis=1)
    check(np.allclose(norms, 1.0, atol=1e-4), "每条向量都已 L2 归一化",
          f"范数 {np.round(norms, 4).tolist()}")

    # ---------- 2. 可复现 ----------
    print("\n=== 2. 可复现性 ===", flush=True)
    again = enc.encode_file(paths[0])
    check(np.array_equal(vecs[0], again), "同一个文件编码两次结果完全相同",
          f"max|Δ|={float(np.abs(vecs[0] - again).max()):.2e}")

    # ---------- 3. 片段级 ----------
    print("\n=== 3. 片段级向量 ===", flush=True)
    import librosa

    dur = librosa.get_duration(path=paths[0])
    clips = enc.encode_clips(paths[0], clip_seconds=10.0)
    expect = max(1, int(np.ceil(dur / 10.0)))
    check(clips.shape == (expect, EMBED_DIM),
          f"时长 {dur:.1f}s → {expect} 个 10 秒片段，形状 {clips.shape}",
          f"实际 {clips.shape[0]} 个")
    check(np.allclose(np.linalg.norm(clips, axis=1), 1.0, atol=1e-4), "片段向量也都归一化")

    # ---------- 4. 文本 ----------
    print("\n=== 4. 文本向量 ===", flush=True)
    text_emb = enc.tag_prompts(tags, templates=["This is a sound of {}", "This audio contains {}"])
    check(text_emb.shape == (len(tags), EMBED_DIM), "50 类标签文本向量形状正确", str(text_emb.shape))

    # ---------- 5. 定性检查 ----------
    print("\n=== 5. 定性检查：钢琴 / 吉他 ===", flush=True)
    for target in ("piano", "guitar"):
        idx = mtt[mtt[target] == 1].head(6)
        sub = enc.encode_files([str(ROOT / "data" / "mtt" / p) for p in idx["mp3_path"]])
        sim = ClapEncoder.similarity(sub.mean(0, keepdims=True), text_emb)[0]
        order = np.argsort(-sim)
        top5 = [(tags[i], round(float(sim[i]), 3)) for i in order[:5]]
        rank = int(np.where(order == tags.index(target))[0][0]) + 1
        print(f"    标注「{target}」的 6 首，相似度前 5：{top5}", flush=True)
        # 零样本在 MTT 上本来就弱（50 类 mAP 只有 0.27），单个标签能进前 1/3 就说明特征是有区分度的
        check(rank <= len(tags) // 3, f"「{target}」的排名进入前 1/3（{len(tags) // 3} 名）",
              f"实际第 {rank} 名")

    # ---------- 6. 耗时 ----------
    print("\n=== 6. 耗时统计 ===", flush=True)
    print(enc.report(), flush=True)
    check(enc.stats.n_clips > 0 and enc.stats.forward_seconds > 0, "统计里有前向耗时记录")
    check(enc.stats.failures == 0, "本轮没有读不了的文件", f"{enc.stats.failures} 个")

    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{'=' * 60}")
    print(f"共 {len(RESULTS)} 项检查，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项"
          f"；总耗时 {time.time() - t_all:.1f} 秒")
    for label in failed:
        print(f"  失败：{label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
