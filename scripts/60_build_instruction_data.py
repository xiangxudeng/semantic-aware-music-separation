"""第 3 周第 4 步：构建跨模态指令数据集（≥1 万条，三类任务）。

数据来源
--------
- **LP-MusicCaps-MTT**：3,000 条 MTT 片段，每条 4 句**真实的音乐描述**（训练划分，
  已确认没有一条落在 MTT 测试集里）
- **MTT 官方标签**：同一批片段的 50 类标签（来自 annotations_final.csv）

三类指令
--------
| 任务 | 提问 | 回答 | 作用 |
|---|---|---|---|
| `tagging` 标签推理 | 给出风格/乐器/情绪/人声标签 | 该片段的真实标签 | 对应 mAP 评测口径 |
| `caption` 描述生成 | 用一句话描述这段音乐 | LP-MusicCaps 的真实描述 | 对应 BLEU 评测口径 |
| `qa` 问答 | 这段音乐里有 {标签} 吗？ | 有 / 没有 | 训练"是/否"判断，含负样本 |

为什么要有负样本
----------------
`qa` 任务里约一半的问句是**该片段没有的标签**。如果只问正标签，模型会退化成"一律回答有"，
这在评测里看不出来，但实际演示时立刻露馅。

划分
----
按**曲目**划分训练/验证（同一首歌的所有指令只进一边），避免同曲泄漏导致验证集虚高。

用法：
    python scripts/60_build_instruction_data.py
"""

from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.run_log import run_record

DATA = ROOT / "data" / "mtt"
LP = ROOT / "data" / "lpmusiccaps"
RESULTS = ROOT / "results"
QA_NEG_PER_POS = 1  # 每个正标签配一个负标签问句
VAL_RATIO = 0.1
SEED = 0

TAGGING_PROMPTS = [
    "听这段音乐，给出它的风格、乐器、情绪与人声标签。",
    "这段音频可以用哪些标签描述？",
    "请为这段音乐打标签。",
]
CAPTION_PROMPTS = [
    "用一句话描述这段音乐。",
    "这段音乐听起来是什么样的？请描述。",
    "为这段音频写一句简短的介绍。",
]
QA_PROMPTS = [
    "这段音乐里有{tag}吗？",
    "这段音乐是{tag}风格吗？",
    "这段音频包含{tag}元素吗？",
]


def main() -> int:
    random.seed(SEED)
    with run_record("w3_build_instruction_data",
                    config={"qa_neg_per_pos": QA_NEG_PER_POS, "val_ratio": VAL_RATIO, "seed": SEED}) as run:
        tags = pd.read_csv(RESULTS / "mtt_50tags.csv")["tag"].tolist()
        anno = pd.read_csv(DATA / "annotations_final.csv", sep="\t")
        anno["clip_id"] = anno["clip_id"].astype(str)
        # 用矩阵取值，不要用 itertuples 的属性名——标签里有 "new age" 这类带空格的，
        # pandas 会把列名改写成位置名，取属性会直接报 AttributeError
        matrix = anno.set_index("clip_id")[tags].astype(int)
        tag_of = {cid: [t for t in tags if row[t] == 1] for cid, row in matrix.iterrows()}
        path_of = dict(zip(anno["clip_id"], anno["mp3_path"]))

        # 只用 LP 的 train 划分（它的 test 划分正好是 MTT 测试集，不能进训练数据）
        caps: dict[str, list[str]] = {}
        for f in sorted(glob.glob(str(LP / "train-*.parquet"))):
            d = pd.read_parquet(f, columns=["track_id", "texts"])
            for tid, texts in zip(d["track_id"], d["texts"]):
                caps.setdefault(str(tid), []).extend(str(t) for t in texts)
        print(f"描述数据：{len(caps)} 首曲子，共 {sum(len(v) for v in caps.values())} 条描述")

        rows = []
        for cid, texts in caps.items():
            if cid not in path_of:
                continue
            audio = str(DATA / path_of[cid])
            positive = tag_of.get(cid, [])
            # ① 标签推理：正样本用真实标签，负样本从"该片段没有的标签"里抽
            if positive:
                rows.append({"clip_id": cid, "audio": audio, "task": "tagging",
                             "prompt": random.choice(TAGGING_PROMPTS),
                             "answer": "、".join(positive)})
            # ② 描述生成：每条真实描述一条样本
            for t in texts:
                rows.append({"clip_id": cid, "audio": audio, "task": "caption",
                             "prompt": random.choice(CAPTION_PROMPTS), "answer": t})
            # ③ 问答：每个正标签一条"有"，再配同数量的"没有"
            negatives = [t for t in tags if t not in positive]
            for t in positive:
                rows.append({"clip_id": cid, "audio": audio, "task": "qa",
                             "prompt": random.choice(QA_PROMPTS).format(tag=t), "answer": "有"})
            for t in random.sample(negatives, min(len(positive) * QA_NEG_PER_POS, len(negatives))):
                rows.append({"clip_id": cid, "audio": audio, "task": "qa",
                             "prompt": random.choice(QA_PROMPTS).format(tag=t), "answer": "没有"})

        random.shuffle(rows)
        n = len(rows)
        by_task = pd.Series([r["task"] for r in rows]).value_counts().to_dict()
        print(f"共生成 {n} 条指令：{by_task}")

        # 按曲目划分，同一首歌不跨边
        clips = sorted({r["clip_id"] for r in rows})
        val_clips = set(random.sample(clips, max(1, int(len(clips) * VAL_RATIO))))
        train = [r for r in rows if r["clip_id"] not in val_clips]
        val = [r for r in rows if r["clip_id"] in val_clips]
        print(f"划分：训练 {len(train)} 条（{len(clips) - len(val_clips)} 首）/ "
              f"验证 {len(val)} 条（{len(val_clips)} 首）")

        for name, data in (("train", train), ("val", val)):
            out = RESULTS / f"instruction_{name}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {out.name}: {len(data)} 条")
            run.artifact(out)

        print("\n=== 质量抽检（随机 8 条）===")
        for r in random.sample(rows, 8):
            print(f"  [{r['task']:7}] Q: {r['prompt'][:46]}")
            print(f"            A: {r['answer'][:70]}")

        stats = {"总条数": n, "按任务": by_task,
                 "训练条数": len(train), "验证条数": len(val),
                 "曲目数": len(clips), "验证曲目数": len(val_clips),
                 "唯一音频数": len({r["audio"] for r in rows})}
        (RESULTS / "instruction_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = n >= 10000
        print(f"\n是否达到计划的 ≥1 万条：{'是' if ok else '否'}（{n}）")
        run.note(f"指令数据 {n} 条（{by_task}），划分 {len(train)}/{len(val)}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
