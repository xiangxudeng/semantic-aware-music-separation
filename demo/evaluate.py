"""真实缓存上的检索与同子集零样本长尾基线，评测结果含完整口径。"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from music import MusicSystem, TEMPLATES, normalize
from analysis_stats import TAIL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "模板检索",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    s = MusicSystem(a.assets)
    start = time.perf_counter()
    e = s._encoder()
    vectors = normalize(
        np.mean([e.text([t.format(tag) for tag in s.tags]) for t in TEMPLATES], axis=0)
    )
    anno = pd.read_csv(
        s.assets / "data" / "mtt" / "annotations_final.csv",
        sep="\t",
        dtype={"clip_id": str},
    )
    test_ids = set(
        pd.read_csv(
            s.assets / "data" / "mtt" / "test_gt_mtt.tsv",
            sep="\t",
            header=None,
            dtype=str,
        )[0]
    )
    train_ids = set(
        pd.read_csv(
            s.assets / "data" / "mtt" / "train_gt_mtt.tsv",
            sep="\t",
            header=None,
            dtype=str,
        )[0]
    )
    test = np.flatnonzero(anno.clip_id.isin(test_ids))
    train = np.flatnonzero(anno.clip_id.isin(train_ids))
    y = anno[s.tags].to_numpy(int)
    # 同 63_eval_lora：按 annotations 的原始行序取前 800 首，训练频率定预测比例。
    subset = test[:800]
    scores = normalize(s.embeddings[subset]) @ vectors.T
    prevalence = y[train].mean(0)
    rows = []
    for i, tag in enumerate(s.tags):
        positives = int(y[subset, i].sum())
        k = max(1, int(round(len(subset) * prevalence[i])))
        top = np.argsort(-scores[:, i], kind="stable")[:k]
        recall = (
            float(y[subset[top], i].sum() / positives) if positives else float("nan")
        )
        rows.append(
            dict(
                tag=tag,
                AP=float(average_precision_score(y[subset, i], scores[:, i]))
                if positives
                else float("nan"),
                recall=recall,
                positives=positives,
                predicted_positive_count=k,
            )
        )
    baseline = pd.DataFrame(rows)
    baseline.to_csv(a.output / "zeroshot_same800.csv", index=False)
    # 用测试库标签作弱相关性标注；不是 MusicCaps 自然语言人工检索基准。
    queries = [f"A music piece featuring {tag}" for tag in s.tags]
    query = e.text(queries)
    matrix = normalize(s.embeddings[test]) @ query.T
    retrieval = []
    for i, tag in enumerate(s.tags):
        ids = np.argsort(-matrix[:, i], kind="stable")[:5]
        hits = int(y[test[ids], i].sum())
        retrieval.append(
            dict(
                query=queries[i],
                label=tag,
                gallery_size=len(test),
                hits_at5=hits,
                precision_at5=hits / 5,
                hit_at5=int(hits > 0),
                clip_ids=";".join(anno.iloc[test[ids]].clip_id.tolist()),
            )
        )
    pd.DataFrame(retrieval).to_csv(a.output / "retrieval_queries.csv", index=False)
    meta = dict(
        model="LAION CLAP HTSAT unfused model_id=1",
        audio_windows=3,
        query_count=50,
        gallery_clips=len(test),
        gallery_split="MTT test",
        relevance="查询标签的 MTT 二值弱标注",
        precision_at5=float(np.mean([x["precision_at5"] for x in retrieval])),
        hit_at5=float(np.mean([x["hit_at5"] for x in retrieval])),
        baseline800_map=float(baseline.AP.mean()),
        baseline800_tail_recall=float(baseline[baseline.tag.isin(TAIL)].recall.mean()),
        elapsed_seconds=round(time.perf_counter() - start, 3),
        limitations=[
            "模板查询代表固定标签检索；自由组合描述、中文和主观风格需另作人工评测",
            "标注缺失可能导致相关音频被视为不相关",
            "评测未用测试标签调参；所有50类查询均报告",
        ],
    )
    (a.output / "evaluation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
