"""按真实配对单位复核统计；原始结果只读，新结果写入独立输出目录。"""

from pathlib import Path
import itertools
import json
import os
import numpy as np
import pandas as pd

TAIL = [
    "country",
    "jazz",
    "baroque",
    "folk",
    "hard rock",
    "modern",
    "eastern",
    "trance",
    "cello",
    "bass",
    "upbeat",
    "heavy",
    "jazzy",
    "instrumental",
]
STEMS = ["vocals", "drums", "bass", "other"]


def paired_test(a, b, repeats=10000, seed=20261007):
    """a-b 均值的 percentile bootstrap CI 与配对符号置换检验。

    H0 为配对差可交换符号；小样本穷举，大样本随机置换加一校正。
    置信区间与 p 值采用不同分布，分别回答估计精度与零假设检验。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("配对数组必须等长且为一维")
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    if len(d) == 0:
        raise ValueError("没有有效配对")
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(repeats, len(d)))].mean(1)
    observed = abs(d.mean())
    exact = len(d) <= 15
    signs = (
        np.array(list(itertools.product([-1, 1], repeat=len(d))))
        if exact
        else rng.choice([-1, 1], size=(repeats, len(d)))
    )
    null = (signs * d).mean(1)
    count = np.count_nonzero(np.abs(null) >= observed - 1e-12)
    p = count / len(null) if exact else (count + 1) / (len(null) + 1)
    return dict(
        n_pairs=len(d),
        delta=float(d.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        p_value=float(p),
        test="exact paired sign permutation"
        if exact
        else "random paired sign permutation",
        bootstrap_repeats=repeats,
        permutation_repeats=len(null),
        statistics_seed=seed,
    )


def holm(p):
    """Holm 家族错误率校正；只在同一预定义比较族内使用。"""
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    return out


def audit(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    r = source / "results"
    rows = []
    for rank in ["base", "r32"]:
        for seed in [1, 2, 3]:
            df = pd.read_csv(r / f"lora_eval_seed{seed}_{rank}.csv").set_index("tag")
            if not set(TAIL) <= set(df.index):
                raise ValueError("长尾标签不完整")
            rows.append(
                dict(
                    config="r16" if rank == "base" else "r32",
                    training_seed=seed,
                    test_clips=800,
                    epochs=1,
                    tail_classes=14,
                    tail_positive_pairs=int(df.loc[TAIL, "positives"].sum()),
                    map=float(df.AP.mean()),
                    tail_recall=float(df.loc[TAIL, "recall@prevalence"].mean()),
                    tail_map=float(df.loc[TAIL, "AP"].mean()),
                )
            )
    seeds = pd.DataFrame(rows)
    seeds.to_csv(output / "longtail_seeds.csv", index=False)
    summary = (
        seeds.groupby("config")[["map", "tail_recall", "tail_map"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["config"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    summary["training_seeds"] = 3
    summary["test_clips"] = 800
    summary.to_csv(output / "longtail_summary.csv", index=False)
    left = seeds[seeds.config == "r32"].set_index("training_seed")
    right = seeds[seeds.config == "r16"].set_index("training_seed")
    seed_tests = [
        dict(
            metric=m,
            training_seeds=3,
            test_clips=800,
            unit="training seed",
            **paired_test(left[m], right[m]),
        )
        for m in ["map", "tail_recall", "tail_map"]
    ]
    pd.DataFrame(seed_tests).to_csv(output / "seed_tests.csv", index=False)
    base = pd.read_csv(r / "cond_eval_none.csv").set_index("track")[STEMS]
    comparisons = []
    for name in [
        "film_enc",
        "film_dec",
        "film_both",
        "add_enc",
        "frozen_film_enc",
        "frozen_film_both",
        "C_cross",
    ]:
        d = pd.read_csv(r / f"cond_eval_{name}.csv").set_index("track")[STEMS]
        if set(d.index) != set(base.index) or not d.index.is_unique:
            raise ValueError("测试曲目未严格配对")
        d = d.loc[base.index]
        for stem in ["four_stem_mean"] + STEMS:
            a, b = (
                (d.mean(1), base.mean(1))
                if stem == "four_stem_mean"
                else (d[stem], base[stem])
            )
            comparisons.append(
                dict(
                    group=name,
                    metric=stem,
                    unit="test track",
                    training_seeds=1,
                    training_seed=0,
                    test_tracks=len(base),
                    **paired_test(a, b),
                )
            )
    cond = pd.DataFrame(comparisons)
    cond["p_holm"] = np.nan
    # 主指标 7 组为一族；28 个逐轨分析另为探索性比较族。
    for mask in [cond.metric == "four_stem_mean", cond.metric != "four_stem_mean"]:
        cond.loc[mask, "p_holm"] = holm(cond.loc[mask, "p_value"])
    cond.to_csv(output / "conditioned_tests.csv", index=False)
    tag_rows = []
    zs = (
        pd.read_csv(r / "clap_zeroshot_ap_official_test5329_w3_t4.csv")
        .set_index("tag")
        .AP
    )
    for tag in ["lora1_full", "lora2_full", "A_cap", "B_proj"]:
        values = pd.read_csv(r / f"lora_eval_{tag}.csv").set_index("tag").AP
        tag_rows.append(
            dict(
                config=tag,
                unit="fixed label class exploratory",
                test_clips=5329,
                training_seeds=1,
                **paired_test(values.loc[zs.index], zs),
            )
        )
    tags = pd.DataFrame(tag_rows)
    tags["p_holm"] = holm(tags.p_value)
    tags.to_csv(output / "tag_exploratory_tests.csv", index=False)
    meta = dict(
        date="2026-10-07",
        confidence=0.95,
        alpha=0.05,
        bootstrap_repeats=10000,
        statistics_seed=20261007,
        seed_ids=[1, 2, 3],
        tail_labels=TAIL,
        missing=[
            f
            for f in ["cond_eval_C_text.csv", "cond_eval_C_zero.csv"]
            if not (r / f).exists()
        ],
        limits=[
            "800 首子集多 seed 不等于 5329 首全量正式配方复现",
            "逐标签重采样仅分析固定标签的异质性，不能代替逐音频 AP 重算",
            "三组 seed 的双侧精确符号检验最低可达 p=0.25",
            "条件分离只有一个训练种子，曲目重采样不覆盖训练方差",
        ],
    )
    (output / "statistics_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(seed_tests).to_string(index=False))
    print(cond[cond.metric == "four_stem_mean"].to_string(index=False))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("AI_MUSIC_ASSETS", Path(__file__).resolve().parent.parent / "ai_music")),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "历史统计",
    )
    args = p.parse_args()
    audit(args.source, args.output)
