"""预先固定12组中英自由组合查询；弱标注约束不冒充独立人工试听。"""

import argparse
import html
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from music import MusicSystem, normalize
from analysis_stats import paired_test

# 在看检索结果前固定描述和约束；每个内层组为OR，组间为AND。
CASES = [
    (
        "slow_piano",
        "Slow piano music without singing, for a quiet evening",
        "安静夜晚适合的慢速钢琴音乐，没有演唱",
        [
            ["piano"],
            ["slow"],
            [
                "no vocals",
                "no voice",
                "no singing",
                "instrumental",
                "no voices",
                "no vocal",
            ],
        ],
    ),
    (
        "female_acoustic",
        "An acoustic guitar accompanying a female singer",
        "木吉他伴奏的女声歌曲",
        [
            ["acoustic", "acoustic guitar"],
            ["guitar", "acoustic guitar"],
            [
                "female",
                "female vocal",
                "female singing",
                "female vocals",
                "female singer",
            ],
        ],
    ),
    (
        "male_rock",
        "A male singer over rock guitar riffs",
        "男声演唱与摇滚吉他相结合的音乐",
        [
            ["male", "male vocal", "male singer", "male vocals", "man singing"],
            ["rock", "hard rock"],
            ["guitar", "electric guitar", "guitars"],
        ],
    ),
    (
        "fast_electronic",
        "Fast electronic dance music driven by drums",
        "鼓点推动的快速电子舞曲",
        [
            ["fast", "quick", "fast beat"],
            ["electronic", "techno", "dance", "electronica", "electro"],
            ["drums", "drum", "beat", "beats"],
        ],
    ),
    (
        "classical_violin",
        "Classical strings with a violin melody",
        "小提琴旋律与古典弦乐相结合的音乐",
        [["classical", "orchestral", "orchestra", "clasical"], ["violin", "violins"]],
    ),
    (
        "jazz_sax",
        "A jazz piece with a saxophone melody",
        "带萨克斯旋律的爵士乐",
        [["jazz", "jazzy"], ["sax"]],
    ),
    (
        "ambient_synth",
        "Ambient synthesizer music for a spacious background",
        "适合作为空间感背景的氛围合成器音乐",
        [["ambient"], ["synth", "synthesizer"]],
    ),
    (
        "soft_flute",
        "Soft flute music with a calm feeling",
        "柔和安静的长笛音乐",
        [["soft", "quiet", "calm", "mellow"], ["flute", "flutes"]],
    ),
    (
        "baroque_keys",
        "A baroque piece played on a harpsichord",
        "羽管键琴演奏的巴洛克风格音乐",
        [["baroque"], ["harpsichord", "harpsicord"]],
    ),
    (
        "slow_cello",
        "A slow cello melody for reflective moments",
        "适合沉思片刻的慢速大提琴旋律",
        [["slow"], ["cello"]],
    ),
    (
        "heavy_drums",
        "Heavy metal music with forceful drums",
        "带有强劲鼓声的重金属音乐",
        [["metal", "heavy metal", "hard rock", "heavy"], ["drums", "drum"]],
    ),
    (
        "piano_choir",
        "Piano accompaniment behind a choir",
        "钢琴为合唱伴奏的音乐",
        [["piano"], ["choir", "choral", "chorus"]],
    ),
]


def constraint_matches(annotation, groups):
    return np.stack(
        [annotation[group].to_numpy(int).any(axis=1) for group in groups], axis=1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "自由查询",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    system = MusicSystem(args.assets)
    annotation = pd.read_csv(
        system.assets / "data/mtt/annotations_final.csv",
        sep="\t",
        dtype={"clip_id": str},
    )
    test_ids = set(
        pd.read_csv(
            system.assets / "data/mtt/test_gt_mtt.tsv", sep="\t", header=None, dtype=str
        )[0]
    )
    indices = np.flatnonzero(annotation.clip_id.isin(test_ids))
    test = annotation.iloc[indices].reset_index(drop=True)
    assert len(test) == 5329 and test.clip_id.is_unique
    specs = [
        dict(case_id=case, language=lang, query=query, required_groups=groups)
        for case, en, zh, groups in CASES
        for lang, query in [("en", en), ("zh", zh)]
    ]
    # 先写查询及判分规则，再编码；不根据结果修改题目或剔除失败项。
    (args.output / "queries.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    start = time.perf_counter()
    vectors = system._encoder().text([row["query"] for row in specs])
    scores = normalize(system.embeddings[indices]) @ normalize(vectors).T
    results, retrieved = [], []
    positive_columns = [c for c in test if c not in ["clip_id", "mp3_path"]]
    for i, spec in enumerate(specs):
        matches = constraint_matches(test, spec["required_groups"])
        relevant = matches.all(axis=1)
        order = np.argsort(-scores[:, i], kind="stable")[:5]
        hits = int(relevant[order].sum())
        results.append(
            dict(
                **spec,
                gallery_relevant=int(relevant.sum()),
                hits_at5=hits,
                precision_at5=hits / 5,
                hit_at5=int(hits > 0),
                constraint_coverage=float(matches[order].mean()),
                clip_ids=";".join(test.iloc[order].clip_id),
            )
        )
        for rank, index in enumerate(order, 1):
            row = test.iloc[index]
            retrieved.append(
                dict(
                    case_id=spec["case_id"],
                    language=spec["language"],
                    query=spec["query"],
                    rank=rank,
                    clip_id=row.clip_id,
                    similarity=float(scores[index, i]),
                    relevant_by_labels=int(relevant[index]),
                    matched_groups=json.dumps(matches[index].astype(int).tolist()),
                    reference_tags=";".join(c for c in positive_columns if row[c] == 1),
                    audio_path=str(system.assets / "data/mtt" / row.mp3_path),
                    listening_score="",
                    reviewer="",
                    review_note="",
                )
            )
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "query_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(retrieved).to_csv(
        args.output / "retrieved_items.csv", index=False, encoding="utf-8-sig"
    )
    summary = {}
    for lang in ["en", "zh"]:
        part = frame[frame.language == lang]
        summary[lang] = dict(
            query_count=len(part),
            precision_at5=float(part.precision_at5.mean()),
            hit_at5=float(part.hit_at5.mean()),
            constraint_coverage=float(part.constraint_coverage.mean()),
            cases_without_known_relevant=int((part.gallery_relevant == 0).sum()),
        )
        eligible = part[part.gallery_relevant > 0]
        summary[lang]["evaluable_queries"] = dict(
            query_count=len(eligible),
            precision_at5=float(eligible.precision_at5.mean()),
            hit_at5=float(eligible.hit_at5.mean()),
        )
    en = frame[frame.language == "en"].set_index("case_id")
    zh = frame[frame.language == "zh"].set_index("case_id")
    overlaps = [
        len(
            set(en.loc[key, "clip_ids"].split(";"))
            & set(zh.loc[key, "clip_ids"].split(";"))
        )
        / 5
        for key in en.index
    ]
    meta = dict(
        model="Official LAION CLAP HTSAT unfused",
        gallery_size=len(test),
        gallery_split="MTT test",
        raw_chinese=True,
        translation_or_query_tuning=False,
        query_seed="none: predefined cases",
        statistics_seed=20261007,
        summary=summary,
        bilingual_overlap_at5=float(np.mean(overlaps)),
        paired_en_minus_zh_precision=paired_test(
            en.precision_at5, zh.loc[en.index].precision_at5
        ),
        evaluable_en_minus_zh_precision=paired_test(
            en.loc[en.gallery_relevant > 0].precision_at5,
            zh.loc[en.index[en.gallery_relevant > 0]].precision_at5,
        ),
        evaluable_rule="known gallery positives > 0, independent of retrieval rank; report all 12 cases and the 10 evaluable cases separately",
        elapsed_seconds=round(time.perf_counter() - start, 3),
        relevance="Every required group must have an explicitly positive MTT label; OR within groups, AND between groups",
        limitations=[
            "不是独立人工试听评测；人工评分列保留空白",
            "场景适用性、主观情绪和否定语义不能完全由标签确认",
            "标签缺失会漏计；no vocals使用显式正标注，不从未标注vocal推断无演唱",
            "12组便利样本不代表所有自由查询；配对区间和p值仅描述这组固定查询",
            "中英文为同组意图但翻译可能有表达差异；无翻译预处理",
        ],
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cards = []
    for item in retrieved:
        path = quote(item["audio_path"].replace("\\", "/"), safe="/:")
        cards.append(
            f'<article><h3>{html.escape(item["case_id"])} / {item["language"]} / Top {item["rank"]}</h3><p>{html.escape(item["query"])}</p><audio controls preload="none" src="http://127.0.0.1:7860/file={path}"></audio><p>片段 {item["clip_id"]}；参考标签：{html.escape(item["reference_tags"])}</p><p>自动标签匹配：{item["relevant_by_labels"]}；人工试听分数未填写。</p></article>'
        )
    document = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>自由查询试听复核</title><style>body{max-width:900px;margin:32px auto;font-family:system-ui}article{border-top:1px solid #ccc;padding:16px}audio{width:100%}</style><h1>自由查询试听复核</h1><p>24条查询、每条Top5。保持本机Demo运行以试听；人工评分请填写retrieved_items.csv中的listening_score、reviewer和review_note。建议0=不相关，1=部分相关，2=大部分相关。该页面未产生人工评测分数。</p>'
        + "".join(cards)
        + "</html>"
    )
    (args.output / "试听复核.html").write_text(document, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
