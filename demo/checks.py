"""最小回归检查：统计、索引排序、真实输入边界，不加载大模型。"""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import soundfile as sf
from music import MusicSystem, topk, format_creation
from analysis_stats import paired_test, holm


def main():
    idx, score = topk(np.eye(3), [0, 1, 0], 2)
    assert idx.tolist() == [1, 0] and score[0] == 1
    try:
        topk(np.eye(3), [0, 0, 0])
        raise AssertionError("零向量未拒绝")
    except ValueError:
        pass
    s = paired_test([1, 2, 3], [1, 2, 3])
    assert s["p_value"] == 1 and s["delta"] == 0
    s = paired_test([2, 2, 2], [1, 1, 1])
    assert s["p_value"] == 0.25 and s["permutation_repeats"] == 8
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    analysis = {
        "tracks": {
            s: {"rms": 0.1, "tags": [{"tag": "piano"}]}
            for s in ["vocals", "drums", "bass", "other"]
        }
    }
    good = format_creation(
        analysis,
        json_text
        := '{"caption":"让音乐陪着画面，把今天的故事慢慢展开。","scenes":["旅行风景","日常记录","城市夜景"]}',
    )
    assert good["source"] == "Qwen结构化草稿" and all(
        f"- {s}" in good["text"] for s in ["人声", "鼓", "贝斯", "其他"]
    )
    for raw in [
        "输出被截断",
        json_text.replace("今天的故事", "混音教学"),
        json_text.replace("城市夜景", "旅行风景"),
    ]:
        assert format_creation(analysis, raw)["source"] == "候选标签规则草稿"
    assert (
        format_creation(analysis, json_text.replace("城市夜景", " 旅行风景 "))["source"]
        == "候选标签规则草稿"
    )
    analysis["tracks"]["other"]["rms"] = 0
    assert "其他：轨道近乎静音" in format_creation(analysis, "")["text"]
    from evaluate_free import constraint_matches
    import pandas as pd

    matched = constraint_matches(
        pd.DataFrame({"a": [1, 0], "b": [0, 1], "c": [1, 0]}), [["a", "b"], ["c"]]
    )
    assert matched.all(axis=1).tolist() == [True, False]
    from evaluate_listening import summarize

    rated = [
        dict(case_id="test", language=lang, rank=str(rank), clip_id=str(rank),
             query=lang, relevant_by_labels="0", listening_score=score, reviewer="test")
        for lang, score in [("en", "2"), ("zh", "0")]
        for rank in range(1, 6)
    ]
    human = summarize(rated, rated)
    assert human["summary"]["en"]["p5_score2"] == 1
    assert human["summary"]["zh"]["p5_score2"] == 0
    assert human["paired_en_minus_zh_p5_score2"]["p_value"] == 1
    for broken in [rated[:-1], rated + [rated[0]], [dict(r, listening_score="3") for r in rated]]:
        try:
            summarize(broken, rated)
            raise AssertionError("无效人工评分未拒绝")
        except ValueError:
            pass
    system = MusicSystem()
    for query in [None, "", "   "]:
        try:
            system.search(query)
            raise AssertionError("空查询未拒绝")
        except ValueError:
            pass
    import torch
    from src.data.augment import WaveformAugment

    targets = torch.randn(4, 2, 1024)
    mixture, augmented = WaveformAugment()(targets.sum(0), targets, force=True)
    assert torch.allclose(mixture, augmented.sum(0), atol=1e-5)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "silent.wav"
        sf.write(path, np.zeros(4410), 44100)
        try:
            system.analyze(path)
            raise AssertionError("静音未拒绝")
        except ValueError:
            pass
    print(
        "PASS: retrieval, empty query, exact p, Holm, cached alignment, silent input, paired augmentation, four-stem assistant, rejected/duplicate drafts, query constraints, human scoring"
    )


if __name__ == "__main__":
    main()
