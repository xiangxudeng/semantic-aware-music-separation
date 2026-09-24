"""从 MTT 标注里统计标签频次，并生成第 1 周要冻结的 50 类标签清单。

产出：
    results/mtt_tag_counts.csv   全部 188 个标签的频次（降序，供查证与后续扩充）
    results/mtt_50tags.csv       选定的 50 类标签，含分组与频次

选标签的原则：
  1. 频次要够高，正样本太少算不出有意义的 AP；
  2. 语义要能写成一句自然语言提示，CLAP 零样本才判得动；
  3. 去掉同义重复（drum/drums、violin/violins、string/strings 只留一个）；
  4. 四个维度都要覆盖：风格、乐器、情绪与节奏、人声属性。

改清单直接改下面的 GROUPS，重跑本脚本即可。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNO = ROOT / "data" / "mtt" / "annotations_final.csv"
RESULTS = ROOT / "results"

GROUPS = {
    "风格": [
        "classical", "techno", "electronic", "rock", "ambient", "opera", "indian",
        "pop", "new age", "dance", "country", "metal", "jazz", "baroque", "folk",
        "hard rock", "modern", "eastern", "trance",
    ],
    "乐器": [
        "guitar", "strings", "drums", "piano", "violin", "synth", "harpsichord",
        "flute", "sitar", "choir", "harp", "cello", "bass", "beat",
    ],
    "情绪与节奏": ["slow", "fast", "loud", "quiet", "soft", "weird", "upbeat", "heavy", "jazzy"],
    "人声属性": ["vocal", "female", "male", "singing", "instrumental", "solo", "voice", "no vocals"],
}


def main() -> int:
    if not ANNO.exists():
        print(f"没找到 {ANNO}，先跑 01_download_data.py mtt")
        return 1

    raw = pd.read_csv(ANNO, sep="\t")
    tags = [c for c in raw.columns if c not in ("clip_id", "mp3_path")]
    counts = raw[tags].astype(int).sum().sort_values(ascending=False)

    RESULTS.mkdir(parents=True, exist_ok=True)
    counts.rename("count").rename_axis("tag").reset_index().to_csv(
        RESULTS / "mtt_tag_counts.csv", index=False
    )

    rows = []
    for category, names in GROUPS.items():
        for name in names:
            if name not in counts.index:
                print(f"  !! 标签 '{name}' 不在 MTT 里，已跳过")
                continue
            rows.append({"category": category, "tag": name, "count": int(counts[name])})

    df = pd.DataFrame(rows)
    df.insert(0, "no", range(1, len(df) + 1))
    df.to_csv(RESULTS / "mtt_50tags.csv", index=False)

    print(f"MTT：{len(raw)} 个片段，{len(tags)} 个标签")
    print(f"选定 {len(df)} 类标签 -> mtt_50tags.csv\n")
    for category in GROUPS:
        sub = df[df["category"] == category]
        print(f"  {category}（{len(sub)}）")
        print("    " + "、".join(f"{r.tag}({r.count})" for r in sub.itertuples()))

    overlap = [t for t in df["tag"] if t in ("drum", "violins", "string", "female vocal", "male voice")]
    if overlap:
        print(f"\n  提示：清单里仍含疑似同义词 {overlap}，建议再确认")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
