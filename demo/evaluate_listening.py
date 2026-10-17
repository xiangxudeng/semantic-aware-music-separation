"""复核人工试听工作簿；保留原弱标签结果，按既定0/1/2评分独立汇总。"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from analysis_stats import paired_test

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_rows(path):
    with ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(s.itertext()) for s in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall(".//m:row", NS):
            values = {}
            for cell in row:
                value = cell.findtext("m:v", "", NS)
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", NS).itertext())
                values[cell.get("r").rstrip("0123456789")] = value
            rows.append(values)
    headers = rows.pop(0)
    return [{name: row.get(col, "") for col, name in headers.items()} for row in rows]


def summarize(rows, original):
    def key(r):
        return r["case_id"], r["language"], int(r["rank"]), int(r["clip_id"])

    keys = [key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {key(r) for r in original}:
        raise ValueError("评分表存在重复、缺失或候选身份变化")
    source = {key(r): r for r in original}
    groups = defaultdict(list)
    for row in rows:
        if row["listening_score"] not in {"0", "1", "2"} or not row["reviewer"].strip():
            raise ValueError("评分必须为0/1/2且评审不能为空")
        if any(row[col] != source[key(row)][col] for col in ["query", "relevant_by_labels"]):
            raise ValueError("原始查询或弱标签判分被修改")
        groups[row["case_id"], row["language"]].append(row)
    if any(sorted(int(r["rank"]) for r in part) != [1, 2, 3, 4, 5] for part in groups.values()):
        raise ValueError("每条查询必须具备Top1至Top5")
    cases = []
    for (case, lang), part in groups.items():
        scores = [int(r["listening_score"]) for r in part]
        cases.append(dict(case_id=case, language=lang, query=part[0]["query"],
                          score_counts=dict(Counter(scores)), mean_score=sum(scores) / 5,
                          p5_score2=scores.count(2) / 5, hit5_score2=int(2 in scores),
                          p5_score_atleast1=sum(s >= 1 for s in scores) / 5))
    summary = {}
    for lang in ["en", "zh"]:
        part = [r for r in rows if r["language"] == lang]
        queries = [c for c in cases if c["language"] == lang]
        summary[lang] = dict(candidate_count=len(part), query_count=len(queries),
                            score_counts=dict(Counter(int(r["listening_score"]) for r in part)),
                            mean_score=sum(c["mean_score"] for c in queries) / len(queries),
                            p5_score2=sum(c["p5_score2"] for c in queries) / len(queries),
                            hit5_score2=sum(c["hit5_score2"] for c in queries) / len(queries),
                            p5_score_atleast1=sum(c["p5_score_atleast1"] for c in queries) / len(queries),
                            score2_label_negative=sum(r["listening_score"] == "2" and r["relevant_by_labels"] == "0" for r in part))
    en = {c["case_id"]: c["p5_score2"] for c in cases if c["language"] == "en"}
    zh = {c["case_id"]: c["p5_score2"] for c in cases if c["language"] == "zh"}
    if en.keys() != zh.keys():
        raise ValueError("中英文意图不配对")
    return dict(score_scale={"0": "不相关", "1": "部分相关", "2": "大部分相关"},
                score_scale_source="原试听复核.html既定标准；2分不等于完全匹配",
                reviewer_count=len({r["reviewer"] for r in rows}), reviewed_candidates=len(rows),
                summary=summary, cases=cases,
                paired_en_minus_zh_p5_score2=paired_test(list(en.values()), [zh[c] for c in en]),
                limitations=["单名评审；盲评与评分一致性未验证", "12组固定查询，不代表所有查询", "2分表示大部分相关，不证明全部条件满足", "人工与弱标签判分回答不同问题，分歧不能一概归因于标签漏标"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.workbook)
    with args.original.open(encoding="utf-8-sig", newline="") as f:
        original = list(csv.DictReader(f))
    result = summarize(rows, original)
    result["source_sha256"] = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "人工试听汇总.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in [("人工试听候选.csv", rows), ("人工试听逐查询.csv", result["cases"])]:
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"reviewed": len(rows), "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
