"""数据增强消融实验（第 2 周 P0 验收项："3 组增强消融"）。

做法：在同一份数据、同一套超参下跑 3 组短程微调，唯一变量是数据增强，
用训练脚本内置的每轮 SI-SDR 验证来比"训练前 → 训练后"的变化。

三组的选法
----------
| 组名 | 增强 | 想回答的问题 |
|---|---|---|
| `none` | 全关 | 不做增强时微调会怎么走（对照组） |
| `mask` | 只开频带掩蔽 + 时间裁剪 | 只看"输入有缺失"这一类增强有没有用 |
| `default` | 五种全开 | 完整增强配置的净效果 |

说明：这里比的是**短程微调里增强带来的相对变化**，不是最终 SDR。
第 2 周已经确认全参数微调整体拿不到收益，所以消融的目的是把
"增强有没有帮助"这件事单独量出来，而不是去刷一个更好的模型。

用法（云端 GPU）：
    python scripts/33_augment_ablation.py --steps 300 --lr 1e-4 --batch-size 3 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STEMS = ("drums", "bass", "other", "vocals")

GROUPS = {
    "none": "none",
    "mask": "mask_only",
    "default": "default",
}


def run_group(name: str, augment: str, args) -> int:
    tag = f"abl_{name}"
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "20_finetune_demucs.py"),
        "--epochs", "1",
        "--steps-per-epoch", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--segment", str(args.segment),
        "--lr", str(args.lr),
        "--augment", augment,
        "--device", args.device,
        "--val-tracks", str(args.val_tracks),
        "--tag", tag,
    ]
    print(f"\n{'=' * 70}\n[{name}] 增强配置 = {augment}\n{'=' * 70}", flush=True)
    print("  " + " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def load_curve(tag: str) -> list[dict] | None:
    path = RESULTS / f"finetune_val_{tag}.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300, help="每组跑多少步")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--segment", type=float, default=7.8)
    ap.add_argument("--val-tracks", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--skip-run", action="store_true", help="只做汇总，不重新训练")
    args = ap.parse_args()

    names = [g.strip() for g in args.groups.split(",") if g.strip()]
    for name in names:
        if name not in GROUPS:
            print(f"未知分组 {name}，可选：{list(GROUPS)}")
            return 1

    if not args.skip_run:
        for name in names:
            code = run_group(name, GROUPS[name], args)
            if code != 0:
                print(f"  !! 分组 {name} 训练失败，退出码 {code}")
                return code

    rows = []
    print(f"\n{'=' * 70}\n消融结果\n{'=' * 70}")
    for name in names:
        curve = load_curve(f"abl_{name}")
        if not curve:
            print(f"  {name:8} 缺少结果文件，跳过")
            continue
        before = {s: float(curve[0][s]) for s in STEMS}
        after = {s: float(curve[-1][s]) for s in STEMS}
        b_mean = float(curve[0]["mean"])
        a_mean = float(curve[-1]["mean"])
        rows.append({
            "组": name,
            "增强": GROUPS[name],
            "训练前均值": round(b_mean, 3),
            "训练后均值": round(a_mean, 3),
            "变化": round(a_mean - b_mean, 3),
            **{f"Δ{s}": round(after[s] - before[s], 3) for s in STEMS},
        })

    if not rows:
        print("没有任何结果可汇总")
        return 1

    cols = ["组", "增强", "训练前均值", "训练后均值", "变化", *[f"Δ{s}" for s in STEMS]]
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) + 2 for c in cols}
    print("  " + "".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  " + "".join(str(r[c]).ljust(width[c]) for c in cols))

    out = RESULTS / "augment_ablation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n明细 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
