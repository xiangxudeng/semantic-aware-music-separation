"""第 1 周数据下载：MUSDB18（分轨）、MTT（标签）、MusicCaps（文字标注）。

走 HuggingFace 国内镜像 hf-mirror.com——Zenodo 官网国内通常打不开，
而且 MUSDB18-HQ 无损版有 20 GB 量级，起步用官方 .stem.mp4 版（5.7 GB）更划算。

用法：
    python scripts/01_download_data.py            # 三个都下
    python scripts/01_download_data.py musdb18    # 只下其中一个
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 必须在 import 前设置
# 镜像只代理元数据，文件字节会重定向到 HuggingFace 的海外存储（cas-bridge.xethub.hf.co），
# 国内链路经常跑一阵就断。默认 10 秒的超时太激进，放宽到 300 秒。
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("缺少 huggingface_hub，先执行：pip install -r requirements.txt")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# name: (repo_id, 说明, 只下哪些文件; None 表示全部)
REPOS = {
    "musdb18": ("artemisweb/MUSDB18", "官方 .stem.mp4 分轨，100 训练 + 50 测试，约 5.7 GB", None),
    "musdb18_test": (
        "artemisweb/MUSDB18",
        "只下测试集 50 首，约 1.9 GB —— 第 1 周算基线够用，训练集留到第 2 周",
        ["test/*"],
    ),
    "mtt": ("confit/magnatagatune", "音频 mp3.zip + 标签 + 官方划分 tsv，约 3 GB", None),
    "musiccaps": ("google/MusicCaps", "只有文字标注 CSV，约 3 MB（音频见 README 说明）", None),
    "musdb18hq": (
        "roro128/musdb18-hq-flac",
        "MUSDB18-HQ 无损版（PCM-16 FLAC 打包进 parquet），12 GB，音质与官方 HQ 等价",
        None,
    ),
}


def download(name: str) -> Path:
    repo_id, note, allow = REPOS[name]
    # musdb18_test 是 musdb18 的子集，共用同一个目录，避免出现两份重复数据
    target = DATA / ("musdb18" if name == "musdb18_test" else name)
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[{name}] {repo_id}\n  {note}\n  -> {target}")
    # 文件字节走的是海外 CDN，连接随时可能被掐断导致文件不完整；
    # hf_hub 遇到大小不符会直接抛错退出，这里自动重试续传。
    for attempt in range(1, 11):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=allow,
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  第 {attempt} 次中断（{type(exc).__name__}: {str(exc)[:80]}），5 秒后续传")
            time.sleep(5)
    else:
        print("  连续 10 次失败，请稍后重跑本脚本")
    return target


def verify(name: str, target: Path) -> bool:
    """返回 True 表示校验通过。"""
    if name == "musdb18hq":
        files = list((target / "data").glob("*.parquet")) if (target / "data").exists() else []
        ok = len(files) >= 14
        print(f"  parquet 文件 {len(files)} 个  {'OK' if ok else '!! 还没下完，重跑本脚本继续'}")
        return ok
    if name in ("musdb18", "musdb18_test"):
        train = list((target / "train").glob("*.stem.mp4"))
        test = list((target / "test").glob("*.stem.mp4"))
        if name == "musdb18_test":
            ok = len(test) == 50
            print(f"  test={len(test)}/50  {'OK' if ok else '!! 还没下完，重跑本脚本继续'}")
        else:
            ok = len(train) == 100 and len(test) == 50
            print(f"  train={len(train)}  test={len(test)}  {'OK' if ok else '!! 数量不对，下载可能不完整'}")
        return ok
    if name == "mtt":
        needed = ["mp3.zip", "train_gt_mtt.tsv", "val_gt_mtt.tsv", "test_gt_mtt.tsv"]
        missing = [f for f in needed if not (target / f).exists()]
        print(f"  缺少 {missing}" if missing else "  文件齐全 OK")
        return not missing
    if name == "musiccaps":
        csv = target / "musiccaps-public.csv"
        print(f"  {csv.name} {'OK' if csv.exists() else '缺失'}")
        return csv.exists()
    return True


def main(argv: list[str]) -> int:
    names = argv[1:] or list(REPOS)
    unknown = [n for n in names if n not in REPOS]
    if unknown:
        print(f"未知目标 {unknown}，可选：{list(REPOS)}")
        return 2

    failed = []
    for name in names:
        target = download(name)
        if not verify(name, target):
            failed.append(name)

    print("\n完成。挂在后台慢慢下即可，中断了重跑本脚本会续传。")
    if failed:
        print(f"校验未通过：{failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
