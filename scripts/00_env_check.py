"""第 1 周环境自检：一次性打印硬件、依赖版本、CUDA、ffmpeg、磁盘。

用法：python scripts/00_env_check.py
输出：控制台摘要 + results/env_check.txt 完整报告
"""

from __future__ import annotations

import ctypes
import datetime
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "env_check.txt"

PACKAGES = [
    "torch",
    "torchaudio",
    "numpy",
    "librosa",
    "soundfile",
    "demucs",
    "musdb",
    "pandas",
    "matplotlib",
    "huggingface_hub",
]


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "NOT INSTALLED"


def ram_gb() -> str:
    """Windows 内存信息，取不到就返回未知。"""
    try:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return f"总 {status.ullTotalPhys / 2**30:.1f} GB / 可用 {status.ullAvailPhys / 2**30:.1f} GB"
    except Exception:  # noqa: BLE001
        return "未知"


def torch_block() -> list[str]:
    lines = ["torch 与加速器"]
    try:
        import torch

        lines.append(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        lines.append(f"  CUDA 设备数              = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        lines.append(f"  torch 编译时 CUDA 版本   = {torch.version.cuda}")
        if not torch.cuda.is_available():
            lines.append("  → 无 NVIDIA CUDA 设备：第 1 周基线可用 CPU 完成；")
            lines.append("    第 2、3 周的微调需要实验室服务器或云 GPU。")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  import torch 失败：{exc}")
    return lines


def ffmpeg_block() -> list[str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return ["ffmpeg", "  未找到：demucs / musdb 读 mp4、mp3 会直接报错，必须先装并加入 PATH"]
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=20, check=False
        )
        return ["ffmpeg", f"  {exe}", f"  {out.stdout.splitlines()[0] if out.stdout else ''}"]
    except Exception as exc:  # noqa: BLE001
        return ["ffmpeg", f"  调用失败：{exc}"]


def disk_block() -> list[str]:
    lines = ["磁盘可用空间"]
    # Windows 看各盘符，Linux（云服务器）看根目录与数据盘
    roots = (
        [Path(f"{c}:/") for c in "CDEFG"]
        if os.name == "nt"
        else [Path("/"), Path("/root/workspace"), Path("/tmp")]
    )
    for root in roots:
        if not root.exists():
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        lines.append(
            f"  {str(root):16} 共 {usage.total / 2**30:.0f} GB，"
            f"已用 {usage.used / 2**30:.0f} GB，可用 {usage.free / 2**30:.0f} GB"
        )
    lines.append("  提示：全套数据约 47 GB；云服务器上务必放数据盘，系统盘通常只有 30 GB。")
    return lines


def build_report() -> str:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blocks = [
        ["系统",
         f"  平台       : {platform.platform()}",
         f"  Python     : {sys.version.split()[0]}",
         f"  解释器路径 : {sys.executable}",
         f"  CPU        : {platform.processor() or platform.machine()}",
         f"  逻辑核心数 : {os.cpu_count()}",
         f"  内存       : {ram_gb()}",
         f"  项目根目录 : {ROOT}"],
        ["依赖版本", *[f"  {name:<18} {pkg_version(name)}" for name in PACKAGES]],
        torch_block(),
        ffmpeg_block(),
        disk_block(),
    ]
    body = "\n\n".join("\n".join(block) for block in blocks)
    return f"环境自检报告  生成时间 {stamp}\n\n{body}\n"


def main() -> int:
    report = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved -> {REPORT}")

    missing = [name for name in PACKAGES if pkg_version(name) == "NOT INSTALLED"]
    if missing:
        print("MISSING:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
