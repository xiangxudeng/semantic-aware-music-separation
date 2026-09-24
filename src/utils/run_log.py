"""运行记录：让每一次实验都能追溯到"当时到底跑了什么"。

为什么需要它
------------
第 2 周踩过好几次"现象对不上、但已经想不起当时的命令和配置"的坑——例如
对比学习一开始全线变差，事后才判断出是学习率过大的过拟合，而不是方法问题。
如果当时每次运行都留下了命令、配置、环境版本和产物清单，这类问题当场就能定位。

每次运行会产出
--------------
```
results/runs/<时间戳>_<名称>/
├── command.txt     完整命令行（直接复制就能复现）
├── config.json     本次的配置字典
├── env.json        Python / torch / CUDA / 依赖版本 / 数据指纹
├── log.txt         运行过程中写下的关键信息（可 append）
└── status.json     结束状态：ok / 异常类型与 traceback
```
同时在 `results/runs/index.csv` 追加一行摘要，方便总览"哪次跑了什么、结果如何"。

用法
----
    from src.utils.run_log import run_record

    with run_record("film_train", config=vars(args)) as run:
        run.log("epoch 1 done")
        run.artifact(results_csv)
        run.note("长尾 AP 0.13")
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
INDEX = RUNS / "index.csv"


def _versions() -> dict:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    for mod in ("torch", "numpy", "pandas", "sklearn", "transformers", "laion_clap", "demucs", "museval"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[mod] = "缺失"
    try:
        import torch

        out["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_mem_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
        out["torch_threads"] = str(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        pass
    return out


def data_fingerprint() -> dict:
    """记录关键数据的规模，便于判断"两次实验用的数据是不是同一份"。"""
    fp: dict[str, object] = {}
    mtt = ROOT / "data" / "mtt"
    if mtt.exists():
        fp["mtt_mp3_count"] = sum(1 for _ in mtt.rglob("*.mp3"))
        tags = ROOT / "results" / "mtt_50tags.csv"
        if tags.exists():
            fp["mtt_50tags_hash"] = f"{tags.stat().st_size}-{int(tags.stat().st_mtime)}"
    musdb = ROOT / "data" / "musdb18hq_wav"
    if musdb.exists():
        for split in ("train", "test"):
            d = musdb / split
            if d.exists():
                fp[f"musdb_{split}_tracks"] = sum(1 for p in d.iterdir() if p.is_dir())
    lp = ROOT / "data" / "lpmusiccaps"
    if lp.exists():
        fp["lpmusiccaps_parquet"] = sum(1 for _ in lp.glob("*.parquet"))
    return fp


class Run:
    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        self.name = name
        self.artifacts: list[str] = []
        self.notes: list[str] = []
        self._t0 = time.time()

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.path / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def note(self, message: str) -> None:
        self.notes.append(message)
        self.log(f"结论：{message}")

    def artifact(self, *paths) -> None:
        for p in paths:
            self.artifacts.append(str(p))


@contextmanager
def run_record(name: str, config: dict | None = None, extra_env: dict | None = None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RUNS / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=True)

    (path / "command.txt").write_text(" ".join([sys.executable, *sys.argv]), encoding="utf-8")
    (path / "config.json").write_text(
        json.dumps(config or {}, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    env = _versions()
    env["data"] = data_fingerprint()
    if extra_env:
        env.update(extra_env)
    (path / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "log.txt").write_text(f"运行 {name} 于 {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    print(f"\n[run_record] 本次运行目录：{path}")
    print(f"[run_record] 命令：{' '.join([Path(sys.argv[0]).name, *sys.argv[1:]])}")
    run = Run(path, name)
    status = "ok"
    err = ""
    try:
        yield run
    except BaseException as exc:  # noqa: BLE001
        status = type(exc).__name__
        err = traceback.format_exc()
        (path / "traceback.txt").write_text(err, encoding="utf-8")
        raise
    finally:
        run.notes.append(f"耗时 {time.time() - run._t0:.1f} 秒")
        (path / "status.json").write_text(
            json.dumps(
                {"status": status, "seconds": round(time.time() - run._t0, 1),
                 "artifacts": run.artifacts, "notes": run.notes, "error": err.splitlines()[-1] if err else ""},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        new = not INDEX.exists()
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["时间", "名称", "状态", "秒", "结论", "目录"])
            w.writerow([
                stamp, name, status, round(time.time() - run._t0, 1),
                " | ".join(run.notes)[:300], path.name,
            ])
        print(f"[run_record] 状态 {status}，索引已更新 {INDEX}")
