"""数据集自检：打印各数据集是否就绪。

等价于在正常 Python 环境下跑 `python -m src.data`，但本机用的是嵌入式 Python
（python310.zip 占了 sys.path[0]，不会自动把当前目录加进去），所以统一走这个脚本，
和"脚本编号即执行顺序"的约定也一致。

用法：
    python scripts/06_data_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import summary

if __name__ == "__main__":
    raise SystemExit(summary())
