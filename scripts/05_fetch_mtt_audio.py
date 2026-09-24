"""下载 MagnaTagATune 的音频包 mp3.zip（单个文件约 3 GB）。

为什么不用 hf_hub：这个仓只有一个 3 GB 的大文件，hf_hub 的 Xet 批量协议在它上面
会卡住（进程空转不报错），而直接用带 Range 的普通 HTTP 请求是通的。
所以这里用最朴素的方式：断点续传 + 无限重试，断了就接着下。

用法：
    python scripts/05_fetch_mtt_audio.py
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://hf-mirror.com/datasets/confit/magnatagatune/resolve/main/mp3.zip"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "mtt" / "mp3.zip"
UA = {"User-Agent": "Mozilla/5.0"}
CHUNK = 4 << 20          # 4 MB
REPORT_EVERY = 100 << 20  # 每 100 MB 报一次
MAX_RETRIES = 500


def remote_size() -> int:
    req = urllib.request.Request(URL, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return int(r.headers.get("Content-Length") or 0)


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size()
    print(f"目标 {DEST}")
    print(f"远端大小 {total / 1e9:.2f} GB")

    for attempt in range(1, MAX_RETRIES + 1):
        done = DEST.stat().st_size if DEST.exists() else 0
        if done >= total:
            print(f"下载完成：{done / 1e9:.2f} GB")
            return 0

        headers = {**UA}
        if done:
            headers["Range"] = f"bytes={done}-"
        print(f"\n第 {attempt} 次尝试，从 {done / 1e6:.0f} MB 续传")
        t0 = time.time()
        start_done = done
        last_report = done
        try:
            with urllib.request.urlopen(
                urllib.request.Request(URL, headers=headers), timeout=120
            ) as resp, open(DEST, "ab") as out:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if done - last_report >= REPORT_EVERY:
                        last_report = done
                        rate = (done - start_done) / max(time.time() - t0, 1)
                        pct = done / total * 100
                        eta = (total - done) / max(rate, 1)
                        print(
                            f"  {done / 1e6:7.0f} MB / {total / 1e6:.0f} MB "
                            f"({pct:5.1f}%)  {rate / 1e6:.2f} MB/s  预计剩余 {eta / 60:.0f} 分钟",
                            flush=True,
                        )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  连接断了（{type(exc).__name__}: {str(exc)[:70]}），5 秒后重试")
            time.sleep(5)
            continue

        if DEST.stat().st_size >= total:
            print(f"\n下载完成：{DEST.stat().st_size / 1e9:.2f} GB")
            return 0
        print("  本次连接正常结束但还没下完，继续续传")
        time.sleep(2)

    print("重试次数用尽，仍没下完，重跑本脚本会接着下")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
