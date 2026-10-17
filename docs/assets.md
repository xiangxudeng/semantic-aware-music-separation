# 模型与数据资产准备

演示代码与资产分开保存。`--assets` 或 `AI_MUSIC_ASSETS` 指向资产根目录，路径由使用者自行选择。

```text
assets/
├── data/
│   ├── mtt/
│   │   ├── annotations_final.csv
│   │   ├── train_gt_mtt.tsv / val_gt_mtt.tsv / test_gt_mtt.tsv
│   │   └── <索引引用的 MP3 文件>
│   ├── musdb18hq_wav/       # 条件训练需要：100 train + 50 test
│   └── lpmusiccaps/         # 指令构建需要：train-*.parquet
├── results/
│   ├── mtt_50tags.csv
│   ├── mtt_index.csv
│   ├── clap_official_emb_all25863_w3_fp16.npy
│   ├── musdb_semantic.npy         # 条件训练需要
│   └── musdb_semantic_index.csv   # 条件训练需要
└── third_party/
    ├── bert-base-uncased/
    ├── roberta-base/
    └── facebook/bart-base/
```

MTT 标签 CSV、索引及矩阵必须保持原 annotations 行顺序；索引至少含 clip_id、mp3_path、tags。音频矩阵为 25,863 × 512，来自官方 LAION CLAP unfused、model_id=1、三窗口及 L2 归一化，不能与其他实现混用。历史缓存采用固定种子的三次裁剪；在线编码使用均匀十秒窗口，复现实验时应分别记录窗口策略。

另需准备以下模型缓存：

- LAION CLAP `630k-audioset-best.pt`：`load_ckpt(model_id=1)` 使用包内默认位置。
- Demucs `htdemucs`：通过 PyTorch/Demucs 缓存解析；资产目录不是其自定义权重搜索位置。
- Hugging Face 缓存中的 `Qwen/Qwen2-1.5B-Instruct`；最终创作助手使用 `local_files_only=True`。
- CLAP 的 BERT、RoBERTa、BART tokenizer；`third_party/` 是本地离线副本。

请按上游发布说明获取数据与模型。根目录 `scripts/01`–`05`、`22_clap_zeroshot_official.py`、`51_build_musdb_semantic.py` 保留历史准备步骤，默认在仓库根目录写入数据和结果；沿用时可以把仓库根目录作为 `--assets`。

`scripts/22_clap_zeroshot_official.py --split all --windows 3 --templates 4` 生成 `clap_official_emb_all25863_w3.npy`。核对 annotations 顺序和索引后，另存为 fp16 文件：

```python
import numpy as np
from pathlib import Path

results = Path('results')
vectors = np.load(results / 'clap_official_emb_all25863_w3.npy')
assert vectors.shape == (25863, 512) and np.isfinite(vectors).all()
np.save(results / 'clap_official_emb_all25863_w3_fp16.npy', vectors.astype(np.float16))
```

历史脚本不是一键离线安装器，外部下载、缓存与许可需使用者自行准备。Linux 与 Windows 本机独立环境曾完成四接口验证，不代表全新系统或其他机器已验证；本次公开整理仅复核已有统计与无需大模型的检查。
