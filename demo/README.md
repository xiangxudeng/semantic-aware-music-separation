# 最终演示与复现

本目录是独立演示入口。根目录脚本及模块保留早期研究；本目录内 `scripts/` 与 `src/` 对应最终实验实现，各阶段结果分别报告。

## 启动

从仓库根目录运行：

```bash
python -m pip install -r demo/requirements.txt
python demo/app.py --assets /path/to/assets --device cpu
```

Windows 也可运行 `demo\start_demo.cmd --assets "E:\music-assets" --device cpu`。浏览器访问 `http://127.0.0.1:7860`，输出在 `demo/outputs/`。模型、标签和索引需提前按[资产说明](../docs/assets.md)准备；创作助手只读取本地 Qwen2 缓存。

## 检查与统计

```bash
python demo/final_statistics.py --self-check
python demo/objective_review.py --self-check
python demo/scripts/52_train_conditioned.py --self-check
python demo/final_statistics.py --lora benchmarks/lora --conditioned benchmarks/conditioned --output demo/outputs/statistics
```

前三项无需模型或数据。最后一项复算三种子汇总及匿名逐曲对照，不重新训练模型或计算逐音频 AP。

准备好资产并设置 `AI_MUSIC_ASSETS` 后，`python demo/checks.py` 可检查检索排序、索引对齐、统计、输入边界、数据增强、评分校验与创作草稿降级。

## 训练

Linux CUDA 环境安装 `demo/requirements-train.txt`：

```bash
export AI_MUSIC_ASSETS=/path/to/assets
python demo/scripts/60_build_instruction_data.py
bash demo/run_gpu_checks.sh
```

脚本重跑正式 LoRA 三种子及四组条件对照，失败即停。LoRA 配方为 batch 8、两轮、每任务上限 8,000，历史正式运行实际 18,608 条指令；P40 峰值 12.11–12.56 GB，不能按 8 GB 预算照搬。

`research_inputs/` 提供固定通用文本的 512 维 CLAP 向量及来源哈希，它不含音频特征。其余音频向量、指令数据与模型权重需自行准备。

## 评测

```bash
python demo/evaluate.py --assets /path/to/assets --output demo/outputs/template-search
python demo/evaluate_free.py --assets /path/to/assets --output demo/outputs/free-search
python demo/objective_review.py --mixture /path/to/mixture.wav --stems /path/to/stems --output demo/outputs/audio-review
```

分轨目录需含 `vocals.wav`、`drums.wav`、`bass.wav`、`other.wav`。客观代理指标不能替代配对参考分轨的 SDR/SI-SDR 或人工音质评价。`evaluate_listening.py` 可处理使用者自己的评分工作簿；原始个人评分材料未公开。
