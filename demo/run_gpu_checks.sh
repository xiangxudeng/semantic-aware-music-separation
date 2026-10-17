#!/usr/bin/env bash
# 在已准备数据、模型与CUDA环境的现有云机串行运行；任一步失败立即停止。
set -euo pipefail
cd "$(dirname "$0")"
: "${AI_MUSIC_ASSETS:?请设置数据与结果所在的ai_music目录}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA不可用"'

# 第四周正式规模配方：每任务上限8000，2轮，batch8；P40峰值12.11–12.56GB。
for seed in 1 2 3; do
  tag="w4_p40_full_seed${seed}"
  python scripts/62_lora_finetune.py --epochs 2 --batch-size 8 --max-per-task 8000 --seed "$seed" --tag "$tag"
  python scripts/63_eval_lora.py --adapter "results/lora_${tag}" --limit 0 --tag "$tag"
done

# 两个常量条件及同机参照；文本向量是原官方CLAP的固定文本编码。
for source in none track zero text; do
  mode=film
  [[ "$source" == none ]] && mode=none
  cond_source="$source"
  [[ "$source" == none ]] && cond_source=track
  tag="w4_p40_C_${source}"
  extra=()
  [[ "$source" == text ]] && extra=(--cond-text-embedding research_inputs/clap_fixed_music_text.npy)
  python scripts/52_train_conditioned.py --cond-source "$cond_source" --cond-mode "$mode" --cond-targets encoder.3 --no-amp --steps 300 --seed 0 --tag "$tag" "${extra[@]}"
  python scripts/52_train_conditioned.py --no-amp --eval-only --ckpt "$AI_MUSIC_ASSETS/results/cond_model_${tag}.th" --cond-source "$cond_source" --cond-mode "$mode" --cond-targets encoder.3 --val-tracks 50 --seed 0 --tag "$tag" "${extra[@]}"
done
