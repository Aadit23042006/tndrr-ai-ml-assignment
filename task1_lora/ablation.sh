#!/usr/bin/env bash
# Optional: evidence for "why these LoRA settings". Trains r=4 and r=64 (alpha = 2r) and evaluates both.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
for R in 4 64; do
  python train_lora.py --model "$MODEL" --r $R --alpha $((2*R)) --out outputs/lora_r$R
  python evaluate.py --model "$MODEL" --adapter outputs/lora_r$R --tag finetuned_r$R
done
python train_lora.py --model "$MODEL" --lr 5e-5 --out outputs/lora_r16_lr5e-5
python evaluate.py --model "$MODEL" --adapter outputs/lora_r16_lr5e-5 --tag finetuned_r16_lr5e-5
