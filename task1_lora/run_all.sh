#!/usr/bin/env bash
# Task 1, everything in order. On a free Colab T4 this takes roughly 10-20 minutes for 0.5B.
#   bash run_all.sh                       # Qwen2.5-0.5B
#   MODEL=Qwen/Qwen2.5-1.5B bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"

python make_dataset.py                                            # data/train.jsonl, data/test.jsonl

# 1) baselines: the base model, with and without instructions in the prompt
python evaluate.py --model "$MODEL" --prompt-style short    --tag base_short
python evaluate.py --model "$MODEL" --prompt-style instruct --tag base_instruct

# 2) fine-tune (r=16, alpha=32, all linear layers, lr 2e-4, 3 epochs)
python train_lora.py --model "$MODEL" --out outputs/lora_r16

# 3) fine-tuned model on the same held-out test set
python evaluate.py --model "$MODEL" --adapter outputs/lora_r16 --tag finetuned

# 4) failure cases to analyse + memory calculation
python failures.py --tag finetuned
python memory_calc.py                                             # the 7B question
echo "Done. Look in results/ and outputs/lora_r16/train_summary.json"
