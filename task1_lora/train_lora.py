"""LoRA / QLoRA fine-tuning with a plain PyTorch loop (no Trainer, so every step is visible).

Example (Colab T4):
    python train_lora.py --model Qwen/Qwen2.5-0.5B --train data/train.jsonl --out outputs/lora_r16

QLoRA (4-bit base model, needs a CUDA GPU + bitsandbytes):
    python train_lora.py ... --qlora

Only the completion (the JSON) is trained on: prompt tokens get label -100.
"""
import argparse
import json
import math
import os
import random
import time

import torch

from common import completion_text, PROMPT_STYLES, read_jsonl
from hf_utils import load_causal_lm, load_tokenizer, pick_dtype

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def build_examples(rows, tok, max_len):
    prompt_fn = PROMPT_STYLES["short"]
    examples, truncated = [], 0
    for r in rows:
        p = tok(prompt_fn(r["text"]), add_special_tokens=False)["input_ids"]
        c = tok(completion_text(r["target"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids, labels = p + c, [-100] * len(p) + c
        if len(ids) > max_len:
            truncated += 1
            ids, labels = ids[:max_len], labels[:max_len]
        examples.append((ids, labels))
    if truncated:
        print(f"warning: {truncated} examples were truncated to max_len={max_len}")
    return examples


def collate(batch, pad_id):
    longest = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="outputs/lora_r16")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (debugging only)")
    ap.add_argument("--smoke", action="store_true", help="mark results as a smoke test")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()
    print(f"device={device} dtype={dtype}")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = load_tokenizer(args.model)
    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = load_causal_lm(args.model, dtype=dtype, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    else:
        model = load_causal_lm(args.model, dtype=dtype)
        model.to(device)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=[t.strip() for t in args.targets.split(",")],
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for prm in model.parameters():          # adapters are trained in fp32 (needed by the fp16 GradScaler on a T4)
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    rows = read_jsonl(args.train)
    examples = build_examples(rows, tok, args.max_len)
    steps_per_epoch = math.ceil(len(examples) / (args.bs * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(args.warmup_frac * total_steps))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):  # linear warmup then cosine decay to 10% of peak
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if (use_amp and dtype == torch.float16) else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    log, step, t0 = [], 0, time.time()
    done = False
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        micro = [order[i:i + args.bs] for i in range(0, len(order), args.bs)]
        for mi, idxs in enumerate(micro):
            ids, attn, labels = collate([examples[i] for i in idxs], tok.pad_token_id)
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            (loss / args.grad_accum).backward() if scaler is None else scaler.scale(loss / args.grad_accum).backward()

            if (mi + 1) % args.grad_accum == 0 or mi == len(micro) - 1:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"epoch {epoch+1}/{args.epochs} step {step}/{total_steps} "
                          f"loss {loss.item():.4f} lr {lr_at(step):.2e}")
                log.append({"step": step, "epoch": epoch + 1, "loss": loss.item()})
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
        if done:
            break

    elapsed = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    summary = {
        "smoke_test": bool(args.smoke),
        "base_model": args.model, "qlora": args.qlora, "device": device, "dtype": str(dtype),
        "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout,
                 "target_modules": args.targets.split(","), "lr": args.lr, "epochs": args.epochs,
                 "batch_size": args.bs * args.grad_accum, "max_len": args.max_len},
        "n_train_examples": len(examples),
        "trainable_params": n_train, "total_params": n_total,
        "trainable_pct": 100 * n_train / n_total,
        "train_seconds": elapsed, "peak_gpu_mem_gb": peak_gb,
        "final_loss": log[-1]["loss"] if log else None,
        "first_loss": log[0]["loss"] if log else None,
    }
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "loss_log": log}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
