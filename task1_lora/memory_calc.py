"""Back-of-envelope GPU memory for fine-tuning a decoder LLM: full FT vs LoRA vs QLoRA.

    python memory_calc.py                          # 7B Llama-style model (the assignment question)
    python memory_calc.py --params 0.494e9 --layers 24 --hidden 896 --inter 4864 \
                          --heads 14 --kv-heads 2 --name Qwen2.5-0.5B
        -> compare with peak_gpu_mem_gb that train_lora.py measured on your own GPU run

All numbers are ESTIMATES (they ignore CUDA context ~0.5-1 GB, fragmentation, and the
exact activation footprint). The point is to show where the memory goes.
Writes results/memory_calc.json and prints the working.
"""
import argparse
import json
import os

GB = 1e9


def lora_params(layers, hidden, inter, heads, kv_heads, r):
    """LoRA on q,k,v,o,gate,up,down. A LoRA pair on a (d_in -> d_out) layer adds r*(d_in+d_out)."""
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    per_layer = (
        r * (hidden + hidden)        # q_proj  hidden -> hidden
        + r * (hidden + kv_dim)      # k_proj  hidden -> kv_dim
        + r * (hidden + kv_dim)      # v_proj  hidden -> kv_dim
        + r * (hidden + hidden)      # o_proj  hidden -> hidden
        + r * (hidden + inter)       # gate_proj
        + r * (hidden + inter)       # up_proj
        + r * (inter + hidden)       # down_proj
    )
    return layers * per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="7B (Llama-2-7B shapes)")
    ap.add_argument("--params", type=float, default=6.74e9)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=11008)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=32)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--act-gb", type=float, default=4.0,
                    help="ASSUMED activation memory (batch ~1-4 x 512 tokens, gradient checkpointing on)")
    ap.add_argument("--out", default="results/memory_calc.json")
    args = ap.parse_args()

    P = args.params
    lp = lora_params(args.layers, args.hidden, args.inter, args.heads, args.kv_heads, args.rank)
    act = args.act_gb
    print(f"Model: {args.name}   P = {P/1e9:.2f}B parameters")
    print(f"LoRA r={args.rank} on q,k,v,o,gate,up,down -> trainable params L = {lp/1e6:.1f}M "
          f"({100*lp/P:.2f}% of P)\n")

    # ---- (a) full fine-tuning, mixed precision (bf16 weights/grads + fp32 Adam + fp32 master copy)
    w16, g16 = 2 * P, 2 * P
    adam = 8 * P            # m and v in fp32: 4 + 4 bytes per parameter
    master = 4 * P          # fp32 master copy of the weights
    full = w16 + g16 + adam + master + act * GB
    # pure-bf16 variant (no master copy) as a lower bound
    full_lo = w16 + g16 + adam + act * GB
    print("(a) FULL fine-tuning (mixed precision, AdamW)")
    print(f"    weights  bf16      2 B x P = {w16/GB:7.1f} GB")
    print(f"    gradients bf16     2 B x P = {g16/GB:7.1f} GB")
    print(f"    Adam m+v fp32      8 B x P = {adam/GB:7.1f} GB")
    print(f"    fp32 master copy   4 B x P = {master/GB:7.1f} GB")
    print(f"    activations (assumed)      = {act:7.1f} GB")
    print(f"    TOTAL ~ {full/GB:.0f} GB   (~{full_lo/GB:.0f} GB if you skip the fp32 master copy)\n")

    # ---- (b) LoRA: base frozen in bf16, only adapters get grads + optimizer state
    lw = 2 * P
    l_train = lp * (4 + 4 + 8)   # fp32 adapter weights + fp32 grads + fp32 Adam m,v
    lora = lw + l_train + act * GB
    print("(b) LoRA (base frozen in bf16, adapters trained in fp32)")
    print(f"    frozen base weights bf16   2 B x P = {lw/GB:7.1f} GB")
    print(f"    adapters: weights+grads+Adam  16 B x L = {l_train/GB:7.2f} GB")
    print(f"    activations (assumed)              = {act:7.1f} GB")
    print(f"    TOTAL ~ {lora/GB:.0f} GB\n")

    # ---- (c) QLoRA: base in 4-bit NF4 (+ double-quantised constants)
    q_bits = 4 + 0.127           # ~0.127 bit/param overhead with double quantisation
    qw = P * q_bits / 8
    qlora = qw + l_train + act * GB
    print("(c) QLoRA (base frozen in 4-bit NF4, adapters in fp32)")
    print(f"    base weights 4-bit   ~{q_bits:.2f} bit x P / 8 = {qw/GB:7.1f} GB")
    print(f"    adapters: weights+grads+Adam  16 B x L = {l_train/GB:7.2f} GB")
    print(f"    activations (assumed)              = {act:7.1f} GB")
    print(f"    TOTAL ~ {qlora/GB:.0f} GB\n")

    print("Rule of thumb check: full FT ~ 16 B/param, LoRA ~ 2 B/param + activations, "
          "QLoRA ~ 0.5 B/param + activations.")

    result = {
        "model": args.name, "params": P, "lora_rank": args.rank, "lora_params": lp,
        "assumed_activation_gb": act,
        "full_ft_gb": full / GB, "full_ft_no_master_gb": full_lo / GB,
        "lora_gb": lora / GB, "qlora_gb": qlora / GB,
        "breakdown_gb": {
            "weights_bf16": w16 / GB, "grads_bf16": g16 / GB, "adam_fp32": adam / GB,
            "master_fp32": master / GB, "lora_trainable_state": l_train / GB, "qlora_weights_4bit": qw / GB,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
