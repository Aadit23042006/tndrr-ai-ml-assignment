"""Back-of-envelope plan: serve N concurrent users on ONE 24 GB GPU. All assumptions are arguments.

    python serving_plan.py                       # 20 users, Qwen2.5-7B-class model, 4k context each
    python serving_plan.py --users 20 --ctx 8192 --gpu-name "RTX 4090" --bw-gbs 1008

Three questions, answered with formulas (printed as working):
  1. MEMORY  : weights + KV cache for all users + overhead  <=  usable GPU memory ?
  2. DECODE  : each generated token must read all weights (shared by the batch) + each user's KV cache
               from GPU memory -> speed is limited by memory bandwidth (roofline, upper bound).
  3. PREFILL : processing the prompts is compute-bound: ~2 x params FLOPs per token.
These are ESTIMATES. Real engines (vLLM, TensorRT-LLM) lose some efficiency; measure before promising anything.
"""
import argparse
import json
import math
import os


def plan(name, a, weight_bits, kv_bytes):
    budget = a.gpu_gb * a.util
    weights = a.params_b * 1e9 * weight_bits / 8 / 1e9
    per_tok = 2 * a.layers * a.kv_heads * a.head_dim * kv_bytes
    kv_user = per_tok * a.ctx / 1e9
    kv_all = kv_user * a.users
    total = weights + kv_all + a.overhead_gb
    max_users = max(0, math.floor((budget - weights - a.overhead_gb) / kv_user))
    # decode: average cache fill ~ half the max context
    step_bytes = weights * 1e9 + a.users * (a.ctx * a.avg_fill) * per_tok
    step_s = step_bytes / (a.bw_gbs * 1e9 * a.eff)
    prefill_s = 2 * a.params_b * 1e9 * a.prompt_tokens / (a.tflops * 1e12 * a.mfu)
    r = {"scenario": name, "weights_gb": weights, "kv_per_token_kib": per_tok / 1024, "kv_per_user_gb": kv_user,
         "kv_all_users_gb": kv_all, "overhead_gb": a.overhead_gb, "total_gb": total, "budget_gb": budget,
         "fits": total <= budget, "max_users_by_memory": max_users,
         "decode_step_ms": step_s * 1e3, "per_user_tokens_per_s": 1 / step_s,
         "aggregate_tokens_per_s": a.users / step_s,
         "prefill_s_per_prompt": prefill_s, "prefill_s_if_all_arrive_together": prefill_s * a.users}
    print(f"--- {name}")
    print(f"  weights      = {a.params_b}B x {weight_bits} bits / 8                    = {weights:6.2f} GB")
    print(f"  KV per token = 2 x {a.layers} x {a.kv_heads} x {a.head_dim} x {kv_bytes} B             = {per_tok/1024:6.1f} KiB")
    print(f"  KV all users = {a.users} users x {a.ctx} tokens x {per_tok/1024:.1f} KiB     = {kv_all:6.2f} GB")
    print(f"  overhead     = CUDA context, activations, graphs (assumed)    = {a.overhead_gb:6.2f} GB")
    print(f"  TOTAL        = {total:6.2f} GB   vs usable {a.gpu_gb:g} GB x {a.util:g} = {budget:5.2f} GB  ->  "
          f"{'FITS' if r['fits'] else 'DOES NOT FIT'}  (max ~{max_users} users at this context)")
    print(f"  decode       : step reads {step_bytes/1e9:5.2f} GB / ({a.bw_gbs:g} GB/s x {a.eff:g}) = {step_s*1e3:5.1f} ms"
          f"  -> {1/step_s:5.1f} tok/s per user, {a.users/step_s:6.0f} tok/s total")
    print(f"  prefill      : 2 x {a.params_b}B x {a.prompt_tokens} tok / ({a.tflops:g} TFLOPs x {a.mfu:g}) = "
          f"{prefill_s:4.2f} s per prompt ({prefill_s*a.users:4.1f} s if all {a.users} arrive at once)\n")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=4096, help="max tokens (prompt+answer) kept per user")
    ap.add_argument("--prompt-tokens", type=int, default=1000, help="typical prompt length for the prefill estimate")
    ap.add_argument("--avg-fill", type=float, default=0.5, help="average fraction of ctx actually filled")
    ap.add_argument("--gpu-name", default="24 GB card (e.g. RTX 4090 / 3090 / L4-class)")
    ap.add_argument("--gpu-gb", type=float, default=24)
    ap.add_argument("--util", type=float, default=0.90, help="fraction of GPU memory the server may use")
    ap.add_argument("--bw-gbs", type=float, default=1008, help="memory bandwidth GB/s (4090: 1008, 3090: 936, L4: 300)")
    ap.add_argument("--eff", type=float, default=0.5, help="achieved fraction of peak bandwidth (assumption)")
    ap.add_argument("--tflops", type=float, default=82, help="usable FP16 TFLOPs (assumption)")
    ap.add_argument("--mfu", type=float, default=0.4, help="fraction of peak FLOPs reached in prefill")
    ap.add_argument("--overhead-gb", type=float, default=2.0)
    ap.add_argument("--params-b", type=float, default=7.6, help="billions of parameters (Qwen2.5-7B: 7.6)")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out", default="results/serving_plan.json")
    a = ap.parse_args()

    print(f"Serving {a.users} concurrent users on one {a.gpu_name}; model ~{a.params_b}B params "
          f"({a.layers} layers, {a.kv_heads} KV heads, head_dim {a.head_dim})\n")
    scen = [plan("A) FP16 weights, FP16 KV", a, 16, 2),
            plan("B) 4-bit weights (AWQ/GPTQ ~4.5 bits incl. scales), FP16 KV", a, 4.5, 2),
            plan("C) 4-bit weights, FP8 KV cache", a, 4.5, 1)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"assumptions": vars(a), "scenarios": scen}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
