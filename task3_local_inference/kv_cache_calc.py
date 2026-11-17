"""KV-cache size calculator.

    python kv_cache_calc.py                                  # table for some well-known models
    python kv_cache_calc.py --layers 28 --kv-heads 2 --head-dim 128 --ctx 2048 --users 1
    python kv_cache_calc.py --config path/to/config.json --ctx 4096 --users 20

Formula (per token):   2 (K and V) x layers x kv_heads x head_dim x bytes_per_value
(with grouped-query attention, kv_heads < attention heads, which shrinks the cache a lot).
VERIFY the architecture numbers against each model's own config.json before quoting them.
"""
import argparse
import json

PRESETS = {  # name: (layers, kv_heads, head_dim)   -- from the models' config.json; double-check yours
    "Qwen2.5-0.5B": (24, 2, 64),
    "Qwen2.5-1.5B": (28, 2, 128),
    "Qwen2.5-7B": (28, 4, 128),
    "Llama-3.1-8B": (32, 8, 128),
    "Llama-2-7B (full MHA)": (32, 32, 128),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2.0):
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def from_config(path):
    c = json.load(open(path))
    heads = c["num_attention_heads"]
    return (c["num_hidden_layers"], c.get("num_key_value_heads", heads),
            c.get("head_dim") or c["hidden_size"] // heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--config")
    ap.add_argument("--bytes", type=float, default=2.0, help="bytes per cached value: 2=fp16, 1=fp8/int8")
    ap.add_argument("--ctx", type=int, default=4096, help="tokens kept in cache per user")
    ap.add_argument("--users", type=int, default=1)
    a = ap.parse_args()

    if a.config:
        a.layers, a.kv_heads, a.head_dim = from_config(a.config)
    if a.layers:
        rows = {"(yours)": (a.layers, a.kv_heads, a.head_dim)}
    else:
        rows = PRESETS
    print(f"KV cache, {a.bytes:g} bytes/value, context {a.ctx} tokens, {a.users} user(s)\n")
    print(f"{'model':<24}{'L':>4}{'kvH':>5}{'hd':>5}{'per token':>12}{'per user':>12}{'all users':>12}")
    for name, (L, H, D) in rows.items():
        b = kv_bytes_per_token(L, H, D, a.bytes)
        print(f"{name:<24}{L:>4}{H:>5}{D:>5}{b/1024:>9.1f} KiB{b*a.ctx/1e9:>9.2f} GB{b*a.ctx*a.users/1e9:>9.2f} GB")
    L, H, D = next(iter(rows.values()))
    print(f"\nworking for the first row: 2 x {L} layers x {H} kv_heads x {D} head_dim x {a.bytes:g} B "
          f"= {kv_bytes_per_token(L, H, D, a.bytes):,.0f} bytes per token")


if __name__ == "__main__":
    main()
