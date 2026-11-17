"""Benchmark one or more Ollama models (e.g. 16-bit vs 4-bit) on the same 10 prompts.

    python benchmark_ollama.py --models qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M

For every model it measures
  * decode speed in tokens/second  (eval_count / eval_duration, reported by Ollama itself)
  * prompt-processing speed and load time
  * memory held by the loaded model (Ollama /api/ps  +  nvidia-smi if there is an NVIDIA GPU,
    else the RSS of the ollama process)
  * the full output text for every prompt (for the quality comparison)
Settings are deterministic (temperature 0, fixed seed) so both models see identical conditions.

Writes results/benchmark.json
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.request


def http(host, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_info():
    """(name, memory_used_MiB) of GPU 0 via nvidia-smi, or (None, None)."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, used = [x.strip() for x in out.split(",")]
        return name, float(used)
    except Exception:
        return None, None


def ollama_rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        if "ollama" in (p.info["name"] or "").lower() and p.info["memory_info"]:
            total += p.info["memory_info"].rss
    return total / 2**20 if total else None


def check_output(prompt_row, text):
    """Automatic correctness check where the prompt has a verifiable answer. None = not checkable."""
    low = text.lower()
    if "expect_any" in prompt_row:
        return any(s.lower() in low for s in prompt_row["expect_any"])
    if "json_keys" in prompt_row:
        from_idx = text.find("{")
        to_idx = text.rfind("}")
        try:
            obj = json.loads(text[from_idx:to_idx + 1])
            return all(k in obj for k in prompt_row["json_keys"])
        except Exception:
            return False
    return None


def unload(host, model):
    try:
        http(host, "/api/generate", {"model": model, "prompt": "", "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def bench_model(host, model, prompts, num_predict, num_ctx, repeats):
    print(f"\n=== {model} ===")
    unload(host, model)
    time.sleep(1)
    _, gpu_before = gpu_info()

    # warm-up: loads the model into memory; load time reported separately, not part of tok/s
    t0 = time.time()
    warm = http(host, "/api/generate", {"model": model, "prompt": "Hello", "stream": False,
                                        "options": {"temperature": 0, "seed": 0, "num_predict": 8, "num_ctx": num_ctx}})
    load_s = warm.get("load_duration", 0) / 1e9 or (time.time() - t0)

    rows = []
    for p in prompts:
        speeds, last = [], None
        for _ in range(repeats):
            r = http(host, "/api/generate", {"model": model, "prompt": p["prompt"], "stream": False,
                                             "options": {"temperature": 0, "seed": 0,
                                                         "num_predict": num_predict, "num_ctx": num_ctx}})
            if r.get("eval_duration"):
                speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))
            last = r
        ok = check_output(p, last["response"])
        rows.append({"id": p["id"], "prompt": p["prompt"], "output": last["response"],
                     "eval_count": last.get("eval_count"), "tokens_per_s": statistics.mean(speeds) if speeds else None,
                     "prompt_eval_count": last.get("prompt_eval_count"),
                     "prompt_tokens_per_s": (last["prompt_eval_count"] / (last["prompt_eval_duration"] / 1e9))
                     if last.get("prompt_eval_duration") else None,
                     "check_passed": ok})
        print(f"  prompt {p['id']:>2}: {rows[-1]['tokens_per_s'] or 0:7.1f} tok/s  "
              f"({last.get('eval_count')} tokens)  check={ok}")

    # memory while the model is still loaded
    ps = {}
    try:
        for m in http(host, "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                ps = m
    except Exception:
        pass
    gpu_name, gpu_after = gpu_info()
    mem = {"ollama_ps_total_gb": ps.get("size", 0) / 1e9 if ps else None,
           "ollama_ps_vram_gb": ps.get("size_vram", 0) / 1e9 if ps else None,
           "nvidia_smi_delta_mib": (gpu_after - gpu_before) if (gpu_after is not None and gpu_before is not None) else None,
           "ollama_rss_mib": ollama_rss_mb()}

    tps = [r["tokens_per_s"] for r in rows if r["tokens_per_s"]]
    total_tokens = sum(r["eval_count"] or 0 for r in rows)
    checks = [r["check_passed"] for r in rows if r["check_passed"] is not None]
    summary = {"model": model, "load_seconds": load_s,
               "mean_tokens_per_s": statistics.mean(tps) if tps else None,
               "median_tokens_per_s": statistics.median(tps) if tps else None,
               "min_tokens_per_s": min(tps) if tps else None, "max_tokens_per_s": max(tps) if tps else None,
               "total_generated_tokens": total_tokens,
               "auto_checks_passed": sum(checks), "auto_checks_total": len(checks),
               "memory": mem, "num_ctx": num_ctx, "num_predict": num_predict, "prompts": rows}
    print(f"  -> mean {summary['mean_tokens_per_s']:.1f} tok/s, memory {mem}")
    unload(host, model)
    return summary, gpu_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-predict", type=int, default=200)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=2, help="runs per prompt; tok/s is averaged")
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    prompts = json.load(open(a.prompts, encoding="utf-8"))
    assert len(prompts) == 10, "assignment asks for the same 10 prompts"
    available = {m.get("name") for m in http(a.host, "/api/tags").get("models", [])}
    missing = [m for m in a.models if m not in available]
    if missing:
        raise SystemExit(f"model(s) not pulled yet: {missing}\n  run:  " + "  &&  ".join(f"ollama pull {m}" for m in missing)
                         + f"\n  (available: {sorted(available)})")

    settings, gpu_name = [], None
    for m in a.models:
        s, gpu_name = bench_model(a.host, m, prompts, a.num_predict, a.num_ctx, a.repeats)
        settings.append(s)

    result = {"smoke_test": bool(a.smoke), "hardware": {"gpu": gpu_name, "note": "fill in CPU/RAM/Colab type in the report"},
              "settings": settings, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
