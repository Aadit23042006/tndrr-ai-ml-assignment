"""Side-by-side quality comparison of the benchmarked settings (needs results/benchmark.json).

    python compare_quality.py

Writes results/quality_comparison.md. The automatic numbers (checks passed, how many outputs are
identical, text similarity) are only a starting point: the REPORT needs YOUR observations after you
read the outputs, e.g. where the 4-bit model got a fact, format or reasoning step wrong.
"""
import difflib
import json
import os


def main(path="results/benchmark.json", out="results/quality_comparison.md"):
    data = json.load(open(path))
    settings = data["settings"]
    base, others = settings[0], settings[1:]
    lines = ["# Output quality comparison (same 10 prompts, temperature 0)\n"]
    lines.append("| setting | auto-checks passed | mean tok/s | memory (Ollama /api/ps) |")
    lines.append("|---|---|---|---|")
    for s in settings:
        mem = s["memory"].get("ollama_ps_total_gb")
        lines.append(f"| `{s['model']}` | {s['auto_checks_passed']}/{s['auto_checks_total']} | "
                     f"{s['mean_tokens_per_s']:.1f} | {mem:.2f} GB |" if mem is not None else
                     f"| `{s['model']}` | {s['auto_checks_passed']}/{s['auto_checks_total']} | "
                     f"{s['mean_tokens_per_s']:.1f} | n/a |")
    lines.append("")
    for o in others:
        sims = []
        identical = 0
        for pb, po in zip(base["prompts"], o["prompts"]):
            r = difflib.SequenceMatcher(None, pb["output"], po["output"]).ratio()
            sims.append(r)
            identical += pb["output"].strip() == po["output"].strip()
        lines.append(f"**`{o['model']}` vs `{base['model']}`:** {identical}/10 identical outputs, "
                     f"mean text similarity {sum(sims)/len(sims):.2f}\n")

    for i, pb in enumerate(base["prompts"]):
        lines.append(f"## Prompt {pb['id']}\n> {pb['prompt']}\n")
        for s in settings:
            p = s["prompts"][i]
            flag = {True: " (check passed)", False: " (check FAILED)", None: ""}[p["check_passed"]]
            lines.append(f"**{s['model']}**{flag}\n```\n{p['output'].strip()}\n```\n")
        lines.append("**My observation:** _(write this yourself after reading both outputs)_\n")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
