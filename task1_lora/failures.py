"""List where the fine-tuned model is still wrong, so YOU can pick 3 and explain why.

    python failures.py --tag finetuned --n 12

Writes results/failures_<tag>.md. The report needs 3 failures + your own explanation of
why each happens. Look for patterns (e.g. spoken times, unseen date formats) and check
them against the training data before you write the explanation.
"""
import argparse
import json
import os

from common import SCHEMA_KEYS, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="finetuned")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    preds = read_jsonl(os.path.join(args.results_dir, f"predictions_{args.tag}.jsonl"))
    wrong = [p for p in preds if not p["score"]["exact_match"]]
    print(f"{len(wrong)} / {len(preds)} examples are not an exact match")

    lines = [f"# Failures for `{args.tag}` ({len(wrong)}/{len(preds)} not exact)\n"]
    for p in wrong[:args.n]:
        bad = [k for k in SCHEMA_KEYS if not p["score"]["fields"][k]]
        lines += [f"## {p['id']}  [{p['split_type']}]",
                  f"**Request:** {p['text']}", "",
                  f"- gold: `{json.dumps(p['gold'])}`",
                  f"- pred: `{json.dumps(p['pred'])}`" if p["pred"] is not None
                  else f"- pred: (no valid JSON) raw=`{p['raw_output'][:200]!r}`",
                  f"- wrong fields: {', '.join(bad) if bad else 'schema/type problem only'}", ""]
    out = os.path.join(args.results_dir, f"failures_{args.tag}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten to {out}")

    # quick pattern summary: which field is wrong most often, split by test type
    from collections import Counter
    c = Counter()
    for p in wrong:
        for k in SCHEMA_KEYS:
            if not p["score"]["fields"][k]:
                c[(p["split_type"], k)] += 1
    print("\nwrong-field counts (split, field):")
    for (s, k), v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {s:<8} {k:<17} {v}")


if __name__ == "__main__":
    main()
