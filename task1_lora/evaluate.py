"""Evaluate a base model or a base+LoRA model on the held-out test set.

    # base model, short prompt (no instructions)
    python evaluate.py --model Qwen/Qwen2.5-0.5B --prompt-style short --tag base_short
    # base model, schema instructions + one example in the prompt (stronger baseline)
    python evaluate.py --model Qwen/Qwen2.5-0.5B --prompt-style instruct --tag base_instruct
    # fine-tuned (base + adapter), short prompt
    python evaluate.py --model Qwen/Qwen2.5-0.5B --adapter outputs/lora_r16 --tag finetuned

Writes results/predictions_<tag>.jsonl and results/metrics_<tag>.json
"""
import argparse
import json
import os

import torch

from common import PROMPT_STYLES, SCHEMA_KEYS, aggregate, extract_json, read_jsonl, score_one, write_jsonl
from hf_utils import load_causal_lm, load_tokenizer, pick_dtype


@torch.no_grad()
def generate_all(model, tok, prompts, device, bs, max_new_tokens):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        print(f"  generated {min(i + bs, len(prompts))}/{len(prompts)}")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir (omit for base model)")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--prompt-style", choices=list(PROMPT_STYLES), default="short")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.adapter or args.model)
    model = load_causal_lm(args.model, dtype=pick_dtype())
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    rows = read_jsonl(args.test)
    assert len(rows) >= 50, "assignment needs >= 50 test examples"
    prompt_fn = PROMPT_STYLES[args.prompt_style]
    raw = generate_all(model, tok, [prompt_fn(r["text"]) for r in rows], device, args.bs, args.max_new_tokens)

    preds, scores = [], []
    for r, text in zip(rows, raw):
        obj = extract_json(text)
        sc = score_one(obj, r["target"])
        scores.append(sc)
        preds.append({"id": r["id"], "split_type": r["split_type"], "text": r["text"],
                      "gold": r["target"], "raw_output": text, "pred": obj, "score": sc})

    metrics = {"smoke_test": bool(args.smoke), "tag": args.tag, "model": args.model,
               "adapter": args.adapter, "prompt_style": args.prompt_style,
               "overall": aggregate(scores)}
    for part in ("in_dist", "shifted"):
        metrics[part] = aggregate([s for s, p in zip(scores, preds) if p["split_type"] == part])

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, f"predictions_{args.tag}.jsonl"), preds)
    with open(os.path.join(args.out_dir, f"metrics_{args.tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    o = metrics["overall"]
    print(f"\n[{args.tag}] n={o['n']}  valid_json={o['valid_json']:.1%}  schema_ok={o['schema_ok']:.1%}  "
          f"exact_match={o['exact_match']:.1%}")
    for k in SCHEMA_KEYS:
        print(f"   field {k:<17} {o['field_' + k]:.1%}")


if __name__ == "__main__":
    main()
