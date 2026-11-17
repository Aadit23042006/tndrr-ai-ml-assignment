"""Offline tests for Task 1 logic (no GPU, no downloads).  Run:  python tests/test_task1.py"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1 = os.path.join(ROOT, "task1_lora")
sys.path.insert(0, T1)

from common import extract_json, score_one, schema_ok, aggregate, completion_text, short_prompt

GOLD = {"title": "lunch", "date": "2026-12-09", "time": "17:40", "duration_minutes": 60,
        "location": None, "attendees": ["Priya", "Vikram"]}


def test_extract_json():
    assert extract_json('junk {"a": 1} tail') == {"a": 1}
    assert extract_json('{"a": {"b": [1, 2]}} {"c": 2}') == {"a": {"b": [1, 2]}}
    assert extract_json('text {"s": "has } brace"} done') == {"s": "has } brace"}
    assert extract_json('{broken {"ok": true}') == {"ok": True}      # skips broken prefix
    assert extract_json("no json here") is None
    assert extract_json('{"unterminated": 1') is None
    assert extract_json("[1, 2, 3]") is None


def test_scoring():
    s = score_one(dict(GOLD), GOLD)
    assert s["valid_json"] and s["schema_ok"] and s["exact_match"]
    bad = dict(GOLD, time="17:41")
    s = score_one(bad, GOLD)
    assert s["schema_ok"] and not s["exact_match"] and not s["fields"]["time"] and s["fields"]["date"]
    assert not schema_ok(dict(GOLD, date="9 Dec 2026"))                  # wrong format
    assert not schema_ok(dict(GOLD, duration_minutes="60"))              # wrong type
    assert not schema_ok(dict(GOLD, duration_minutes=True))              # bool is not int
    extra = dict(GOLD, bonus=1)
    assert not schema_ok(extra) and not score_one(extra, GOLD)["exact_match"]
    none = score_one(None, GOLD)
    assert not none["valid_json"] and not none["exact_match"]
    agg = aggregate([score_one(dict(GOLD), GOLD), score_one(None, GOLD)])
    assert agg["n"] == 2 and agg["exact_match"] == 0.5 and agg["valid_json"] == 0.5


def test_dataset():
    from make_dataset import build
    train = build(300, 1, shifted=False)
    test_in = build(50, 2, shifted=False, forbidden={r["text"] for r in train})
    test_sh = build(30, 3, shifted=True, forbidden={r["text"] for r in train})
    assert not ({r["text"] for r in train} & {r["text"] for r in test_in + test_sh})
    assert len({r["text"] for r in train}) == 300
    again = build(300, 1, shifted=False)
    assert [r["text"] for r in again] == [r["text"] for r in train]      # deterministic
    for r in train + test_in + test_sh:
        assert schema_ok(r["target"]), r
        assert r["target"]["title"] in r["text"]
        for n in r["target"]["attendees"]:
            assert n in r["text"]
        if r["target"]["location"]:
            assert r["target"]["location"] in r["text"]
    # spoken-time arithmetic spot checks
    seen = {r["text"]: r["target"]["time"] for r in train + test_sh}
    for text, t in seen.items():
        if "half past" in text:
            assert t.endswith(":30")
        if "quarter past" in text:
            assert t.endswith(":15")
        if "quarter to" in text:
            assert t.endswith(":45")
        if " noon" in text:
            assert t == "12:00"


def test_label_masking():
    try:
        import torch  # noqa
        from transformers import PreTrainedTokenizerFast  # noqa
    except Exception:
        print("  (skipped: torch/transformers not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        subprocess.run([sys.executable, os.path.join(T1, "make_dataset.py"), "--out-dir", d + "/data"],
                       check=True, capture_output=True, cwd=T1, env=env)
        subprocess.run([sys.executable, os.path.join(T1, "smoke_model.py"), "--data", d + "/data/train.jsonl",
                        "--out", d + "/tiny"], check=True, capture_output=True, cwd=T1, env=env)
        from transformers import AutoTokenizer
        from train_lora import build_examples, collate
        from common import read_jsonl
        tok = AutoTokenizer.from_pretrained(d + "/tiny")
        rows = read_jsonl(d + "/data/train.jsonl")[:20]
        for r, (ids, labels) in zip(rows, build_examples(rows, tok, 256)):
            n_prompt = len(tok(short_prompt(r["text"]), add_special_tokens=False)["input_ids"])
            assert all(l == -100 for l in labels[:n_prompt])               # prompt not trained on
            trained = [l for l in labels if l != -100]
            assert trained[-1] == tok.eos_token_id                          # ends with EOS
            assert tok.decode(trained[:-1]) == completion_text(r["target"])  # exactly the JSON
        ids, attn, lab = collate([([1, 2, 3], [-100, 2, 3]), ([1], [1])], pad_id=0)
        assert ids.shape == (2, 3) and attn.tolist() == [[1, 1, 1], [1, 0, 0]] and lab[1, 1] == -100


def test_memory_calc():
    out = subprocess.run([sys.executable, os.path.join(T1, "memory_calc.py"), "--out", tempfile.mktemp()],
                         capture_output=True, text=True, cwd=T1, check=True).stdout
    assert "40.0M" in out and "TOTAL ~ 112 GB" in out and "TOTAL ~ 18 GB" in out and "TOTAL ~ 8 GB" in out


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
