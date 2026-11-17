"""Offline tests for Task 3 using a MOCK Ollama server (real Ollama/GPU not needed).
Run:  python tests/test_task3.py"""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2, T3 = os.path.join(ROOT, "task2_rag"), os.path.join(ROOT, "task3_local_inference")
sys.path.insert(0, os.path.join(ROOT, "tests"))

# model -> (decode seconds for 50 tokens, loaded size bytes)
MODELS = {"mock:fast": (0.5, 3_100_000_000), "mock:slow": (1.0, 1_000_000_000)}
ANSWERS = {"17 * 23": "391", "capital of Australia": "Canberra", "is_prime": "def is_prime(n):\n    return n>1",
           "Pride and Prejudice": "It was written by Jane Austen.", "60 km in 45": "80 km/h"}
LOADED = set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m} for m in MODELS]})
        elif self.path == "/api/ps":
            self._send({"models": [{"name": m, "size": MODELS[m][1], "size_vram": MODELS[m][1]} for m in LOADED]})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        m = req["model"]
        if req.get("keep_alive") == 0:
            LOADED.discard(m); return self._send({})
        LOADED.add(m)
        text = "Sure."
        for k, v in ANSWERS.items():
            if k in req["prompt"]:
                text = v
        if "JSON" in req["prompt"]:
            text = 'Here you go: {"name": "Asha", "age": 30}'
        if "api/generate" in self.path:
            if "rag-context" in req.get("_x", ""):
                pass
            if req["prompt"].startswith("Answer the question using ONLY"):
                text = "validators [alpha.pdf p.1]"
            return self._send({"response": text, "eval_count": 50, "eval_duration": int(MODELS[m][0] * 1e9),
                               "prompt_eval_count": 20, "prompt_eval_duration": int(0.1e9), "load_duration": int(2e9)})


def start():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def run(args, cwd=T3):
    return subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)


def test_benchmark_and_compare():
    srv, host = start()
    with tempfile.TemporaryDirectory() as tmp:
        out = tmp + "/benchmark.json"
        r = run(["benchmark_ollama.py", "--models", "mock:fast", "mock:slow", "--host", host,
                 "--out", out, "--repeats", "2", "--smoke"])
        assert r.returncode == 0, r.stdout + r.stderr
        d = json.load(open(out))
        assert d["smoke_test"] is True and [s["model"] for s in d["settings"]] == ["mock:fast", "mock:slow"]
        fast, slow = d["settings"]
        assert abs(fast["mean_tokens_per_s"] - 100) < 1e-6 and abs(slow["mean_tokens_per_s"] - 50) < 1e-6
        assert abs(fast["memory"]["ollama_ps_total_gb"] - 3.1) < 1e-9 and abs(slow["memory"]["ollama_ps_total_gb"] - 1.0) < 1e-9
        assert len(fast["prompts"]) == 10 and fast["load_seconds"] == 2.0
        # auto checks: prompts 1,2,3,6,8,10 are answered correctly by the mock; 4 (gare) is not -> 6/7
        assert (fast["auto_checks_passed"], fast["auto_checks_total"]) == (6, 7), fast["auto_checks_passed"]
        r = run(["compare_quality.py"], cwd=T3) if False else subprocess.run(
            [sys.executable, "-c", f"import compare_quality as c; c.main(r'{out}', r'{tmp}/q.md')"],
            cwd=T3, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        md = open(tmp + "/q.md").read()
        assert "10/10 identical" in md and "mock:fast" in md and "My observation" in md
    srv.shutdown()


def test_missing_model_message():
    srv, host = start()
    r = run(["benchmark_ollama.py", "--models", "does-not-exist:1b", "--host", host, "--out", "/tmp/x.json"])
    assert r.returncode != 0 and "ollama pull does-not-exist:1b" in (r.stdout + r.stderr)
    srv.shutdown()


def test_ask_with_mock_llm():
    from test_task2 import make_corpus
    srv, host = start()
    with tempfile.TemporaryDirectory() as tmp:
        facts = make_corpus(tmp + "/docs", n_pages=12)
        dname, q, sent, n = facts[5]
        sys.path.insert(0, T2)
        import ask  # noqa
        from rag_lib import Retriever, load_corpus, HashEmbedder
        ret = Retriever(load_corpus(tmp + "/docs"), 60, 12, HashEmbedder())
        prompt, ans = ask.answer(ret, q, 5, "bm25", None)
        assert ans is None and "Context:" in prompt and sent[:40] in prompt        # gold sentence reached the prompt
        orig = ask.call_ollama
        ask.call_ollama = lambda model, p, host_=host: orig(model, p, host_)
        prompt, ans = ask.answer(ret, q, 5, "bm25", "ollama:mock:fast")
        assert "validators" in ans
    srv.shutdown()


def test_calculators():
    out = subprocess.run([sys.executable, "serving_plan.py", "--out", tempfile.mktemp()], cwd=T3,
                         capture_output=True, text=True, check=True).stdout
    assert out.count("DOES NOT FIT") == 1 and out.count("->  FITS") == 2 and "max ~65 users" in out
    kv = subprocess.run([sys.executable, "kv_cache_calc.py", "--layers", "28", "--kv-heads", "4", "--head-dim", "128",
                         "--ctx", "4096", "--users", "20"], cwd=T3, capture_output=True, text=True, check=True).stdout
    assert "57,344 bytes per token" in kv and "4.70 GB" in kv
    # architecture from a config.json
    cfg = tempfile.mktemp(suffix=".json")
    json.dump({"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}, open(cfg, "w"))
    kv = subprocess.run([sys.executable, "kv_cache_calc.py", "--config", cfg, "--ctx", "1000"], cwd=T3,
                        capture_output=True, text=True, check=True).stdout
    assert "131,072 bytes per token" in kv


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa
                import traceback; traceback.print_exc()
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
