"""End-to-end test of report/build_report.py using FIXTURE data produced by the real scripts on
tiny/mock inputs inside a temp dir (nothing from here is ever shipped or reused as real results).
Run:  python tests/test_report.py"""
import json, os, random, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1, T2, T3 = (os.path.join(ROOT, d) for d in ("task1_lora", "task2_rag", "task3_local_inference"))
sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, T2)

from test_task2 import make_corpus
from test_task3 import start, run as run3


def sh(args, cwd):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    return r


def make_fixture(root):
    # ---- task 1 (tiny random model; real scripts, real output shapes)
    d = root + "/_t1"
    sh(["make_dataset.py", "--out-dir", d + "/data"], T1)
    sh(["smoke_model.py", "--data", d + "/data/train.jsonl", "--out", d + "/tiny"], T1)
    sh(["train_lora.py", "--model", d + "/tiny", "--train", d + "/data/train.jsonl", "--out", root + "/task1_lora/outputs/lora_r16",
        "--max-steps", "15"], T1)
    res1 = root + "/task1_lora/results"
    for tag, extra in (("base_short", ["--prompt-style", "short"]), ("base_instruct", ["--prompt-style", "instruct"]),
                       ("finetuned", ["--adapter", root + "/task1_lora/outputs/lora_r16"])):
        sh(["evaluate.py", "--model", d + "/tiny", "--test", d + "/data/test.jsonl", "--tag", tag, "--out-dir", res1,
            "--max-new-tokens", "40"] + extra, T1)
    sh(["memory_calc.py", "--out", res1 + "/memory_calc.json"], T1)
    # ---- task 2 (synthetic corpus, hash embedder)
    facts = make_corpus(root + "/_docs", 36)
    qs = [{"id": i + 1, "question": q, "doc": dn, "gold_passage": s, "answer": str(n)}
          for i, (dn, q, s, n) in enumerate(random.Random(3).sample(facts, 15))]
    os.makedirs(root + "/task2_rag", exist_ok=True)
    json.dump(qs, open(root + "/task2_rag/questions.json", "w"))
    sh(["evaluate_retrieval.py", "--docs", root + "/_docs", "--questions", root + "/task2_rag/questions.json", "--embedder", "hash",
        "--chunk-sizes", "60", "200", "--out-dir", root + "/task2_rag/results"], T2)
    # ---- task 3 (mock ollama)
    srv, host = start()
    res3 = root + "/task3_local_inference/results"
    sh(["benchmark_ollama.py", "--models", "mock:fast", "mock:slow", "--host", host, "--out", res3 + "/benchmark.json"], T3)
    sh(["serving_plan.py", "--out", res3 + "/serving_plan.json"], T3)
    srv.shutdown()
    # ---- answers (dummy text of sufficient length)
    keys = ["T1_LORA_WHY", "T1_FAIL_1", "T1_FAIL_2", "T1_FAIL_3", "T1_MEMORY_NOTES", "T2_CHUNK_ANALYSIS", "T2_FAIL_1", "T2_FAIL_2",
            "T2_BM25", "T2_RERANKER", "T3_QUALITY_OBS", "T3_KV_CACHE", "T3_QUANT", "T3_SERVING_PLAN"]
    os.makedirs(root + "/report", exist_ok=True)
    body = "FIXTURE TEXT " + "this is dummy placeholder wording only used inside the automated test. " * 6
    open(root + "/report/my_answers.md", "w").write("<!-- c -->\n" + "\n".join(f"## {k}\n{body}\n" for k in keys))


def build(root, extra=None, out=None):
    args = [os.path.join(ROOT, "report/build_report.py"), "--root", root, "--out", out or root + "/r.pdf",
            "--name", "Test Student", "--github", "https://github.com/x/y", "--hardware", "Colab T4 (test fixture)"] + (extra or [])
    return subprocess.run([sys.executable] + args, capture_output=True, text=True)


def test_report_pipeline():
    try:
        import torch  # noqa
    except Exception:
        print("  (skipped: torch missing)"); return
    root = tempfile.mkdtemp()
    make_fixture(root)
    r = build(root)
    assert r.returncode == 0, r.stdout + r.stderr
    from pypdf import PdfReader
    pages = len(PdfReader(root + "/r.pdf").pages)
    assert 1 <= pages <= 4, pages
    text = "\n".join(p.extract_text() for p in PdfReader(root + "/r.pdf").pages)
    for needle in ("Task 1", "Task 2", "Task 3", "LoRA settings", "recall@5", "tok/s", "Test Student", "FIXTURE TEXT", "QLoRA"):
        assert needle in text, needle
    assert "DRAFT" not in text
    shutil.copy(root + "/r.pdf", os.environ.get("REPORT_FIXTURE_COPY", "/tmp/report_fixture.pdf"))

    # --- refusals -----------------------------------------------------------------------------
    ans = open(root + "/report/my_answers.md").read()
    open(root + "/report/my_answers.md", "w").write(ans.replace("## T2_BM25\nFIXTURE TEXT", "## T2_BM25\nTODO FIXTURE TEXT", 1))
    r = build(root); assert r.returncode == 1 and "T2_BM25 still contains TODO" in r.stdout
    open(root + "/report/my_answers.md", "w").write(ans.replace("## T3_QUANT\n" + ans.split("## T3_QUANT\n")[1].split("\n## ")[0], "## T3_QUANT\nshort answer\n"))
    r = build(root); assert r.returncode == 1 and "T3_QUANT is too short" in r.stdout
    open(root + "/report/my_answers.md", "w").write(ans)
    r = build(root, extra=[]); assert r.returncode == 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "report/build_report.py"), "--root", root, "--out", root + "/n.pdf"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "--name is required" in r.stdout                         # no name/github/hardware
    # smoke-flagged results must be rejected
    m = json.load(open(root + "/task1_lora/results/metrics_finetuned.json")); m["smoke_test"] = True
    json.dump(m, open(root + "/task1_lora/results/metrics_finetuned.json", "w"))
    r = build(root); assert r.returncode == 1 and "smoke test" in r.stdout
    r = build(root, extra=["--draft"], out=root + "/d.pdf"); assert r.returncode == 0
    assert "DRAFT" in "\n".join(p.extract_text() for p in PdfReader(root + "/d.pdf").pages)
    # missing answers file
    os.remove(root + "/report/my_answers.md")
    r = build(root); assert r.returncode == 1 and "my_answers.md" in r.stdout
    shutil.rmtree(root)


def test_template_is_rejected():
    root = tempfile.mkdtemp()
    os.makedirs(root + "/report")
    shutil.copy(os.path.join(ROOT, "report/my_answers.template.md"), root + "/report/my_answers.md")
    r = build(root)
    assert r.returncode == 1 and "still contains TODO" in r.stdout
    shutil.rmtree(root)


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
