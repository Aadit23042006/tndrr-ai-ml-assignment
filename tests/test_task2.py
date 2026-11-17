"""Offline tests for Task 2 (RAG). Uses a synthetic 36-page PDF + a hashing embedder (no downloads).
Run:  python tests/test_task2.py"""
import json, math, os, random, subprocess, sys, tempfile, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(ROOT, "task2_rag")
sys.path.insert(0, T2)

import numpy as np
from rag_lib import (BM25, Doc, HashEmbedder, Retriever, VectorIndex, chunk_doc, first_hit_rank, is_hit,
                     load_corpus, locate_gold, metrics_from_ranks, normalize, rrf, bm25_tokens)


# ------------------------------------------------------------------ synthetic corpus
def make_word(rng):
    syl = ["zor", "vax", "qui", "llon", "mep", "tra", "nu", "bex", "dro", "fil", "gan", "hov", "jik", "lum", "wep"]
    return "".join(rng.choice(syl) for _ in range(3))


def make_corpus(folder, n_pages=36, seed=0):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    rng = random.Random(seed)
    facts = []                                   # (doc, question, gold sentence)
    os.makedirs(folder, exist_ok=True)
    for dname, pages in (("alpha.pdf", n_pages // 2), ("beta.pdf", n_pages - n_pages // 2)):
        c = canvas.Canvas(os.path.join(folder, dname), pagesize=A4)
        for _ in range(pages):
            y = 800
            for _ in range(4):                   # 4 paragraphs per page
                a, b, n = make_word(rng), make_word(rng), rng.randint(3, 99)
                sent = (f"The {a} protocol of the {b} district requires exactly {n} independent validators "
                        f"before any ledger entry becomes final and visible to every registered observer.")
                filler = " ".join(rng.choice(["system", "process", "record", "market", "signal", "channel", "layer"])
                                  for _ in range(60))
                facts.append((dname, f"How many validators does the {a} protocol of the {b} district require?", sent, n))
                for line in textwrap.wrap(sent + " " + filler, 95):
                    c.drawString(40, y, line); y -= 14
                y -= 10
            c.showPage()
        c.save()
    return facts


# ------------------------------------------------------------------ unit tests
def test_normalize():
    assert normalize("retrie-\nval  works\n\nwell­ly") == "retrieval works wellly".replace("wellly", "well" + "ly")
    assert normalize("  a \t b \n c ") == "a b c"


def test_chunking_offsets_and_coverage():
    text = " ".join(f"w{i}" for i in range(1000))
    d = Doc("t.txt", text, [0, 400, 3000])
    for size, ov in ((100, 20), (50, 0), (333, 100)):
        ch = chunk_doc(d, size, ov)
        for c in ch:
            assert d.text[c.start:c.end] == c.text                           # exact offsets
            assert len(c.text.split()) <= size
        assert ch[0].start == 0 and ch[-1].end == len(text)                  # full coverage
        covered = set()
        for c in ch:
            covered.update(range(c.start, c.end))
        assert all(i in covered for i, chr_ in enumerate(text) if chr_ != " ")
        assert len(ch) == math.ceil((1000 - size) / (size - ov)) + 1 or len(ch) == math.ceil(1000 / (size - ov))
        assert all(len(c.text.split()) == size for c in ch[:-1])
    assert d.page_of(0) == 1 and d.page_of(399) == 1 and d.page_of(400) == 2 and d.page_of(2999) == 2 and d.page_of(3000) == 3


def test_bm25_against_manual_formula():
    docs = ["apple banana apple", "banana cherry", "cherry cherry date"]
    bm = BM25(docs, k1=1.5, b=0.75)
    toks = [bm25_tokens(x) for x in docs]
    avgdl = sum(map(len, toks)) / 3
    def manual(q, i):
        s = 0.0
        for t in set(bm25_tokens(q)):
            df = sum(t in tk for tk in toks)
            idf = math.log(1 + (3 - df + 0.5) / (df + 0.5))
            tf = toks[i].count(t)
            s += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * len(toks[i]) / avgdl))
        return s
    for q in ("apple", "banana cherry", "date apple", "zzz"):
        sc = bm.scores(q)
        for i in range(3):
            assert abs(sc[i] - manual(q, i)) < 1e-5, (q, i, sc[i], manual(q, i))
    assert bm.search("apple", 1)[1][0] == 0


def test_rrf():
    # 2 is rank 1 in one list and rank 2 in the other -> 1/62+1/61 beats every other doc
    assert rrf([[1, 2, 3], [2, 3, 1]], k=60)[:1] == [2]
    # convexity: ranks (1,3) and (3,1) tie and both beat (2,2)
    out = rrf([[1, 2, 3], [3, 2, 1]], k=60)
    assert set(out[:2]) == {1, 3} and out[2] == 2
    out = rrf([[5, 6], [6, 7]], k=0)
    assert out[0] == 6 and set(out) == {5, 6, 7}


def test_vector_index_faiss_numpy_parity():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 32)).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:5] + 0.01
    a = VectorIndex(v, use_faiss=True).search(q, 7)[1]
    b = VectorIndex(v, use_faiss=False).search(q, 7)[1]
    assert (a == b).all() and (a[:, 0] == np.arange(5)).all()


def test_locate_and_hit():
    d1 = Doc("a.txt", normalize("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"), [0])
    d2 = Doc("b.txt", normalize("gamma delta"), [0])
    g = locate_gold([d1, d2], "GAMMA   delta\nepsilon", None)
    assert g[0] == "a.txt" and d1.text[g[1]:g[2]].lower() == "gamma delta epsilon"
    assert locate_gold([d1, d2], "gamma delta", "b.txt")[0] == "b.txt"
    try:
        locate_gold([d1], "not present anywhere"); assert False
    except ValueError:
        pass
    ch = chunk_doc(d1, 4, 0)                                    # [Alpha..delta][epsilon..iota? ...]
    hits = [is_hit(c, g) for c in ch]
    assert any(hits) and not all(hits)
    assert not is_hit(chunk_doc(d2, 2, 0)[0], g)                # same words, wrong document
    assert first_hit_rank(ch, g) == hits.index(True) + 1
    m = metrics_from_ranks([1, 3, None, 6], ks=(1, 3, 5))
    assert m["recall@1"] == .25 and m["recall@3"] == .5 and m["recall@5"] == .5
    assert abs(m["mrr"] - (1 + 1/3 + 0 + 1/6) / 4) < 1e-9


def test_retrieval_end_to_end_two_chunk_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        facts = make_corpus(tmp + "/docs", n_pages=36)
        docs = load_corpus(tmp + "/docs")
        assert sum(d.n_pages for d in docs) == 36
        rng = random.Random(1)
        sample = rng.sample(facts, 15)
        emb = HashEmbedder()
        for size in (60, 200):
            ret = Retriever(docs, size, size // 5, emb)
            ranks = []
            for dname, q, sent, n in sample:
                g = locate_gold(docs, sent, dname)
                ids = ret.retrieve(q, k=5, method="hybrid")
                ranks.append(first_hit_rank([ret.chunks[i] for i in ids], g))
            m = metrics_from_ranks(ranks)
            # BM25 is exact on these rare invented words; the toy hash embedder is weak on long chunks,
            # so only BM25 (and hybrid for the small chunk size) are asserted
            bm = []
            for dname, q, sent, n in sample:
                g = locate_gold(docs, sent, dname)
                bm.append(first_hit_rank([ret.chunks[i] for i in ret.retrieve(q, k=5, method="bm25")], g))
            assert metrics_from_ranks(bm)["recall@5"] >= 0.9, (size, bm)
            if size == 60:
                assert m["recall@5"] >= 0.9, (size, m, ranks)


def run_cli(args, cwd=T2):
    return subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)


def test_cli_evaluate_and_error_paths():
    with tempfile.TemporaryDirectory() as tmp:
        facts = make_corpus(tmp + "/docs", n_pages=36)
        docs = load_corpus(tmp + "/docs")
        rng = random.Random(2)
        qs = [{"id": i + 1, "question": q, "doc": dn, "gold_passage": s, "answer": str(n)}
              for i, (dn, q, s, n) in enumerate(rng.sample(facts, 15))]
        qfile = tmp + "/q.json"
        json.dump(qs, open(qfile, "w"))
        base = ["evaluate_retrieval.py", "--docs", tmp + "/docs", "--questions", qfile, "--embedder", "hash",
                "--chunk-sizes", "60", "200", "--out-dir", tmp + "/res", "--smoke"]
        r = run_cli(base)
        assert r.returncode == 0, r.stdout + r.stderr
        res = json.load(open(tmp + "/res/retrieval_metrics.json"))
        assert res["smoke_test"] is True and res["n_pages"] == 36
        assert set(res["chunk_sizes"]) == {"60", "200"}
        for e in res["chunk_sizes"].values():
            assert set(e["methods"]) == {"dense", "bm25", "hybrid"}
            assert e["methods"]["bm25"]["recall@5"] >= 0.9 and e["methods"]["bm25"]["n_questions"] == 15
            assert 0 <= e["methods"]["hybrid"]["recall@5"] <= 1 and len(e["methods"]["dense"]["per_question_rank"]) == 15
        assert os.path.exists(tmp + "/res/retrieval_failures.md")

        # without --smoke and with fewer than 15 questions -> must refuse
        json.dump(qs[:5], open(tmp + "/q5.json", "w"))
        r = run_cli(["evaluate_retrieval.py", "--docs", tmp + "/docs", "--questions", tmp + "/q5.json",
                     "--embedder", "hash", "--out-dir", tmp + "/res2", "--min-pages", "10"])
        assert r.returncode != 0 and "needs 15" in (r.stdout + r.stderr)
        # gold passage that is not in the corpus -> clear error
        bad = [dict(q) for q in qs]; bad[3]["gold_passage"] = "this text is nowhere in any document at all"
        json.dump(bad, open(tmp + "/qbad.json", "w"))
        r = run_cli(["evaluate_retrieval.py", "--docs", tmp + "/docs", "--questions", tmp + "/qbad.json",
                     "--embedder", "hash", "--out-dir", tmp + "/res3", "--min-pages", "10"])
        assert r.returncode != 0 and "q4" in (r.stdout + r.stderr) and "not found" in (r.stdout + r.stderr)
        # the TODO template must be rejected
        r = run_cli(["evaluate_retrieval.py", "--docs", tmp + "/docs", "--questions", "questions_template.json",
                     "--embedder", "hash", "--out-dir", tmp + "/res4", "--smoke"])
        assert r.returncode != 0 and "TODO" in (r.stdout + r.stderr)
        # corpus under 30 pages must be refused (without --smoke)
        small = tmp + "/small"; make_corpus(small, n_pages=10)
        r = run_cli(["evaluate_retrieval.py", "--docs", small, "--questions", qfile, "--embedder", "hash",
                     "--out-dir", tmp + "/res5"])
        assert r.returncode != 0 and "needs >= 30" in (r.stdout + r.stderr)
        # find_passage finds a phrase
        r = run_cli(["find_passage.py", "independent validators", "--docs", tmp + "/docs", "--max", "1"])
        assert r.returncode == 0 and "page" in r.stdout


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
