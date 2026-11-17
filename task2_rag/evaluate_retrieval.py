"""Measure retrieval quality (recall@k, MRR) for several chunk sizes and retrieval methods.

    python evaluate_retrieval.py --docs docs --questions questions.json \
        --chunk-sizes 100 250 --embedder st:BAAI/bge-small-en-v1.5

Methods: dense (embeddings), bm25 (keywords), hybrid (RRF fusion of the two),
and with --rerank each of them again followed by a cross-encoder re-ranker.

Writes results/retrieval_metrics.json and results/retrieval_failures.md
"""
import argparse
import json
import os
import sys

from rag_lib import (CrossEncoderReranker, Retriever, corpus_fingerprint, first_hit_rank, load_corpus,
                     locate_gold, make_embedder, metrics_from_ranks)


def load_questions(path, docs, min_questions):
    qs = json.load(open(path, encoding="utf-8"))
    problems, golds = [], []
    for q in qs:
        qid = q.get("id", "?")
        if any("TODO" in str(q.get(k, "")) for k in ("question", "gold_passage")):
            problems.append(f"q{qid}: still contains TODO")
            continue
        if not q.get("question") or not q.get("gold_passage"):
            problems.append(f"q{qid}: needs 'question' and 'gold_passage'")
            continue
        try:
            golds.append(locate_gold(docs, q["gold_passage"], q.get("doc")))
        except ValueError as e:
            problems.append(f"q{qid}: {e} (use find_passage.py to copy exact text)")
    if len(qs) < min_questions:
        problems.append(f"only {len(qs)} questions, the assignment needs {min_questions}")
    if problems:
        sys.exit("Problems in questions file:\n  " + "\n  ".join(problems))
    return qs, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[100, 250], help="chunk size in WORDS")
    ap.add_argument("--overlap-frac", type=float, default=0.2)
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("--rerank", action="store_true", help="also evaluate <method>+rerank (cross-encoder)")
    ap.add_argument("--failure-method", default="dense")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--min-questions", type=int, default=15)
    ap.add_argument("--min-pages", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="flag results as a pipeline test")
    a = ap.parse_args()

    docs = load_corpus(a.docs)
    n_pages = sum(d.n_pages for d in docs)
    print(f"{len(docs)} documents, {n_pages} pages, {sum(len(d.text.split()) for d in docs)} words")
    if n_pages < a.min_pages and not a.smoke:
        sys.exit(f"corpus has {n_pages} pages; the assignment needs >= {a.min_pages}")

    questions, golds = load_questions(a.questions, docs, a.min_questions if not a.smoke else 1)
    embedder = make_embedder(a.embedder)
    reranker = CrossEncoderReranker() if a.rerank else None

    result = {"smoke_test": bool(a.smoke), "embedder": a.embedder, "n_docs": len(docs), "n_pages": n_pages,
              "corpus_fingerprint": corpus_fingerprint(docs), "overlap_frac": a.overlap_frac,
              "ks": a.ks, "chunk_sizes": {}}
    failures = ["# Retrieval failures\n", f"Method: `{a.failure_method}`; a failure = gold passage not in top-{max(a.ks)}.\n"]
    kmax = max(a.ks)

    for size in a.chunk_sizes:
        overlap = int(size * a.overlap_frac)
        ret = Retriever(docs, size, overlap, embedder, reranker=None, use_faiss=not a.no_faiss)
        max_tok = getattr(embedder, "max_tokens", None)
        if max_tok and size * 1.4 > max_tok:
            print(f"  WARNING: {size}-word chunks (~{int(size*1.4)} tokens) may be truncated by the embedder "
                  f"(max {max_tok} tokens) - dense results for this size are then not comparable.")
        entry = {"overlap_words": overlap, "n_chunks": len(ret.chunks), "methods": {}}
        methods = list(a.methods) + ([m + "+rerank" for m in a.methods] if a.rerank else [])
        for m in methods:
            base = m.replace("+rerank", "")
            ret.reranker = reranker if m.endswith("+rerank") else None
            ranks, per_q = [], []
            for q, g in zip(questions, golds):
                ids = ret.retrieve(q["question"], k=max(kmax, 10), method=base)
                chunks = [ret.chunks[i] for i in ids]
                rank = first_hit_rank(chunks, g)
                ranks.append(rank)
                per_q.append({"id": q.get("id"), "rank": rank})
                if m == a.failure_method and (rank is None or rank > kmax):
                    failures.append(f"## chunk={size}w  q{q.get('id')}: {q['question']}\n")
                    failures.append(f"- gold ({g[0]}, p.{docs[[d.name for d in docs].index(g[0])].page_of(g[1])}): "
                                    f"{q['gold_passage'][:300]}")
                    failures.append(f"- first relevant rank within top-10: {rank}")
                    for r, c in enumerate(chunks[:3], 1):
                        failures.append(f"  - top{r}: {c.doc} p.{c.page}: {c.text[:220]}...")
                    failures.append("")
            entry["methods"][m] = {**metrics_from_ranks(ranks, a.ks), "per_question_rank": per_q}
        result["chunk_sizes"][str(size)] = entry

    print(f"\n{'chunk':>6} {'method':<16} " + " ".join(f"R@{k:<3}" for k in a.ks) + "  MRR   (#chunks)")
    for size, e in result["chunk_sizes"].items():
        for m, r in e["methods"].items():
            print(f"{size:>6} {m:<16} " + " ".join(f"{r[f'recall@{k}']:.2f} " for k in a.ks)
                  + f" {r['mrr']:.3f}  ({e['n_chunks']})")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "retrieval_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.out_dir, "retrieval_failures.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(failures))
    print(f"\nsaved {a.out_dir}/retrieval_metrics.json and retrieval_failures.md")


if __name__ == "__main__":
    main()
