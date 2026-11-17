"""Question answering over the corpus: retrieve top-k chunks, then let ANY LLM write the answer.

    # just show what would be sent to the LLM
    python ask.py --question "What does LoRA freeze?"
    # answer with a local Ollama model (same Ollama you use in Task 3)
    python ask.py --question "What does LoRA freeze?" --llm ollama:qwen2.5:1.5b-instruct

    # run all questions from questions.json through the pipeline -> results/qa_samples.md
    python ask.py --questions questions.json --llm ollama:qwen2.5:1.5b-instruct --limit 5
"""
import argparse
import json
import os
import urllib.request

from rag_lib import Retriever, load_corpus, make_embedder

PROMPT = """Answer the question using ONLY the context below. If the context does not contain the answer, say "I don't know".
Cite the sources like [file p.N].

Context:
{context}

Question: {question}
Answer:"""


def call_ollama(model, prompt, host="http://localhost:11434"):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def answer(ret, question, k, method, llm):
    ids = ret.retrieve(question, k=k, method=method)
    ctx = "\n\n".join(f"[{ret.chunks[i].doc} p.{ret.chunks[i].page}] {ret.chunks[i].text}" for i in ids)
    prompt = PROMPT.format(context=ctx, question=question)
    if not llm:
        return prompt, None
    if llm.startswith("ollama:"):
        return prompt, call_ollama(llm[len("ollama:"):], prompt)
    raise ValueError("--llm must look like ollama:<model-tag>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--question")
    ap.add_argument("--questions", help="questions.json to batch over")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=250, help="chunk size in words")
    ap.add_argument("--overlap-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--out", default="results/qa_samples.md")
    a = ap.parse_args()

    docs = load_corpus(a.docs)
    ret = Retriever(docs, a.chunk, int(a.chunk * a.overlap_frac), make_embedder(a.embedder))

    if a.question:
        prompt, ans = answer(ret, a.question, a.k, a.method, a.llm)
        print(prompt if ans is None else f"{ans}")
    elif a.questions:
        qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]
        lines = ["# Sample end-to-end answers\n"]
        for q in qs:
            _, ans = answer(ret, q["question"], a.k, a.method, a.llm)
            lines += [f"**Q{q.get('id')}:** {q['question']}", f"- reference: {q.get('answer', '')}",
                      f"- system: {ans}", ""]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write("\n".join(lines))
        print(f"saved {a.out}")
    else:
        ap.error("give --question or --questions")


if __name__ == "__main__":
    main()
