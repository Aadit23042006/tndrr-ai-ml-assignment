"""Search the corpus for a phrase so you can copy the EXACT gold passage for a question.

    python find_passage.py "low-rank adaptation" --docs docs --context 300

Prints each match with doc name, page and surrounding text (taken from the normalised text
that the evaluator uses, so copying from here is guaranteed to match).
"""
import argparse
import re

from rag_lib import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phrase")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--context", type=int, default=250, help="characters shown on each side")
    ap.add_argument("--max", type=int, default=8)
    a = ap.parse_args()

    pat = re.compile(r"\s+".join(re.escape(w) for w in a.phrase.split()), re.IGNORECASE)
    shown = 0
    for d in load_corpus(a.docs):
        for m in pat.finditer(d.text):
            s, e = max(0, m.start() - a.context), min(len(d.text), m.end() + a.context)
            print(f"--- {d.name}  page {d.page_of(m.start())}  chars {m.start()}-{m.end()}")
            print(d.text[s:e], "\n")
            shown += 1
            if shown >= a.max:
                return
    if not shown:
        print("no match (PDF text extraction can split words; try a shorter phrase)")


if __name__ == "__main__":
    main()
