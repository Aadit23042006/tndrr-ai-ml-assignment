"""Download the public PDFs listed in sources.txt into docs/ and check the 30+ page requirement.

    python fetch_docs.py
You can also drop ANY public PDF / .txt files into docs/ yourself and skip this script.
"""
import os
import sys
import urllib.request

from pypdf import PdfReader


def main():
    os.makedirs("docs", exist_ok=True)
    for line in open("sources.txt", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, url = [x.strip() for x in line.split("|", 1)]
        dest = os.path.join("docs", name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"have {name}")
            continue
        print(f"downloading {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (assignment-rag)"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    total = 0
    for fn in sorted(os.listdir("docs")):
        if fn.lower().endswith(".pdf"):
            n = len(PdfReader(os.path.join("docs", fn)).pages)
            total += n
            print(f"  {fn}: {n} pages")
    print(f"TOTAL pages: {total}")
    if total < 30:
        sys.exit("need at least 30 pages in total - add more PDFs to docs/")


if __name__ == "__main__":
    main()
