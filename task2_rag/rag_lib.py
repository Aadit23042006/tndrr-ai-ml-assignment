"""RAG from scratch: ingestion, chunking, embeddings, vector search, BM25, fusion, evaluation.

No LangChain / LlamaIndex. Only: pypdf (PDF -> text), numpy, faiss (optional, numpy fallback),
and an embedding model (sentence-transformers) for the dense side.

Key design choice - evaluation is *chunk-size independent*:
  every document is one long normalised string; a chunk is a character span [start, end) of it;
  the gold passage you write is located as a character span too. A retrieved chunk counts as a
  hit when it comes from the right document and overlaps the gold span enough
  (see `is_hit`). That lets you compare different chunk sizes fairly.
"""
import hashlib
import math
import os
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


# ============================================================================ ingestion
def normalize(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)    # "retrie-\nval" -> "retrieval"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class Doc:
    name: str
    text: str            # normalised full text
    page_starts: list    # char offset where each page begins (len = n_pages)

    @property
    def n_pages(self):
        return len(self.page_starts)

    def page_of(self, offset: int) -> int:
        """1-based page number containing a character offset."""
        page = 1
        for i, s in enumerate(self.page_starts):
            if s <= offset:
                page = i + 1
            else:
                break
        return page


def load_pdf(path: str) -> Doc:
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts, starts, pos = [], [], 0
    for page in reader.pages:
        t = normalize(page.extract_text() or "")
        starts.append(pos)
        parts.append(t)
        pos += len(t) + 1                               # +1 for the joining space
    return Doc(os.path.basename(path), " ".join(parts), starts)


def load_text(path: str) -> Doc:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return Doc(os.path.basename(path), normalize(f.read()), [0])


def load_corpus(folder: str) -> list:
    docs = []
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if fn.lower().endswith(".pdf"):
            docs.append(load_pdf(p))
        elif fn.lower().endswith((".txt", ".md")):
            docs.append(load_text(p))
    if not docs:
        raise FileNotFoundError(f"no .pdf/.txt/.md files in {folder}")
    return docs


# ============================================================================ chunking
@dataclass
class Chunk:
    doc: str
    start: int
    end: int
    text: str
    page: int


def chunk_doc(doc: Doc, size_words: int, overlap_words: int) -> list:
    """Sliding window over words. Offsets are exact character positions in doc.text."""
    assert 0 <= overlap_words < size_words
    spans = [m.span() for m in re.finditer(r"\S+", doc.text)]
    step = size_words - overlap_words
    chunks, i = [], 0
    while i < len(spans):
        j = min(i + size_words, len(spans))
        s, e = spans[i][0], spans[j - 1][1]
        chunks.append(Chunk(doc.name, s, e, doc.text[s:e], doc.page_of(s)))
        if j == len(spans):
            break
        i += step
    return chunks


def chunk_corpus(docs, size_words, overlap_words):
    out = []
    for d in docs:
        out.extend(chunk_doc(d, size_words, overlap_words))
    return out


# ============================================================================ embedders
class HashEmbedder:
    """Deterministic bag-of-hashed-(uni+bi)grams embedder. For OFFLINE TESTING of the pipeline only."""
    name = "hash"

    def __init__(self, dim=1024):
        self.dim = dim

    def _vec(self, text):
        toks = re.findall(r"\w+", text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size=64):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts):
        return self.encode_docs(texts)


class STEmbedder:
    """sentence-transformers model. bge models want an instruction prefix on the QUERY side only."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = ("Represent this sentence for searching relevant passages: "
                             if "bge" in model_name.lower() and "en" in model_name.lower() else "")
        self.max_tokens = getattr(self.model, "max_seq_length", None)

    def encode_docs(self, texts, batch_size=32):
        return np.asarray(self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                            show_progress_bar=len(texts) > 200), dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray(self.model.encode([self.query_prefix + t for t in texts],
                                            normalize_embeddings=True), dtype=np.float32)


def make_embedder(spec: str):
    if spec == "hash":
        return HashEmbedder()
    if spec.startswith("st:"):
        return STEmbedder(spec[3:])
    raise ValueError("embedder must be 'hash' or 'st:<sentence-transformers model name>'")


# ============================================================================ vector index
class VectorIndex:
    """Inner-product search over L2-normalised vectors (= cosine). FAISS if available, else NumPy."""

    def __init__(self, vectors: np.ndarray, use_faiss=True):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.faiss = None
        if use_faiss:
            try:
                import faiss
                self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self.faiss.add(self.vectors)
            except ImportError:
                pass

    def search(self, q: np.ndarray, k: int):
        """q: (nq, d) -> (scores, ids) each (nq, k)."""
        k = min(k, len(self.vectors))
        if self.faiss is not None:
            return self.faiss.search(np.ascontiguousarray(q, dtype=np.float32), k)
        sims = q @ self.vectors.T
        ids = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, ids, axis=1), ids


# ============================================================================ BM25 (from scratch)
_STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its".split())


def bm25_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25. score(q,d) = sum_t idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)            # term -> [(doc_id, tf)]
        self.doc_len = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = bm25_tokens(t)
            self.doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(texts)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(bm25_tokens(query)):
            for i, tf in self.postings.get(term, ()):
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += self.idf[term] * tf * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int):
        s = self.scores(query)
        ids = np.argsort(-s)[:k]
        return s[ids], ids


# ============================================================================ fusion + rerank
def rrf(rank_lists, k=60, top=None):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1/(k + rank). Ranks start at 1."""
    score = defaultdict(float)
    for lst in rank_lists:
        for r, d in enumerate(lst, start=1):
            score[int(d)] += 1.0 / (k + r)
    ranked = sorted(score, key=lambda d: -score[d])
    return ranked[:top] if top else ranked


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunk_texts, ids):
        pairs = [(query, chunk_texts[i]) for i in ids]
        sc = self.model.predict(pairs)
        return [ids[i] for i in np.argsort(-np.asarray(sc))]


# ============================================================================ retriever
class Retriever:
    """Holds chunks + dense index + BM25 and answers queries by method: dense | bm25 | hybrid."""

    def __init__(self, docs, chunk_words, overlap_words, embedder, reranker=None, use_faiss=True):
        self.chunks = chunk_corpus(docs, chunk_words, overlap_words)
        self.texts = [c.text for c in self.chunks]
        self.embedder = embedder
        self.dense = VectorIndex(embedder.encode_docs(self.texts), use_faiss=use_faiss)
        self.bm25 = BM25(self.texts)
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 5, method: str = "dense", pool: int = 50, rerank_pool: int = 20):
        qv = self.embedder.encode_queries([query])
        if method == "dense":
            ids = list(self.dense.search(qv, pool)[1][0])
        elif method == "bm25":
            ids = list(self.bm25.search(query, pool)[1])
        elif method == "hybrid":
            d = list(self.dense.search(qv, pool)[1][0])
            b = list(self.bm25.search(query, pool)[1])
            ids = rrf([d, b])
        else:
            raise ValueError(method)
        if self.reranker is not None:
            ids = self.reranker.rerank(query, self.texts, [int(i) for i in ids[:rerank_pool]]) \
                  + [int(i) for i in ids[rerank_pool:]]
        return [int(i) for i in ids[:k]]


# ============================================================================ gold + metrics
def locate_gold(docs, gold_passage: str, doc_name=None):
    """Find the gold passage as a char span. Whitespace/case-insensitive. Returns (doc_name, start, end)."""
    words = gold_passage.split()
    if not words:
        raise ValueError("empty gold passage")
    pat = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
    hits = []
    for d in docs:
        if doc_name and d.name != doc_name:
            continue
        for m in pat.finditer(d.text):
            hits.append((d.name, m.start(), m.end()))
    if not hits:
        raise ValueError("gold passage not found in corpus text")
    return hits[0]


def is_hit(chunk: Chunk, gold, min_frac=0.5) -> bool:
    """Chunk is relevant if same doc and overlap >= min_frac * min(len(gold), len(chunk))."""
    g_doc, g_s, g_e = gold
    if chunk.doc != g_doc:
        return False
    overlap = min(chunk.end, g_e) - max(chunk.start, g_s)
    return overlap >= min_frac * min(g_e - g_s, chunk.end - chunk.start)


def first_hit_rank(retrieved_chunks, gold):
    """1-based rank of the first relevant chunk, or None."""
    for r, c in enumerate(retrieved_chunks, start=1):
        if is_hit(c, gold):
            return r
    return None


def metrics_from_ranks(ranks, ks=(1, 3, 5)):
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    out["n_questions"] = n
    return out


def corpus_fingerprint(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update(d.name.encode())
        h.update(d.text.encode())
    return h.hexdigest()[:12]
