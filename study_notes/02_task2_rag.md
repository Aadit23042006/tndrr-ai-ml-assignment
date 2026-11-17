# Task 2 - RAG from scratch

## Pipeline
1. **Ingest:** PDF -> text (pypdf), whitespace/hyphen normalise.
2. **Chunk:** words ki sliding window (size, overlap). Har chunk ke character offsets save hote hain (taaki gold passage se overlap nikal sako).
3. **Embed:** sentence-transformers model -> vectors (normalised). FAISS `IndexFlatIP` = inner product = cosine similarity.
4. **Retrieve:** query ko embed karo (bge models mein query pe instruction prefix lagta hai), top-k chunks.
5. **Generate:** top-k chunks + question ko LLM ko do ("sirf context se jawab do, source cite karo").

## Chunk size trade-off (apne numbers se jawab do)
- **Chhota chunk:** embedding focused hoti hai (precise), lekin context kat sakta hai; answer do chunks mein bant sakta hai; chunks zyada => search noise zyada.
- **Bada chunk:** poora context milta hai, lekin ek vector mein kai topics mix => embedding dilute; embedding model ki max length (bge-small: 512 tokens)
  cross ho to **truncation** hota hai (text silently kat jata hai!). 1 word ~ 1.3-1.4 tokens.
- Kaun sa better nikla yeh tumhare numbers batayenge. Sirf wahi explanation likho jo tumhare failures/numbers support karte hon.

## Metrics
- **recall@k** = kitne questions mein sahi chunk top-k mein aaya (hit rate). 15 questions => ek question = 6.7 points, isliye chhote differences ko "noise" bhi bol sakte ho.
- **MRR** = average of 1/rank of first correct chunk.
- Hit ki definition (repo mein): same document + gold span se overlap >= 50% of min(gold, chunk). Yeh chunk size se independent hai.

## BM25 kya add karta hai
- Dense embeddings *meaning* pakadte hain (synonyms, paraphrase) lekin rare exact tokens (naam, error codes, numbers, acronyms) kabhi miss kar dete hain.
- BM25 = keyword scoring: `idf(term)` (rare word zyada important) x saturating term frequency (tf badhne pe diminishing returns) x document length normalisation.
- Hybrid = dono ki rankings ko **RRF** (reciprocal rank fusion: score = sum 1/(60 + rank)) se jodna. Dhyan: agar ek retriever bahut kharab ho to RRF accuracy gira bhi sakta hai - apne results dekho.

## Re-ranker kya karta hai
- Bi-encoder (embedding model) query aur chunk ko **alag-alag** encode karta hai => fast, pre-compute ho sakta hai, lekin kam precise.
- Cross-encoder query+chunk ko **saath** model mein deta hai (full attention across both) => zyada accurate, lekin har (query, chunk) pair ke liye ek forward pass => sirf top ~20-50 candidates pe chalate hain.
- Pipeline: fast first-stage (dense/BM25/hybrid, top-50) -> cross-encoder re-rank -> top-5 LLM ko.

## 15 questions kaise likho (yeh tumhe khud karna hai)
- Documents ke kuch sections padho, aise sawal likho jinka jawab ek hi jagah ho. Mix rakho: definition, number/fact, "kyun" wale, aur 3-4 aise jo query mein exact keywords na use karte hon (paraphrase) - tabhi dense vs BM25 ka farq dikhega.
- `python find_passage.py "phrase"` se exact gold passage copy karo (25-80 words).
- Rule: question mein gold passage ke sentences copy mat karo, warna BM25 ka result artificially achha dikhega.
