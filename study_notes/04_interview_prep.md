# Live walk-through ke liye taiyari

Har item ka jawab bina notes ke 1-2 minute mein de sako - aur apne repo ke code mein dikha sako.

## Task 1
- Dataset kaise bana? Train/test mein overlap kyun nahi hai? `shifted` split kyun?
- Loss sirf completion pe kyun (`labels = -100`)? `train_lora.py: build_examples` dikhao.
- LoRA formula likho. Trainable params kaise gine (r*(d_in+d_out))? `print_trainable_parameters` ka output kya tha?
- r aur alpha badalne se kya hota? (apna ablation result)
- Base model ne kyun fail kiya? Fine-tuned ne kahan fail kiya aur kyun?
- 7B ke liye 112 / 18 / 8 GB kaise aaye? Kaunsi cheez assumption hai?
- QLoRA mein kya 4-bit hai aur kya nahi (adapters, activations, optimizer state)?

## Task 2
- Chunk offsets aur `is_hit` kaise kaam karte hain? Chunk-size-independent metric kyun?
- recall@5 vs MRR ka farq. 15 questions pe kitna bharosa?
- FAISS kya karta hai; flat index vs approximate (IVF/HNSW) - yahan flat kyun theek hai?
- BM25 formula ke 3 hisse; RRF formula; cross-encoder vs bi-encoder.
- Tumhare 2 failure cases: kya retrieve hua, kyun, fix kya?

## Task 3
- tok/s kaise measure kiya? Warm-up kyun? Memory kahan se liya?
- Q4 aur FP16 ka quality farq - kaunse prompts pe dikha?
- KV cache size ka formula; Qwen2.5 ke liye number nikaal ke dikhao.
- 20 users plan: memory, decode, prefill - kaun sa bottleneck pehle aayega?

## Honesty
Agar kisi number pe sharmindagi ho ("yeh kyun aaya pata nahi") to us run ko dobara chalao aur samjho. Interview mein "maine yeh run kiya, yeh dekha, yeh guess hai" kehna bilkul theek hai.
