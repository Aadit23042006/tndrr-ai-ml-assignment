# tndrr AI/ML Internship - Assignment (Tasks 1-3)

Three folders, one per task, plus the report builder:

```
task1_lora/             LoRA fine-tuning: English calendar request -> JSON  (make_dataset, train_lora, evaluate, failures, memory_calc)
task2_rag/              RAG from scratch: pypdf + embeddings + FAISS/NumPy + BM25 (rag_lib, evaluate_retrieval, ask)
task3_local_inference/  Ollama 16-bit vs 4-bit benchmark + KV-cache and 20-user serving calculators
report/                 build_report.py -> 4-page PDF from your results + your own written answers
study_notes/            concept notes in Hinglish (read, then write the report in your own words)
tests/                  offline tests of the code logic (run_tests.sh)
run_on_colab.ipynb      the same steps as a Colab notebook
```

## Important: what is and is not in this repo
The **code is tested offline** (`bash run_tests.sh`: dataset/scoring, label masking, a training + evaluation run on a tiny random model,
RAG chunking/BM25/FAISS/metrics on a synthetic PDF corpus, the Ollama benchmark against a mock server, and the PDF report builder).
There are **no result numbers in this repo**: the assignment requires every number to come from your own runs, so you must
run the three tasks on a real GPU / real Ollama. The report builder refuses to make a final PDF from smoke-test results, from
missing results, or from unanswered (`TODO`) questions.

## Steps (free Colab T4 works; total roughly 2-3 hours including reading outputs)

### Task 1 - ~15 min of compute
```bash
cd task1_lora && pip install -r requirements.txt
bash run_all.sh                 # data -> base baselines -> LoRA training -> fine-tuned eval -> failures -> 7B memory calc
bash ablation.sh                # optional (r=4, r=64, lr=5e-5): evidence for "why these settings"
python memory_calc.py --name Qwen2.5-0.5B --params 0.494e9 --layers 24 --hidden 896 --inter 4864 --heads 14 --kv-heads 2 --act-gb 1 --out results/memory_calc_0p5b.json
```
Then read `results/failures_finetuned.md`, pick 3 failures, check *why* against the training data.
`outputs/lora_r16/train_summary.json` has the LoRA settings, trainable params, training time and the **measured peak GPU memory**.

### Task 2 - ~10 min of compute, plus the time to write 15 questions
```bash
cd task2_rag && pip install -r requirements.txt
python fetch_docs.py                                   # 4 public arXiv papers (or put your own PDFs in docs/)
python find_passage.py "low-rank" --docs docs          # find exact gold passages
cp questions_template.json questions.json              # write 15 questions YOURSELF (see study_notes/02)
python evaluate_retrieval.py --chunk-sizes 100 250 --rerank   # recall@1/3/5 + MRR: dense, bm25, hybrid (+ cross-encoder)
python ask.py --questions questions.json --llm ollama:qwen2.5:1.5b-instruct --limit 5   # optional end-to-end answers
```
`results/retrieval_failures.md` lists the questions where retrieval missed - pick 2 and explain.

### Task 3 - ~20 min
```bash
# install + start Ollama (Colab: apt-get install -y zstd first), then pull two settings of the same model
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q4_K_M            # verify the exact tags with `ollama list` / the Ollama library page
cd task3_local_inference
python benchmark_ollama.py --models qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M
python compare_quality.py                           # read results/quality_comparison.md and write your observations
python kv_cache_calc.py --ctx 4096 --users 20
python serving_plan.py --users 20 --ctx 4096        # edit assumptions (GPU bandwidth etc.) to match your plan
```

### Report
```bash
cp report/my_answers.template.md report/my_answers.md      # fill every TODO IN YOUR OWN WORDS
python report/build_report.py --name "Your Name" --github https://github.com/<you>/<repo> --hardware "Colab T4 16GB"
```
-> `report/report.pdf` (max 4 pages, enforced). Use `--draft` to preview while you are still writing.

### Submit
Push this folder to a GitHub repo (public or shared), keep `results/` and `train_summary.json`, then submit repo link + `report.pdf`
at the form before **Thu 19 Nov 2026, 11:59 PM IST**.

## Design notes (short)
* Task 1 uses a plain PyTorch loop (visible training step) with PEFT `LoraConfig`; `--qlora` switches the base model to 4-bit NF4 (needs CUDA + bitsandbytes).
* Task 2 evaluates retrieval by **character-span overlap with your gold passage**, so different chunk sizes are compared fairly.
* Task 3 reads Ollama's own timing fields (`eval_count`/`eval_duration`) and its `/api/ps` memory report (plus `nvidia-smi` if present).
