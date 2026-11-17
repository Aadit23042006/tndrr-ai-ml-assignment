<!--
HOW TO USE: copy this file to my_answers.md and replace every TODO with YOUR OWN WORDS.
Lines inside <!-- --> comments are ignored. The headings (## KEY) must stay exactly as they are.
Use only what you really measured/understood: you will be asked to explain it live.
You can use **bold** and `code`. Keep the whole report within 4 pages - be concise.
Use study_notes/ to understand the concepts, then close it and write from your own head.
-->

## T1_LORA_WHY
<!-- Why this rank, alpha, target modules, learning rate, epochs? What did you observe if you ran ablation.sh (r=4 / r=64 / lr 5e-5)? (50-120 words) -->
TODO

## T1_FAIL_1
<!-- Failure 1: paste the request, the expected JSON and the model's output (from results/failures_finetuned.md), then WHY you think it fails. Check your guess against the training data (e.g. grep the train file for that format). -->
TODO

## T1_FAIL_2
TODO

## T1_FAIL_3
TODO

## T1_MEMORY_NOTES
<!-- 2-4 sentences: which assumptions matter most (activations, batch size, optimizer, gradient checkpointing)? How does the formula compare with the peak GPU memory you measured for the 0.5B model? (run memory_calc.py with the 0.5B parameters to compare) -->
TODO

## T2_CHUNK_ANALYSIS
<!-- Which chunk size worked better in YOUR table and why (precision of the embedding vs. context, truncation, number of chunks, what the questions looked like)? Only claim what your numbers support. -->
TODO

## T2_FAIL_1
<!-- Failure case 1 from results/retrieval_failures.md: the question, what was retrieved instead, why, and what would fix it (hybrid, re-ranker, smaller/larger chunks, query rewriting, ...). -->
TODO

## T2_FAIL_2
TODO

## T2_BM25
<!-- In your own words: what does BM25 add to embedding search? When does keyword matching win? Refer to your hybrid/bm25 numbers if they show it. -->
TODO

## T2_RERANKER
<!-- In your own words: what does a re-ranker (cross-encoder) do, how is it different from the embedding model, and why is it only run on the top ~20 candidates? -->
TODO

## T3_QUALITY_OBS
<!-- Your observations after reading both outputs for all 10 prompts (results/quality_comparison.md): where did the quantised model differ, which prompts got worse/better, any formatting/reasoning/fact errors? -->
TODO

## T3_KV_CACHE
<!-- In your own words: what is the KV cache, what problem does it solve, and how does its size grow (use the formula from kv_cache_calc.py with your model's numbers)? -->
TODO

## T3_QUANT
<!-- In your own words: why does quantisation save memory (and what does it cost)? Relate it to the memory numbers you measured for 16-bit vs 4-bit. -->
TODO

## T3_SERVING_PLAN
<!-- Your plan for 20 concurrent users on one 24 GB GPU: model choice, quantisation, serving engine (e.g. vLLM with continuous batching + paged attention), context limit, what you would measure first, and what you'd do if latency is too high. Refer to the table above. -->
TODO
