## T1_LORA_WHY

I used LoRA to fine-tune Qwen2.5-0.5B without updating all base-model parameters. The configuration used rank 16, alpha 32, dropout 0.05, and attention and MLP projection layers as LoRA targets. Training used BF16 on a Colab T4 GPU, learning rate 0.0002, 3 epochs, batch size 16, and maximum sequence length 256.

The model had 8,798,208 trainable parameters out of 502,830,976 (about 1.75%). Training took 612.67 seconds, with measured peak GPU memory of 8.43 GB.

On 100 held-out examples, the fine-tuned model achieved 100% valid JSON, 100% schema validity, and 93% exact match. Exact match was 100% on the 70 in-distribution examples and 76.67% on the 30 shifted examples.

## T1_FAIL_1

Example test-076: A code review request specified an hour and a quarter (75 minutes).

- Expected duration: 75 minutes.
- Predicted duration: 120 minutes.
- Training-data check: 0 examples with a 75-minute target and 189 examples with a 120-minute target.

The expected duration was absent from the inspected training examples. This coverage gap may have contributed to the error.

## T1_FAIL_2

Example test-080: A doctor visit request specified 1 hour 45 minutes.

- Expected duration: 105 minutes.
- Predicted duration: 145 minutes.
- Training-data check: 0 examples with a 105-minute target.

The expected duration was absent from the inspected training examples. This indicates a training-data coverage gap for this duration expression.

## T1_FAIL_3

Example test-084: A sprint-planning request specified that the meeting would take place over Teams.

- Expected location: Teams.
- Predicted location: null.
- Training-data check: 0 examples with Teams as the target location.

The model failed to extract the meeting platform as the location. The inspected training data contained no examples with Teams as the target location.

## T1_MEMORY_NOTES

For Qwen2.5-0.5B, the memory calculation script estimated:

- Full fine-tuning: 12.045 GB
- LoRA: 5.146 GB
- QLoRA: 4.400 GB

The calculation assumed 4 GB of activation memory, so these are estimates rather than measured runtime values. Actual LoRA training reached a measured peak GPU memory of 8.425 GB. The difference can reflect runtime overhead, CUDA allocations, and other training memory usage.

The training run completed and produced the LoRA adapter. An allocation warning occurred during execution but did not prevent completion.

## T2_CHUNK_ANALYSIS


The corpus contained 4 papers, 83 pages and 43,290 words. Using BAAI/bge-small-en-v1.5, 20% overlap and 15 questions, 100-word chunks produced 543 chunks and 250-word chunks produced 217 chunks. At 100 words, Recall@5 was 0.467 dense, 0.667 BM25 and 0.533 hybrid. At 250 words, it was 0.667 dense, 0.600 BM25 and 0.733 hybrid. The best overall Recall@5 was hybrid with 250-word chunks (11/15). Larger chunks likely preserve more surrounding context, while smaller chunks provide more precise lexical matches. The evaluation set is small, so results should be interpreted cautiously.

## T2_FAIL_1


q03 asks about the purpose of multi-head attention. With 100-word hybrid retrieval, the gold passage appeared at rank 8, outside top-5. Retrieved chunks included general attention material and nearby Transformer sections, but did not prioritize the exact explanation that multiple heads attend to different representation subspaces. This suggests that short chunks can separate the definition from its surrounding explanatory context. A larger chunk or query expansion with terms such as multiple heads and representation subspaces may help.

## T2_FAIL_2


q06 asks why BERT uses next sentence prediction. With 100-word hybrid retrieval, the gold passage ranked 9; with 250-word hybrid retrieval it was not found in the top 10. The top result began with the explanation of the NSP task, but the annotated gold passage starts with the preceding motivation about sentence relationships. This indicates a mismatch between the question's wording, the passage boundary and retrieval scoring. A query expansion using sentence relationship, sentence pair and NSP may improve retrieval.

## T2_BM25


BM25 achieved Recall@5 of 0.667 with 100-word chunks, exceeding dense retrieval (0.467) and hybrid retrieval (0.533). With 250-word chunks, BM25 achieved 0.600, below dense (0.667) and hybrid (0.733). Hybrid retrieval performed best overall at 250 words, suggesting that combining lexical and semantic signals is useful when chunks preserve sufficient context. However, hybrid was not consistently superior at every chunk size. The results support evaluating retrieval methods empirically rather than assuming one method always wins.

## T2_RERANKER


Reranking did not consistently improve Recall@5. At 100 words, BM25 and hybrid Recall@5 decreased from 0.667 to 0.533, while their MRR increased from 0.340 and 0.374 to 0.451. This indicates improved ordering for some relevant results without increasing the number of questions retrieved in the top five. At 250 words, reranking reduced hybrid Recall@5 from 0.733 to 0.600 and MRR from 0.450 to 0.360. The reranker therefore did not improve this benchmark overall. Its value depends on candidate quality and retrieval configuration.

## T3_QUALITY_OBS


Hardware: Google Colab Tesla T4 (14.56 GB reported GPU memory).
Model: Qwen2.5-1.5B-Instruct, FP16 vs Q4_K_M.
Both models received the same 10 prompts with temperature 0, seed 0 and num_ctx 2048. Generation was capped at 200 tokens.

FP16 measured 62.23 tokens/s, 64.68 seconds load time and 3.22 GB Ollama model memory. Q4_K_M measured 135.50 tokens/s, 25.31 seconds load time and 1.11 GB model memory. Q4_K_M was approximately 2.18x faster and used approximately 65.5% less model memory.

Automatic checks: FP16 6/7; Q4_K_M 7/7. Six of ten output pairs were identical, with mean text similarity 0.76. Both produced the correct arithmetic, capital, French translation, summary and JSON fields. FP16 failed the train-speed check because its response was truncated before the final answer; Q4_K_M correctly returned 80 km/h. Longer prime-function and TCP/UDP answers were also truncated by the generation limit. The hash-table answers differed in length and explanation. These observations are from this 10-prompt test only; they do not establish general superiority of quantization.

## T3_KV_CACHE


Verified Qwen2.5-1.5B architecture assumptions used: 28 transformer layers, 2 key/value heads and head dimension 128. With FP16 KV values (2 bytes), cache per token = 2 x 28 x 2 x 128 x 2 = 28,672 bytes = 28 KiB.

At 2,048 tokens per user, estimated KV cache is approximately 0.0587 GB per user. For 20 users it is approximately 1.17 GB (decimal units). This is a theoretical cache estimate; actual serving memory also depends on batching, cache allocation, runtime and context usage.

## T3_QUANT


Measured Ollama comparison on Tesla T4:
FP16: 62.23 tokens/s average, 3.22 GB model memory, 64.68 s load time, 6/7 automatic checks.
Q4_K_M: 135.50 tokens/s average, 1.11 GB model memory, 25.31 s load time, 7/7 automatic checks.

Q4_K_M was approximately 2.18x faster in this benchmark and reduced reported model memory by approximately 65.5%. Six of ten responses were identical and average text similarity was 0.76. Quantization is the better measured choice for this constrained GPU workload, while the small test set and generation truncation limit conclusions about general quality.

## T3_SERVING_PLAN


An actual 20-request concurrent test against the Q4_K_M Ollama model completed with 20/20 successful responses. Total wall-clock time was 10.683 seconds. Individual request times ranged from 3.974 to 10.632 seconds. This verifies successful handling of this test workload, not a guaranteed production SLA.

The serving-plan script estimated memory for 20 users at 2,048 tokens each:
FP16 weights + FP16 KV: 6.25 GB total.
4-bit weights + FP16 KV: 4.04 GB total.
4-bit weights + FP8 KV: 3.45 GB total.
The estimates assumed 2 GB overhead, 90% GPU memory utilisation and 320 GB/s bandwidth. They are model-based estimates, not measured production throughput. FP8 KV is a hypothetical scenario and was not enabled in the Ollama benchmark.

Recommendation: use Q4_K_M for this T4-class constrained deployment, monitor latency and memory under representative traffic, and validate with a production serving engine before making capacity or SLA commitments.
