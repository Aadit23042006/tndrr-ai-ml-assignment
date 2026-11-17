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

TODO

## T2_FAIL_1

TODO

## T2_FAIL_2

TODO

## T2_BM25

TODO

## T2_RERANKER

TODO

## T3_QUALITY_OBS

TODO

## T3_KV_CACHE

TODO

## T3_QUANT

TODO

## T3_SERVING_PLAN

TODO
