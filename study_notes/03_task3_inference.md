# Task 3 - Local inference & optimisation

## KV cache
- Transformer decoding mein har naya token pichhle **saare** tokens pe attend karta hai. Attention ko har pichhle token ke **Key** aur **Value** vectors chahiye.
- Bina cache ke har step pe saare pichhle tokens ke K,V dobara compute hote (bahut waste). KV cache = unhe memory mein rakh lo, har step mein sirf naye token ka K,V compute karo.
- Size: `2 (K aur V) x layers x kv_heads x head_dim x bytes x tokens x concurrent users` (`kv_cache_calc.py`). Sequence length aur users ke saath **linearly** badhta hai.
- GQA (grouped-query attention) mein kv_heads kam hote hain => cache chhota (Qwen2.5-7B: 4 kv heads; purana Llama-2-7B: 32 => 9x bada cache).
- Ek GPU pe 20 users ka asli bottleneck aksar **KV cache memory** hota hai, sirf weights nahi.

## Quantization kyun memory bachata hai
- Weights ko 16-bit (2 bytes) ki jagah 4-bit (~0.5 byte + scales) mein store karte hain => weights ~3.5-4x chhote. Group-wise scale/zero-point se approximate values wapas banti hain.
- Cost: rounding error => quality thodi gir sakti hai (khaas kar chhote models/reasoning/precise formats mein); dequantization ka overhead.
- Decode **memory-bandwidth-bound** hota hai (har token ke liye saare weights GPU memory se padhne padte hain), isliye kam bytes => aksar tok/s *badhta* hai.
- Q4_K_M (llama.cpp/Ollama) alag-alag layers ko mixed 4/6-bit mein quantize karta hai; AWQ/GPTQ alag methods hain.
- KV cache alag cheez hai: weights quantize karne se cache apne aap chhota nahi hota (uske liye fp8 KV cache jaise options hote hain).

## Ollama numbers kaise padhein
- `eval_count / eval_duration` = decode tok/s. `prompt_eval_*` = prompt processing. `load_duration` = model load (tok/s mein count nahi hota).
- Memory: `/api/ps` ka `size` (loaded model + KV buffer for `num_ctx`) aur `nvidia-smi` ka delta. Dono slightly alag ho sakte hain; report mein batao kaunsa kya hai.
- Fair comparison: temperature 0, same seed, same prompts, same num_ctx, pehle warm-up (load) phir measure.
- Ollama tags (e.g. `qwen2.5:1.5b-instruct-fp16`, `qwen2.5:1.5b-instruct-q4_K_M`) - `ollama list` / Ollama library page pe **verify karo**, mere yaad pe depend mat karo.

## 20 users on one 24 GB GPU (serving_plan.py)
Teen sawal: (1) memory fit hota hai? (2) decode speed? (3) prefill?
- Memory: weights + 20 x ctx x KV/token + overhead <= ~90% of 24 GB.
- Decode (roofline): har step mein weights ek baar (poore batch ke liye shared) + har user ka KV padhna => `step_time ~ bytes / bandwidth`; per-user tok/s ~ 1/step_time. Yeh **upper bound** hai; real engines kam efficiency dete hain.
- Prefill compute-bound hota hai: ~2 x params FLOPs per token. 20 log ek saath aaye to queue banegi => TTFT (time to first token) badhta hai.
- Engine: vLLM jaisa (continuous batching + PagedAttention + prefix caching). Naive "ek request ek baar" serving se 20 users ke liye bahut slow.
- Assumptions state karo: model size, quantization, context per user, "20 concurrent" vs "20 registered" users, GPU bandwidth. Aur likho ki sabse pehle kya measure karoge.
