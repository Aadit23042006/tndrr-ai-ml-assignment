# Task 1 - LoRA / QLoRA

## Idea
Full fine-tuning mein har weight matrix `W` (d_out x d_in) update hoti hai. LoRA mein `W` **freeze** hoti hai aur
sirf ek chhota "delta" seekha jata hai:

    W_effective = W + (alpha / r) * B @ A        A: r x d_in,  B: d_out x r

- `r` (rank) = delta ki capacity. r chhota => kam parameters, kam capacity.
- `B` zero se start hoti hai, isliye training ke pehle step pe model bilkul base model jaisa hota hai.
- `alpha / r` = scaling. Common choice: alpha = 2r (jaise r=16, alpha=32). Effective "learning strength" isse badalti hai.
- Ek layer ke trainable params = `r * (d_in + d_out)`. Poore model ka ~0.5-1% hi train hota hai.
- `target_modules`: kin layers pe adapter lagana hai. Attention-only (q,v) sasta; **all linear layers** (q,k,v,o,gate,up,down)
  usually behtar quality deta hai (QLoRA paper ka bhi yehi observation hai). Is repo ka default: all linear.

## Is repo ke defaults aur "kyun"
| setting | value | sochne ka tareeka |
|---|---|---|
| r | 16 | task narrow hai (fixed JSON); 4 shayad kaafi hota, 64 overkill. `ablation.sh` se **khud measure karo** |
| alpha | 32 | 2r convention |
| dropout | 0.05 | 1500 examples pe halka regularisation |
| lr | 2e-4 | LoRA mein full-FT se ~10x bada lr chalta hai (kyunki sirf chhote adapters train ho rahe hain) |
| epochs | 3 | train loss dekho; zyada epochs = overfit ka risk (test pe check karo) |
| loss | sirf completion pe | prompt tokens ka label = -100, model ko sirf JSON likhna seekhna hai |

Apne `my_answers.md` mein wahi likho jo **tumne** ablation/loss curve mein dekha. Agar ablation nahi chalaya to
yeh mat likho ki "r=64 worse tha".

## Base vs fine-tuned - fair comparison
- Base model ko do tarah se test kiya: (1) wahi short prompt, (2) instruction + 1 example wala prompt. Dusra *strong baseline* hai;
  isse tum dikha sakte ho ki fine-tuning ka fayda sirf "prompt engineering" se zyada hai ya nahi.
- Test set mein `shifted` hissa hai (naye phrasing/format): yahan failures expected hain. "Kyun fail hua?" ka jawab
  training data dekh ke do (grep karo: kya yeh format train mein tha?).

## 7B memory math (memory_calc.py ye hi karta hai)
P = 6.74e9 params. Bytes per parameter:
- **Full FT (mixed precision, AdamW):** weights bf16 (2) + grads bf16 (2) + Adam m,v fp32 (8) + fp32 master copy (4) = **16 B/param**
  => 16 x 6.74e9 ~ 108 GB + activations => **~110 GB** (kam se kam ~85 GB agar master copy na ho). Isliye ek 80 GB GPU kaafi nahi.
- **LoRA:** frozen weights bf16 = 2 B x P = 13.5 GB; adapters (~40M params) ke liye weights+grads+Adam = 16 B x 40M = 0.64 GB;
  + activations => **~18 GB**.
- **QLoRA:** frozen weights 4-bit NF4 ~ 0.5 B x P = ~3.5 GB; adapters 0.64 GB; + activations => **~8 GB**.
- Activations batch size, sequence length aur gradient checkpointing pe depend karte hain - isliye yeh **assumption** hai, calculator mein `--act-gb`.

## QLoRA kya alag karta hai
- Base weights 4-bit **NF4** (normal-distributed weights ke liye designed) mein store hote hain; double quantization = scale constants ko bhi quantize.
- Forward/backward mein block-wise dequantize hoke bf16 mein matmul hota hai; gradient sirf LoRA adapters ko milta hai.
- Paged optimizers = memory spikes handle karne ke liye. Fayda: memory; nuksan: dequantization ki wajah se thoda slow.
