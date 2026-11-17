"""Small helpers so the scripts work across transformers versions."""
import torch


def pick_dtype():
    """bf16 if the GPU supports it (A100/L4/H100), fp16 on T4, fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_causal_lm(path, dtype=None, **kwargs):
    """from_pretrained with `dtype=` (new name) falling back to `torch_dtype=` (old name)."""
    from transformers import AutoModelForCausalLM
    if dtype is None:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
