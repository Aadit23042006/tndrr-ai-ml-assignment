"""Build a TINY random Qwen2-style model + tokenizer locally (no downloads).

Used only to test that train_lora.py / evaluate.py run end to end on a machine without
internet or GPU. The model is random and tiny, so its numbers are meaningless: every
results file produced with it is flagged "smoke_test": true and the report builder
refuses to use it.

    python smoke_model.py --data data/train.jsonl --out /tmp/tiny_model
"""
import argparse

from common import completion_text, read_jsonl, short_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=600)
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    rows = read_jsonl(args.data)
    corpus = [short_prompt(r["text"]) + completion_text(r["target"]) for r in rows]

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                  pad_token="<|endoftext|>")

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"tiny model ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) saved to {args.out}")


if __name__ == "__main__":
    main()
