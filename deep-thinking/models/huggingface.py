from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from anchor import checkpoints_root


def build_model_signature(model_type, model_size):
    if model_type == "opt":
        # ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b"]
        return f"facebook/opt-{model_size}"
    if model_type == "gpt2":
        # ["sm", "medium", "large", "xl"]
        if model_size == "sm":
            return "gpt2"
        return f"gpt2-{model_size}"
    if model_type == "e-gpt":
        # ["neo-125M", "neo-1.3B", "neo-2.7B", "j-6B", "neox-20b"]
        return f"EleutherAI/gpt-{model_size}"
    if model_type == "bloom":
        # ["560m", "1b1", "1b7", "3b", "7b1"]
        return f"bigscience/bloom-{model_size}"


def build_tokenizer(model_type, model_size, padding_side="left", use_fast=False):
    sign = build_model_signature(model_type, model_size)
    if not use_fast:
        tok = AutoTokenizer.from_pretrained(sign, padding_side=padding_side, cache_dir=str(checkpoints_root))
    else:
        tok = PreTrainedTokenizerFast.from_pretrained(sign, padding_side=padding_side, cache_dir=str(checkpoints_root))
    if model_type in ["gpt2", "e-gpt"]:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_type, model_size, in_8bit):
    sign = build_model_signature(model_type, model_size)
    model = AutoModelForCausalLM.from_pretrained(
        sign,
        cache_dir=str(checkpoints_root),
        device_map="auto",
        load_in_8bit=in_8bit,
    )
    model.eval()
    return model
