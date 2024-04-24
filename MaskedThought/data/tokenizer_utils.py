from transformers import AutoTokenizer
import torch
def prepare_tokenizer(model, cache_dir, **kwargs):
    special_tokens = kwargs.pop("special_tokens", None)
    if special_tokens:
        special_tokens = special_tokens.split(",")
    if model == 'Yale-LILY/brio-xsum-cased':
        model = 'google/pegasus-xsum'
    if model == 'Yale-LILY/brio-cnndm-uncased':
        model = 'facebook/bart-large-cnn'
    auto_tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    auto_tokenizer.add_tokens(special_tokens)
    return auto_tokenizer