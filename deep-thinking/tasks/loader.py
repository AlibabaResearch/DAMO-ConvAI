import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger("task")


class TokenizedForMCRightPad(Dataset):
    def __init__(self, data, tok: PreTrainedTokenizer, prompt_fn):
        # data: [query: str, choices: list(str)]
        self.tok = tok
        self.prompt_fn = prompt_fn
        self.max_length = self._find_max_length(data)
        self.data = self._build_mc_data(data)
        logger.info(f"Tokenization finished: {len(self.data)}, max_length={self.max_length}")

    def _find_max_length(self, data):
        max_len = 0

        def tok_len(t):
            return len(self.tok.encode(t))

        for ex in tqdm(data, desc="Data preprocessing(1/2)"):
            query = ex["query"]
            len_choices = [tok_len(self.prompt_fn(query, c)[1]) for c in ex["choices"]]
            max_len = max(max_len, *len_choices)

        return max_len

    def _build_mc_data(self, data):
        processed = []
        num_choices = set(len(e["choices"]) for e in data)
        if not len(num_choices) == 1:
            raise ValueError(f"Queries have different number of choices, which is not supported! #choices: {num_choices}")
        for ex in tqdm(data, desc="Data preprocessing(2/2)"):
            query, choices = ex["query"], ex["choices"]
            processed_input = [self.prompt_fn(query, choice) for choice in choices]
            # print(f"Question: [{processed_input[0][0]}]")
            # print("-" * 30)
            # for idx, cs in enumerate(processed_input):
            #     print(f"Comb {idx}: ")
            #     print(f"[{cs[1]}]")
            #     print("-" * 30)
            # exit(0)
            processed_input = [self.tokenize(t_query, t_full) for t_query, t_full in processed_input]
            processed.append(processed_input)

        logger.info("Multiple choice dataset: finish!")
        logger.info(f"Num of choices: {num_choices}")
        return processed

    def tokenize_demonstration(self, demonstration):
        e = self.tok(demonstration)
        return torch.LongTensor(e["input_ids"]), torch.LongTensor(e["attention_mask"])  # no padding

    def tokenize(self, only_query, full_text):
        tok_only_query = self.tok(only_query, add_special_tokens=False)
        tok_full_no_padding = self.tok(full_text, add_special_tokens=False)
        tok_full = self.tok(
            full_text,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=False,
        )  # <pad> is not a special token
        # tok_only_query = self.tok(only_query)
        # tok_full_no_padding = self.tok(full_text)
        # tok_full = self.tok(
        #     full_text,
        #     padding="max_length",
        #     max_length=self.max_length,
        # )  # <pad> is not a special token

        # print(f"tok_only_query: {self.tok.convert_ids_to_tokens(tok_only_query.input_ids)}")
        # print(f"tok_full_no_padding: {self.tok.convert_ids_to_tokens(tok_full_no_padding.input_ids)}")
        # print(f"tok_full: {self.tok.convert_ids_to_tokens(tok_full.input_ids)}")
        # exit(0)

        len_full = len(tok_full_no_padding.input_ids)
        len_query = len(tok_only_query.input_ids)
        e = {
            "input_ids": tok_full.input_ids,
            "attention_mask": tok_full.attention_mask,
            "choice_start": len_query,
            "choice_end": len_full,
        }
        # print("Attn:")
        # print(tok_full.attention_mask)
        # print("input_ids:")
        # print(tok_full.input_ids)

        dcd_sp = self.tok.convert_ids_to_tokens(tok_full.input_ids, skip_special_tokens=False)

        # print(f'{e["choice_start"]}: {e["choice_end"]} = [{self.tok.convert_tokens_to_string(dcd_sp[e["choice_start"] : e["choice_end"]])}]')

        return e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def _get_one_item(e):
            return torch.LongTensor(e["input_ids"]), torch.LongTensor(e["attention_mask"]), e["choice_start"], e["choice_end"]

        es = self.data[idx]
        # num_choices * (input_ids, attn, start_idx, end_idx)
        # input_ids, attn: [B, L]
        # start_idx, end_idx: [B, ]
        return [_get_one_item(e) for e in es]


if __name__ == "__main__":
    from anchor import hf_datasets_root

    import datasets

    csqa1 = datasets.load_dataset("commonsense_qa", cache_dir=str(hf_datasets_root), split="validation")
