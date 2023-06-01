import torch
import os
import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
    AutoTokenizer
)


def get_pretrained_tokenizer(from_pretrained, special_tokens_path=None, replace_unused_tokens:bool=False):
    if special_tokens_path != None:
        with open(special_tokens_path) as f:
            special_tokens = json.load(f)
            
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if special_tokens_path!=None:
                if replace_unused_tokens:
                    BertTokenizer.from_pretrained(
                        from_pretrained , truncation_side="left" , do_lower_case="uncased" in from_pretrained , never_split=special_tokens
                    )
                else:
                    BertTokenizer.from_pretrained(
                        from_pretrained , truncation_side="left" , do_lower_case="uncased" in from_pretrained
                    )
        torch.distributed.barrier()
    
    if special_tokens_path!=None:
        if replace_unused_tokens:
            tokenizer = BertTokenizer.from_pretrained(
                from_pretrained, truncation_side="left" ,do_lower_case="uncased" in from_pretrained , never_split=special_tokens
            )
        else:
            tokenizer = BertTokenizer.from_pretrained(
                from_pretrained, truncation_side="left" ,do_lower_case="uncased" in from_pretrained
            )
            tokenizer.add_special_tokens(special_tokens)
    else:
        tokenizer = BertTokenizer.from_pretrained(
                from_pretrained, truncation_side="left" ,do_lower_case="uncased" in from_pretrained
            )
    return tokenizer


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]
        self.use_segment_ids = _config["use_segment_ids"]
        self.max_image_len = _config["max_image_len"]
        self.max_pred_len = _config["max_pred_len"]
        self.max_source_len = _config["max_source_len"]
        self.mask_prob = _config["mlm_prob"]
        self.whole_word_masking = _config["whole_word_masking"]
        self.mask_source_words = _config["mask_source_words"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer,_config["special_tokens_file"],_config["replace_unused_tokens"])
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )
        if tokenizer == 'gpt2':
            add_special_tokens = _config["add_special_tokens"]
            if add_special_tokens:
                if not os.path.exists(add_special_tokens):
                    raise ValueError(
                        "Additional special tokens file {args.add_special_tokens} not found}"
                    )
                with open(add_special_tokens, "rb") as handle:
                    special_tokens_dict = json.load(handle)
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                print(f"Added {num_added_toks} tokens")
                print(f"All special tokens: {self.tokenizer.all_special_tokens}")
            self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=False)
        else:
            self.mlm_collator = collator(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
            )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            max_image_len=self.max_image_len,
            use_segment_ids=self.use_segment_ids,
            mask_prob = self.mask_prob,
            max_pred_len = self.max_pred_len,
            whole_word_masking = self.whole_word_masking,
            mask_source_words = self.mask_source_words,
            max_source_len = self.max_source_len
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            max_image_len=self.max_image_len,
            use_segment_ids=self.use_segment_ids,
            mask_prob = self.mask_prob,
            max_pred_len = self.max_pred_len,
            whole_word_masking = self.whole_word_masking,
            mask_source_words = self.mask_source_words,
            max_source_len = self.max_source_len
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                max_image_len=self.max_image_len,
                use_segment_ids=self.use_segment_ids,
                mask_prob = self.mask_prob,
                max_pred_len = self.max_pred_len,
                whole_word_masking = self.whole_word_masking,
                mask_source_words = self.mask_source_words,
                max_source_len = self.max_source_len
            )

    def make_no_false_val_dset(self, image_only=False, image_list=None, image_dir=None):
        if image_list == None:
            return self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=image_only,
                max_image_len=self.max_image_len,
                use_segment_ids=self.use_segment_ids,
                mask_prob = self.mask_prob,
                max_pred_len = self.max_pred_len,
                whole_word_masking = self.whole_word_masking,
                mask_source_words = self.mask_source_words,
                max_source_len = self.max_source_len
            )
        else:
            return self.dataset_cls_only_image(
                self.data_dir,
                self.val_transform_keys,
                image_size=self.image_size,
                image_only=image_only,
                image_list=image_list,
            )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            max_image_len=self.max_image_len,
            use_segment_ids=self.use_segment_ids,
            mask_prob = self.mask_prob,
            max_pred_len = self.max_pred_len ,
            whole_word_masking = self.whole_word_masking,
            max_source_len = self.max_source_len
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
