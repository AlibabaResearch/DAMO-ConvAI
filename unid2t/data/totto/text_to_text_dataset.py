import json
import torch

from transformers import T5Tokenizer

from data.abstract_dataset import AbstractDataset


class Text2TextDataset(AbstractDataset):
    def __init__(self, data_dir, datatype, tokenizer: T5Tokenizer, special_tokens: list,
                 max_inp_len: int,
                 max_target_len: int,
                 n_example=-1, task_source_prefix=None):
        
        super(Text2TextDataset, self).__init__(data_dir=data_dir, datatype=datatype, tokenizer=tokenizer,
                                               special_tokens=special_tokens,
                                               max_inp_len=max_inp_len,
                                               max_target_len=max_target_len,
                                               n_example=n_example, task_source_prefix=task_source_prefix)

    def __getitem__(self, item):
        example: dict = self.examples[item]

        metadata_str = example['subtable_metadata_str']

        # subtable_str = example['subtable_str']

        # enc_inp = metadata_str + ' ' + subtable_str
        if self.task_source_prefix is not None and len(self.task_source_prefix) > 0:
            enc_inp = self.task_source_prefix + metadata_str
        else:
            enc_inp = metadata_str
        enc_inp_tokens = self.tokenizer.tokenize(enc_inp)
        if self.max_inp_len > 0:
            enc_inp_tokens = enc_inp_tokens[:self.max_inp_len]

        if 'sentence_annotations' in example:
            target_text = []
            for ann in example['sentence_annotations']:
                target_text.append(ann['final_sentence'])
        else:
            target_text = None

        # target_text = example['sentence_annotations'][0]['final_sentence'] if 'sentence_annotations' in example else None
        dec_tokens = None if target_text is None else self.tokenizer.tokenize(target_text[0])

        enc_inp = self.tokenizer.convert_tokens_to_ids(enc_inp_tokens)
        dec_token_ids = self.tokenizer.convert_tokens_to_ids(dec_tokens) if dec_tokens is not None else None
        dec_inp = [self.bos_token_id] + dec_token_ids if dec_token_ids is not None else None
        dec_out = dec_token_ids + [self.eos_token_id] if dec_token_ids is not None else None

        if self.max_target_len > 0:
            dec_inp = dec_inp[:self.max_target_len]
            dec_out = dec_out[:self.max_target_len]

        return_example = {
            'enc_inp': enc_inp,
            'dec_inp': dec_inp,
            'dec_out': dec_out,
            'target_text': target_text
        }

        return return_example

    def collate_fn(self, batch):
        enc_inp = []
        dec_inp = []
        dec_out = []

        target_text = []
        for example in batch:
            enc_inp.append(example['enc_inp'])
            if example['dec_inp'] is not None:
                dec_inp.append(example['dec_inp'])
                dec_out.append(example['dec_out'])
                target_text.append(example['target_text'])

        enc_inp = self.sequence_padding(enc_inp, self.pad_token_id)
        dec_inp = self.sequence_padding(dec_inp, self.pad_token_id) if len(dec_inp) else None
        dec_out = self.sequence_padding(dec_out, -100) if len(dec_out) else None
        enc_attention_mask = enc_inp.ne(self.pad_token_id)

        collated_batch = {
            'enc_inp': enc_inp,
            'enc_attention_mask': enc_attention_mask,
            'dec_inp': dec_inp,
            'label': dec_out,
            'target_text': target_text
        }

        return collated_batch







