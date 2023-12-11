#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import math
from typing import List, Union
import torch
from tqdm.auto import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from uie.extraction.predict_parser import get_predict_parser
from uie.extraction.record_schema import RecordSchema
from uie.seq2seq.constraint_decoder import get_constraint_decoder
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer


class RecordExtractor:
    """记录抽取器，动态更换解码的 Schema
    """

    def __init__(self,
                 tokenizer: T5TokenizerFast,
                 model: T5ForConditionalGeneration,
                 tree_parser=None, constraint_decoder=None, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.tree_parser = tree_parser
        self.constraint_decoder = constraint_decoder

        self.device = device

        self.to_remove_token_list = list()
        if tokenizer.bos_token:
            self.to_remove_token_list += [tokenizer.bos_token]
        if tokenizer.eos_token:
            self.to_remove_token_list += [tokenizer.eos_token]
        if tokenizer.pad_token:
            self.to_remove_token_list += [tokenizer.pad_token]

        sys.stdout.write(f'to remove: {self.to_remove_token_list}\n')

    @staticmethod
    def from_pretrained(model_path, device: Union[int, List[int]] = None):
        """读取预训练模型到指定 device

        Args:
            model_path (str): 模型路径
            device ([int]], optional): Transformer 模型所在 GPU

        Returns:
            RecordExtractor : 记录抽取器不包含的 tree_parser constraint_decoder
        """
        print(f'load model from {model_path} ...')
        if "t5-char" in model_path:
            tokenizer = T5BertTokenizer.from_pretrained(model_path)
        else:
            tokenizer = T5TokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.eval()

        if device is not None:
            print("Moving mdoel to %s" % device)
            model = model.to(device)

        return RecordExtractor(
            tokenizer=tokenizer,
            model=model,
            tree_parser=None,
            constraint_decoder=None,
            device=device,
        )

    def preds_to_sequence_texts(self, preds):
        """ 预测结果进行后处理，Index -> Token

        Args:
            preds ([type]): [description]

        Returns:
            List[str]: Seq2Seq 模型预测结果
        """
        test_preds = self.tokenizer.batch_decode(
            preds,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        test_preds = [self.postprocess_text(pred) for pred in test_preds]
        return test_preds

    def postprocess_text(self, x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in self.to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()

    @staticmethod
    def load_record_schema(tokenizer,
                           record_schema: RecordSchema,
                           decoding_schema='spotasoc',
                           prefix='meta: ',
                           task_name='record'):

        # 读取解析器
        tree_parser = get_predict_parser(
            decoding_schema=decoding_schema,
            label_constraint=record_schema,
        )

        # 读取受限解码器
        constraint_decoder = get_constraint_decoder(
            tokenizer=tokenizer,
            type_schema=record_schema,
            decoding_schema=decoding_schema,
            task_name=task_name,
            source_prefix=prefix,
        )

        return tree_parser, constraint_decoder

    def renew_record_schema(self, record_schema, decoding_schema='spotasoc', prefix='meta: ', task_name='record'):

        """ 使用新的解码框架 """
        sys.stdout.write(f"Renew schema: \n`{record_schema}`\n")
        sys.stdout.write(f"Renew decoding: `{decoding_schema}`\n")
        sys.stdout.write(f"Renew prefix: `{prefix}`\n")

        tree_parser, constraint_decoder = self.load_record_schema(
            tokenizer=self.tokenizer,
            record_schema=record_schema,
            decoding_schema=decoding_schema,
            prefix=prefix,
            task_name=task_name,
        )
        self.tree_parser = tree_parser
        self.constraint_decoder = constraint_decoder

    def extract_record(self, text_list, constrained_decoding=False, batch_size=32, max_length=512):
        text_list = [self.constraint_decoder.source_prefix + text for text in text_list]

        model_inputs = self.tokenizer(
            text_list,
            max_length=max_length,
            padding=False,
            truncation=True
        )

        num_batch = math.ceil(len(text_list) / batch_size)
        outputs = list()
        self.model.eval()

        for index in tqdm(range(num_batch)):
            batch_model_inputs = {
                k: v[index * batch_size: (index + 1) * batch_size]
                for k, v in model_inputs.items()
            }

            batch_model_inputs = self.tokenizer.pad(
                batch_model_inputs,
                padding=True,
                return_tensors="pt",
            )

            if self.device is not None:
                batch_model_inputs = batch_model_inputs.to(self.device)

            def prefix_allowed_tokens_fn(batch_id, sent):
                src_sentence = batch_model_inputs['input_ids'][batch_id]
                return self.constraint_decoder.constraint_decoding(
                    src_sentence=src_sentence,
                    tgt_generated=sent
                )

            with torch.no_grad():
                batch_outputs = self.model.generate(
                    input_ids=batch_model_inputs["input_ids"],
                    attention_mask=batch_model_inputs["attention_mask"],
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if constrained_decoding else None,
                    max_length=max_length,
                )

            outputs += batch_outputs

        assert len(outputs) == len(text_list)

        # Index -> Str
        sequence_list = self.preds_to_sequence_texts(outputs)

        # Str -> Record
        record_list, _ = self.tree_parser.decode(
            pred_list=sequence_list,
            gold_list=[],
            text_list=text_list
        )
        record_list = [event['pred_record'] for event in record_list]

        return text_list, sequence_list, record_list
