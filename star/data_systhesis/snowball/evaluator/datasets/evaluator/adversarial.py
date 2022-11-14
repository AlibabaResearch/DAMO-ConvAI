import logging
import os
from tqdm import tqdm
import json

from dataclasses import dataclass
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from evaluator.datasets.utils import pad_and_tensorize_sequence

from torch.utils.data.dataset import Dataset
from torch.utils.data import sampler

import random

logger = logging.getLogger(__name__)

class AdversarialDataset(Dataset):
  """
  Dataset for training task: SQL (+ schema) -> text
  """
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path, block_size, translated_logic=False, local_rank=-1,evaluate = False, multi_task=False):
    assert os.path.isfile(file_path)
    logger.info("Creating features from dataset file at {}".format(file_path))

    replacee = 'spider'
    replacer = 'logic2text'
    replaceetoken = ['<SQL>']
    replacertoken = ['<LOGIC>']
    if replacee not in file_path:
        replacee = 'logic2text'
        replacer = 'spider'
        replaceetoken = ['<LOGIC>']
        replacertoken = ['<SQL>']

    if multi_task:
        print('replacee: {}, replacer: {}'.format(replacee, replacer))

    raw_file = open(file_path, encoding='utf-8')
    self.preprocess_data(translated_logic, tokenizer, raw_file, evaluate, replaceetoken)

    if multi_task and os.path.exists(file_path.replace(replacee, replacer)):
        raw_file = open(file_path.replace(replacee, replacer), encoding='utf-8')
        self.preprocess_data(translated_logic, tokenizer, raw_file, evaluate, replacertoken)


  def preprocess_data(self, translated_logic, tokenizer, raw_file, evaluate, datastart):
      logic_key = 'sql'
      text_key = 'question'
      if translated_logic:
          logic_key = 'translated_sql'

      self.examples = []

      invali_idx = []
      add_prefix_space = isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, RobertaTokenizer)

      for idx, line in tqdm(enumerate(raw_file)):
          example = json.loads(line)

          if evaluate:
              logic = example[logic_key]
              text = example[text_key]
              remark = example['remark']
              label = int(example['label'])
              sql_question_token = datastart + [tokenizer.cls_token] + tokenizer.tokenize(logic,
                                                                              add_prefix_space=add_prefix_space) + [
                                       tokenizer.eos_token] + tokenizer.tokenize(text,
                                                                                 add_prefix_space=add_prefix_space) + [
                                       tokenizer.sep_token]

              sql_question_token_ids = tokenizer.convert_tokens_to_ids(sql_question_token)
              self.examples.append({
                  "sql_question_token_ids": sql_question_token_ids,
                  "label": label,
                  "logic": logic,
                  "text": text,
                  "remark": remark})
          else:
              mutated_logic = example["mutated_logic"]
              mutated_text = example['mutated_text']
              original_logic = example['original_logic']
              original_text = example['original_text']

              # negative down-sampling
              if random.random() < 0.3:
                  neg_sql_question_token = datastart + [tokenizer.cls_token] + tokenizer.tokenize(mutated_logic,
                                                                                      add_prefix_space=add_prefix_space) + [
                                               tokenizer.eos_token] + tokenizer.tokenize(original_text,
                                                                                         add_prefix_space=add_prefix_space) + [
                                               tokenizer.sep_token]

                  neg_sql_question_token_ids = tokenizer.convert_tokens_to_ids(neg_sql_question_token)
                  self.examples.append({
                      "sql_question_token_ids": neg_sql_question_token_ids,
                      "label": 0,
                      "logic": mutated_logic,
                      "text": original_text,
                      "remark": 'negative'})

              else:
                  neg_sql_question_token = datastart + [tokenizer.cls_token] + tokenizer.tokenize(original_logic,
                                                                                      add_prefix_space=add_prefix_space) + [
                                               tokenizer.eos_token] + tokenizer.tokenize(mutated_text,
                                                                                         add_prefix_space=add_prefix_space) + [
                                               tokenizer.sep_token]

                  neg_sql_question_token_ids = tokenizer.convert_tokens_to_ids(neg_sql_question_token)
                  self.examples.append({
                      "sql_question_token_ids": neg_sql_question_token_ids,
                      "label": 0,
                      "logic": original_logic,
                      "text": mutated_text,
                      "remark": 'negative'})

              # postive up-sampling
              if random.random() < 0.8:
                  pos_sql_question_token = datastart + [tokenizer.cls_token] + tokenizer.tokenize(original_logic,
                                                                                      add_prefix_space=add_prefix_space) + [
                                               tokenizer.eos_token] + tokenizer.tokenize(original_text,
                                                                                         add_prefix_space=add_prefix_space) + [
                                               tokenizer.sep_token]

                  pos_sql_question_token_ids = tokenizer.convert_tokens_to_ids(pos_sql_question_token)
                  self.examples.append({
                      "sql_question_token_ids": pos_sql_question_token_ids,
                      "label": 1,
                      "logic": original_logic,
                      "text": original_text,
                      "remark": 'positive'})

              else:
                  pos_sql_question_token = datastart + [tokenizer.cls_token] + tokenizer.tokenize(mutated_logic,
                                                                                      add_prefix_space=add_prefix_space) + [
                                               tokenizer.eos_token] + tokenizer.tokenize(mutated_text,
                                                                                         add_prefix_space=add_prefix_space) + [
                                               tokenizer.sep_token]

                  pos_sql_question_token_ids = tokenizer.convert_tokens_to_ids(pos_sql_question_token)
                  self.examples.append({
                      "sql_question_token_ids": pos_sql_question_token_ids,
                      "label": 1,
                      "logic": mutated_logic,
                      "text": mutated_text,
                      "remark": 'positive'})
    
  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    example = self.examples[i]

    return example
  

@dataclass
class DataCollatorForAdversarial:
  """

  """
  tokenizer: PreTrainedTokenizer

  def __post_init__(self):
    self.label_bos_id = self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id

  def collate_batch(self, examples):

    sql_question_ids_sequences = [example["sql_question_token_ids"] for example in examples]
    logics = [example["logic"] for example in examples]
    texts = [example["text"] for example in examples]
    labels = [example["label"] for example in examples]
    remarks = [example["remark"] for example in examples]
    
    padded_sql_question_ids_tensor = pad_and_tensorize_sequence(
      sql_question_ids_sequences, padding_value=self.tokenizer.pad_token_id)
    
    
    try:
        label_tensor = pad_and_tensorize_sequence(
          labels, tensorize = True, padding_value=self.tokenizer.pad_token_id)
    except:
        print(labels)

    return {
      "encoder_input_ids": padded_sql_question_ids_tensor,
      "labels": label_tensor,
      "logics": logics,
      "texts": texts,
      "remarks": remarks,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id,
      "label_padding_id": self.tokenizer.pad_token_id
    }
