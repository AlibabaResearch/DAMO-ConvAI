import logging
import os
from tqdm import tqdm
import json
import random

from dataclasses import dataclass
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from evaluator.datasets.utils import pad_and_tensorize_sequence

from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

class RelogicDataset(Dataset):
  """
  Dataset for training task: logic (+ schema) -> text
  """
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path, block_size, local_rank=-1,translated_logic=False, snow_ball=False, preprocess_path = None, 
               mutation_data_path = None, aug_sample_num = 5, augmented=False, snowball_iteration = 0, total_snowball_iteration = 1, multi_task = False):
    assert os.path.isfile(file_path)
    logger.info("Creating features from dataset file at {}".format(file_path))
    add_prefix_space = isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, RobertaTokenizer)
    self.examples = []
    invalid_idx = []
    logic_key = 'query'
    text_key = 'question'
    total = 0

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
    self.process_data(augmented, snowball_iteration, total_snowball_iteration, aug_sample_num, raw_file,
                      tokenizer,
                      add_prefix_space,
                      translated_logic, preprocess_path, snow_ball, mutation_data_path, logic_key, text_key,
                      invalid_idx, replaceetoken)
    if multi_task and os.path.exists(file_path.replace(replacee, replacer)):
        raw_file = open(file_path.replace(replacee, replacer), encoding='utf-8')
        if translated_logic and preprocess_path:
            preprocess_path = preprocess_path.replace(replacee, replacer)
        if snow_ball and mutation_data_path:
            mutation_data_path = mutation_data_path.replace(replacee, replacer)
        self.process_data(augmented, snowball_iteration, total_snowball_iteration, aug_sample_num, raw_file, tokenizer,
                 add_prefix_space,
                 translated_logic, preprocess_path, snow_ball,
                          mutation_data_path, logic_key, text_key, invalid_idx, replacertoken)

    logger.info('Invalid examples ids: {}'.format(invalid_idx))
  def process_data(self, augmented, snowball_iteration, total_snowball_iteration, aug_sample_num, raw_file, tokenizer, add_prefix_space,
                   translated_logic, preprocess_path, snow_ball, mutation_data_path, logic_key, text_key, invalid_idx, datastart):
      if augmented:
          logic_dict = {}
          max_sample_num = int((float(snowball_iteration) / float(total_snowball_iteration)) * aug_sample_num)
          logger.info("Max sampling number for generator augmentation is {}".format(max_sample_num))
          for line in raw_file:
              example = json.loads(line)

              mutated_logic = example["mutated_logic"]
              mutated_text = example["mutated_text"]
              original_logic = example["original_logic"]
              original_text = example["original_text"]

              if original_logic not in logic_dict:
                  logic_dict[original_logic] = 0

                  original_text_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(original_text,
                                                                                    add_prefix_space=add_prefix_space) + [
                                             tokenizer.sep_token]
                  original_logic_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(original_logic,
                                                                                     add_prefix_space=add_prefix_space) + [
                                              tokenizer.sep_token]
                  original_text_token_ids = tokenizer.convert_tokens_to_ids(original_text_tokens)
                  original_logic_token_ids = tokenizer.convert_tokens_to_ids(original_logic_tokens)

                  self.examples.append({
                      "logic": mutated_logic,
                      "original_logic": original_logic,
                      "text_token_ids": original_text_token_ids,
                      "logic_token_ids": original_logic_token_ids})

              if logic_dict[original_logic] <= max_sample_num:
                  logic_dict[original_logic] += 1

                  mutated_text_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(mutated_text,
                                                                                   add_prefix_space=add_prefix_space) + [
                                            tokenizer.sep_token]
                  mutated_logic_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(mutated_logic,
                                                                                    add_prefix_space=add_prefix_space) + [
                                             tokenizer.sep_token]
                  mutated_text_token_ids = tokenizer.convert_tokens_to_ids(mutated_text_tokens)
                  mutated_logic_token_ids = tokenizer.convert_tokens_to_ids(mutated_logic_tokens)

                  self.examples.append({
                      "logic": mutated_logic,
                      "original_logic": original_logic,
                      "text_token_ids": mutated_text_token_ids,
                      "logic_token_ids": mutated_logic_token_ids})



      else:
          raw_examples = json.load(raw_file)

          if translated_logic and preprocess_path:
              preprocess_file = open(preprocess_path, encoding='utf-8')
              preprocess_mapping = json.load(preprocess_file)

          if snow_ball and mutation_data_path:
              mutation_file = open(mutation_data_path, encoding='utf-8')
              mutation_mapping = json.load(mutation_file)

          for idx, example in tqdm(enumerate(raw_examples)):
              logic = example[logic_key]
              text = example[text_key]
              if snow_ball:
                  try:
                      sample_num = min(aug_sample_num, len(mutation_mapping[logic]))
                      if translated_logic:
                          mutations = random.sample(list(mutation_mapping[logic].values()), sample_num)
                      else:
                          mutations = random.sample(list(mutation_mapping[logic].keys()), sample_num)
                      if translated_logic:
                          logic = preprocess_mapping[logic]

                  except:
                      invalid_idx.append(idx)
                      continue

                  for mutated_logic in mutations:
                      try:
                          text_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(text,
                                                                                   add_prefix_space=add_prefix_space) + [
                                            tokenizer.sep_token]
                          logic_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(mutated_logic,
                                                                                    add_prefix_space=add_prefix_space) + [
                                             tokenizer.sep_token]
                          text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                          logic_token_ids = tokenizer.convert_tokens_to_ids(logic_tokens)

                          self.examples.append({
                              "logic": mutated_logic,
                              "original_logic": logic,
                              "text_token_ids": text_token_ids,
                              "logic_token_ids": logic_token_ids})
                      except:
                          continue

              else:
                  if translated_logic:
                      try:
                          logic = preprocess_mapping[logic]
                      except:
                          invalid_idx.append(idx)
                          continue

                  text_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(text, add_prefix_space=add_prefix_space) + [
                      tokenizer.sep_token]

                  logic_tokens = datastart + [tokenizer.cls_token] + tokenizer.tokenize(logic,
                                                                            add_prefix_space=add_prefix_space) + [
                                     tokenizer.sep_token]
                  text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                  logic_token_ids = tokenizer.convert_tokens_to_ids(logic_tokens)
                  self.examples.append({
                      "logic": logic,
                      "original_logic": logic,
                      "text_token_ids": text_token_ids,
                      "logic_token_ids": logic_token_ids})



  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]

@dataclass
class DataCollatorForRelogic:
  """

  """
  tokenizer: PreTrainedTokenizer

  def __post_init__(self):
    self.label_bos_id = self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id

  def collate_batch(self, examples):
    logics = [example["logic"] for example in examples]
    original_logic = [example["original_logic"] for example in examples]
    text_ids_sequences = [example["text_token_ids"] for example in examples]
    logic_ids_sequences = [example["logic_token_ids"] for example in examples]

    padded_text_ids_tensor = pad_and_tensorize_sequence(
      text_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    padded_logic_ids_tensor = pad_and_tensorize_sequence(
      logic_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    return {
      "logics": logics,
      "original_logic": original_logic,
      "input_ids": padded_logic_ids_tensor,
      "labels": padded_text_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id,
      "label_padding_id": self.tokenizer.pad_token_id
    }
