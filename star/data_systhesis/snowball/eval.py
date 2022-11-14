
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
import json

from transformers import (
  MODEL_WITH_LM_HEAD_MAPPING,
  AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
)
from generator.models.relogic import RelogicModel
from generator.datasets.text_generation.relogic import RelogicDataset, DataCollatorForRelogic
from generator.scorers.text_generation import TextGenerationScorer
from generator.trainer_refer import Generator_Trainer
from generator.training_args import Generator_TrainingArguments

from evaluator.models.adversarial_evaluator import AdversarialModel
from evaluator.datasets.evaluator.adversarial import AdversarialDataset, DataCollatorForAdversarial
from evaluator.scorers.adv_eval import EvalScorer
from evaluator.trainer import Evaluator_Trainer
from evaluator.training_args import Evaluator_TrainingArguments

from snowball_utils_refer import train_generator, train_evaluator, augment_data, create_save_dir


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



parser = HfArgumentParser((Generator_TrainingArguments, Evaluator_TrainingArguments))
gen_training_args, eval_training_args = parser.parse_args_into_dataclasses()
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large',cache_dir=None)
block_size = min(1, tokenizer.max_len)
file1 = 'question_sql.json'
file2 = 'logic.json'
raw_test_data_path = os.path.join('../preprocessed', file1)
preprocess_test_data_path = os.path.join('../preprocessed', file2)
test_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=raw_test_data_path,
                                       preprocess_path=preprocess_test_data_path,
                                       block_size=block_size,
                                       translated_logic=True,
                                       snow_ball=False)

generator_path = os.path.join('saves', 'checkpoint-epoch-10.0/pytorch_model.bin')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = RelogicModel('facebook/bart-large').to(device) 
generator_param = torch.load(generator_path)
generator.load_state_dict(generator_param)
logger.info("Successfully loaded generator")

data_collator = DataCollatorForRelogic(tokenizer=tokenizer)
label_bos_id = data_collator.label_bos_id
label_eos_id = data_collator.label_eos_id
scorer = TextGenerationScorer(bos_id=label_bos_id, eos_id=label_eos_id, tokenizer=tokenizer, output_path='test_my')
trainer = Generator_Trainer(
        model=generator,
        args=gen_training_args,
        data_collator=data_collator,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=scorer,
        prediction_loss_only = False,
        model_name=gen_training_args.gen_model,
        output_dump_dir = 'test_my',
        reranker = None
    )

output_test = trainer.predict(test_dataset=test_dataset, mode_name='test')
ids = output_test.predictions.tolist()
start = 0
result = []
for item in output_test.predictions_size.tolist():
    temp = ids[start:start+item]
    result.append(tokenizer.decode(temp,skip_special_tokens=True).strip())
    start += item
with open('../preprocessed/final_generation.json','w') as f:
    json.dump(result,f)
