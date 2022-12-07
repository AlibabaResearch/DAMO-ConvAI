import argparse
import json
import os
import copy
import random
import numpy as np
import logging
import re
import pprint

# config
# use ROLE embedding to separate user and system utterances
# use special token [THIS] 作为 当前轮语句的标示
# token:    [CLS] sys utter [SEP] user utter [SEP] sys utter [THIS] user utter [THIS]
# position:   0   1    2     3     4    ...
# role        0   0    0      1    1    1     0      0   0      1     1    1      1
# intent classify at the [CLS] token
# the first sentence has to be system utter
import torch

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.few_shot_num = args.few_shot  # 小样本训练数目
        self.scenario = args.scenario  # 场景
        self.max_history = args.max_history
        self.max_utter = args.max_utter

        self.data_dir = 'data'
        
        logger.info("Loading files of the scenario %s ->" % self.scenario)
        
        self.train_ids = json.load(open(os.path.join(self.data_dir, self.scenario, 'train_ids.json')))
        self.dev_ids = json.load(open(os.path.join(self.data_dir, self.scenario, 'dev_ids.json')))
        self.test_ids = json.load(open(os.path.join(self.data_dir, self.scenario, 'test_ids.json')))
        self.int2act = json.load(open(os.path.join(self.data_dir, self.scenario, 'int2act.json')))

        self.intents = ['拒识'] + list(self.int2act.keys())
        logger.info("intents: %s" % str(self.intents))
        logger.info("total: %d" % len(self.intents))

        self.intent_dic = dict(zip(self.intents, range(len(self.intents))))
        self.intent_dic_inv = dict(zip(range(len(self.intents)), self.intents))
        
        
        with open(os.path.join(self.data_dir, self.scenario, 'selected_simu_data.jsonasr.json')) as f:
            self.simu_data = json.load(f)
        with open(os.path.join(self.data_dir, self.scenario, 'real_data.jsonasr.json')) as f:
            self.real_data = json.load(f)
        
        self.all_data = {}
        for k in self.simu_data: self.all_data[k] = copy.deepcopy(self.simu_data[k])
        for k in self.real_data: self.all_data[k] = copy.deepcopy(self.real_data[k])
        
        del self.simu_data
        del self.real_data
        
        self.train_dialogs = {k:self.all_data[k] for k in self.train_ids}
        logger.info("#training dialogs: %d" % len(self.train_dialogs))
        self.dev_dialogs = {k:self.all_data[k] for k in self.dev_ids}
        logger.info("#dev dialogs: %d" % len(self.dev_dialogs))
        self.test_dialogs = {k:self.all_data[k] for k in self.test_ids}
        logger.info("#test dialogs: %d" % len(self.test_dialogs))

        
        if not os.path.exists(os.path.join(self.data_dir, self.scenario, 'train_instances.json')) or \
                self.args.overwrite_cache:
            self.train_instances = self.get_instance(self.train_dialogs)
            json.dump(self.train_instances, open(os.path.join(self.data_dir, self.scenario, 'train_instances.json'), 'w'), ensure_ascii=False, indent=1)
            
        else:
            logger.info("load train_instances from existing files")
            self.train_instances = json.load(open(os.path.join(self.data_dir, self.scenario, 'train_instances.json')))
        
        logger.info("#training examples: %d" % len(self.train_instances))
        
        
        if not os.path.exists(os.path.join(self.data_dir, self.scenario, 'dev_instances.json')) or \
                self.args.overwrite_cache:
            self.dev_instances = self.get_instance(self.dev_dialogs, False)
            json.dump(self.dev_instances,
                      open(os.path.join(self.data_dir, self.scenario, 'dev_instances.json'), 'w'), ensure_ascii=False,
                      indent=1)
        else:
            logger.info("load dev_instances from existing files")
            self.dev_instances = json.load(open(os.path.join(self.data_dir, self.scenario, 'dev_instances.json')))
            
        logger.info("#dev examples: %d" % len(self.dev_instances))

        if not os.path.exists(os.path.join(self.data_dir, self.scenario, 'test_instances.json')) or \
                self.args.overwrite_cache:
            self.test_instances = self.get_instance(self.test_dialogs, False)
            json.dump(self.test_instances,
                      open(os.path.join(self.data_dir, self.scenario, 'test_instances.json'), 'w'), ensure_ascii=False,
                      indent=1)
        else:
            logger.info("load test_instances from existing files")
            self.test_instances = json.load(open(os.path.join(self.data_dir, self.scenario, 'test_instances.json')))
        
        logger.info("#test examples: %d" % len(self.test_instances))
       
    
    def get_instance(self, dialogs, is_train=True):
        # 根据 ids 选出对话，并转化成 turn-level instances
        # 记录 turn_id, dial_id||turn_XX||origin, dial_id||turn_XX||ASR_X
        turn_data = []
        for dial_id, dial in dialogs.items():
            history = []
            for turn_id, turn in enumerate(dial):
                if 'usr_query' in turn:
                    turn_data.append(
                        {
                            'guid': '%s||turn_%d||origin'% (dial_id, turn_id),
                            'history': copy.deepcopy(history),
                            'intent': turn['usr_intent'],
                            'query': self.normalize(turn['usr_query'])
                        }
                    )
                    if 'asr_results' in turn:
                        if is_train:
                            asr_ans = set(turn['asr_results'])
                            if turn['usr_query'] in asr_ans:
                                asr_ans.remove(turn['usr_query'])
                        else:
                            asr_ans = turn['asr_results']
                        for asr_id, asr in enumerate(asr_ans):
                            turn_data.append(
                                {
                                    'guid': '%s||turn_%d||asr_%d' % (dial_id, turn_id, asr_id),
                                    'history': copy.deepcopy(history),
                                    'intent': turn['usr_intent'],
                                    'query': self.normalize(asr)
                                }
                            )
                if 'sys_response' in turn:
                    if self.args.use_sys_act:
                        history.append(self.normalize(turn['sys_action']))
                    else:
                        history.append(self.normalize(turn['sys_response']))
                if 'usr_query' in turn:
                    history.append(self.normalize(turn['usr_query']))

        all_examples = []
        for inst in turn_data:
            guid = inst['guid']
            history = inst['history']
            intent = inst['intent']
            query = inst['query']
            input_tok, input_id, input_seg = self.convert2tokens(history, query)
    
            all_examples.append(
                {
                    'guid': guid,
                    'history': json.dumps(history, ensure_ascii=False),
                    'intent': intent,
                    'query': query,
                    'label': self.intent_dic[intent],
                    'input_tok': json.dumps(input_tok, ensure_ascii=False),
                    'input_id': json.dumps(input_id, ensure_ascii=False),
                    'input_seg': json.dumps(input_seg, ensure_ascii=False)
                }
            )
        return all_examples
        
    
    def normalize(self, sent):
        sent = str(sent)
        sent = re.sub(r"[“”]", "\"", sent)
        sent = re.sub(r"[…]{1,10}", "。", sent)
        sent = re.sub(r"\[NUM\]", "number", sent)  # 一些特殊token 转化成单词
        sent = re.sub(r"\.", " ", sent)
        if '"{"action": "repeat"}"' in sent: sent = 'repeat'
        return sent
    
    
    def get_batch_iterator(self, exmaples, batch_size=16):
        bid = 0
        while bid<len(exmaples):
            # print('bid:', bid, bid+batch_size)
            batch_examples = exmaples[bid:bid+batch_size]
            input_ids, input_mask, input_type_ids, matching_label_id , guids\
                = self.convert_examples_to_features(batch_examples)
            bid += batch_size
            yield {
                'guids': copy.deepcopy(guids),
                'input_ids': torch.LongTensor(input_ids),
                'input_mask': torch.LongTensor(input_mask),
                'input_type_ids': torch.LongTensor(input_type_ids),
                'matching_label_id': torch.LongTensor(matching_label_id)
            }
        
    def convert2tokens(self, context, response):
        input_seg = [0]
        input_tok = ['[CLS]']
    
        tmp_tok = []
        tmp_seg = []
        for i, turn in enumerate(context):
            if i % 2 == 0:
                toks = self.tokenizer.tokenize(turn + ' [SEP]')
                tmp_tok.append(toks)
                tmp_seg.append([0 for _ in range(len(toks))])
            else:
                toks = self.tokenizer.tokenize(turn + ' [SEP]')
                tmp_tok.append(toks)
                tmp_seg.append([1 for _ in range(len(toks))])
        
        
        while sum(map(len, tmp_tok)) > self.max_history - 1:
            tmp_tok.pop(0)
            tmp_seg.pop(0)
            tmp_tok.pop(0)
            tmp_seg.pop(0)
    
        for tok in tmp_tok:
            input_tok.extend(tok)
        for seg in tmp_seg:
            input_seg.extend(seg)

        input_tok.pop()
        input_seg.pop()
    
        toks = self.tokenizer.tokenize(response)
        toks = toks[:self.max_utter - 2]
        toks = ['[THIS]'] + toks + ['[THIS]']
        input_tok.extend(toks)
        input_seg.extend([1 for _ in range(len(toks))])
        input_id = self.tokenizer.convert_tokens_to_ids(input_tok)
    
        return input_tok, input_id, input_seg
     
        

    def convert_examples_to_features(self, batch_examples):
        batch_size = len(batch_examples)
        max_seq_len = max([len(json.loads(exam['input_tok'])) for exam in batch_examples])
        matching_label_id = []
        guids = []
        input_ids = []
        input_masks = []
        input_type_ids = []

        for example in batch_examples:
            guids.append(example['guid'])
            input_id = json.loads(example['input_id'])
            input_seg = json.loads(example['input_seg'])
            
            if max_seq_len > self.max_history+self.max_utter:
                raise ValueError('long sentence ...')
            matching_label_id.append(example['label'])
            input_id = input_id + [0 for _ in range(max_seq_len-len(input_id))]
            assert len(input_id) == max_seq_len
            input_ids.append(copy.deepcopy(input_id))
            input_mask = [1 for _ in range(len(input_id))] + [0 for _ in range(max_seq_len-len(input_id))]
            assert len(input_mask) == max_seq_len
            input_masks.append(copy.deepcopy(input_mask))
            input_type_id = input_seg + [0 for _ in range(max_seq_len - len(input_seg))]
            assert len(input_type_id) == max_seq_len
            input_type_ids.append(copy.deepcopy(input_type_id))

        # assert len(input_ids[0]) == max_seq_len*batch_size
        # assert len(input_masks[0]) == max_seq_len*batch_size
        # assert len(input_masks[0]) == max_seq_len*batch_size

        return input_ids, input_masks, input_type_ids, matching_label_id, guids


if __name__ == '__main__':
    from transformers import BertTokenizer as tokenizer_class
    import numpy as np
    
    from transformers import BertConfig
    
    directory = '/Users/daiyp/Documents/projects/DialogManagerProject/saved_models/original_sbert'
    tokenizer = tokenizer_class.from_pretrained(directory)
    tokenizer.add_tokens(['[THIS]'])
    
    
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--scenario", default=None, type=str, required=True,
                        help="Name of the dataset, e.g. 交通-山东ETC")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--cache_dir", default="./cache_dir", type=str,
                        help="The directory where cache features save")
    parser.add_argument("--few_shot", default=0, type=int,
                        help="few shot number, 0 means using all training data")

    # Other parameters
    parser.add_argument("--max_history", default=250, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--max_utter", default=50, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training. evaluate on dev")
    parser.add_argument("--do_eval", action='store_true',
                        help="eval on dataset.")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=20, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_epochs', type=int, default=0.5,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--use_sys_act', action='store_true')

    args = parser.parse_args()
    
    dp = DataProcessor(tokenizer, args)
    
    args.class_types = dp.intent_dic