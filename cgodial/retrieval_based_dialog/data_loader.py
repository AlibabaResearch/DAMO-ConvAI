import json
import os
import random
import copy
import torch
import pprint
from collections import Counter


class DataLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_history = config.max_history
        self.max_response = config.max_response
        
        with open('train.json') as f:
            self.train = json.load(f)
        with open('dev.json') as f:
            self.dev = json.load(f)
        with open('test.json') as f:
            self.test = json.load(f)
            
        if os.path.exists('train_examples.json'):
            with open('train_examples.json') as f:
                self.train_examples = json.load(f)
        else:
            self.train_examples = self.convert_dialogs_to_examples(self.train)
            with open('train_examples.json', 'w') as f:
                json.dump(self.train_examples, f, ensure_ascii=False, indent=1)

        if os.path.exists('dev_examples.json'):
            with open('dev_examples.json') as f:
                self.dev_examples = json.load(f)
        else:
            self.dev_examples = self.convert_dialogs_to_examples(self.dev)
            with open('dev_examples.json', 'w') as f:
                json.dump(self.dev_examples, f, ensure_ascii=False, indent=1)

        if os.path.exists('test_examples.json'):
            with open('test_examples.json') as f:
                self.test_examples = json.load(f)
        else:
            self.test_examples = self.convert_dialogs_to_examples(self.test)
            with open('test_examples.json', 'w') as f:
                json.dump(self.test_examples, f, ensure_ascii=False, indent=1)

        print('training examples:', len(self.train_examples))
        print('dev examples:', len(self.dev_examples))
        print('test examples:', len(self.test_examples))
    
    
    def get_batch_iterator(self, exmaples, batch_size=16):
        bid = 0
        while bid<len(exmaples):
            # print('bid:', bid, bid+batch_size)
            batch_examples = exmaples[bid:bid+batch_size]
            input_ids, input_mask, input_type_ids, matching_label_id \
                = self.convert_examples_to_features(batch_examples)
            bid += batch_size
            yield {
                'input_ids': torch.LongTensor(input_ids),
                'input_mask': torch.LongTensor(input_mask),
                'input_type_ids': torch.LongTensor(input_type_ids),
                'matching_label_id': torch.LongTensor(matching_label_id)
            }
    
    def convert_dialogs_to_examples(self, dialogs):
        examples = []
        lens = []
        for dial_id, dialog in dialogs.items():
            history = dialog['history']
            true_ans = dialog['true']
            false_ans = dialog['false']
            for true_a in true_ans:
                input_tok, input_id, input_seg = self.convert2tokens(history, true_a)
                data = {
                        'dial_id': dial_id,
                        'history': history,
                        'response': true_a,
                        'label': 1,
                        'input_tok': input_tok,
                        'input_id': input_id,
                        'input_seg': input_seg
                    }
                # print(history)
                # print(input_tok)
                # print(input_id)
                # print(input_seg)
                examples.append(data)
                
                
            for false_a in false_ans:
                input_tok, input_id, input_seg = self.convert2tokens(history, false_a)
                data = {
                        'dial_id': dial_id,
                        'history': history,
                        'response': false_a,
                        'label': 0,
                        'input_tok': input_tok,
                        'input_id': input_id,
                        'input_seg': input_seg
                    }
                # print(history)
                # print(input_tok)
                # print(input_id)
                # print(input_seg)
                examples.append(data)
        # print(Counter(lens).most_common())
        return examples
    
    def convert2tokens(self, context, response):
        input_seg = [0]
        input_tok = ['[CLS]']
        
        tmp_tok = []
        tmp_seg = []
        for i, turn in enumerate(context):
            if i %2  == 0:
                toks = self.tokenizer.tokenize(turn + ' [SEP]')
                tmp_tok.append(toks)
                tmp_seg.append([0 for _ in range(len(toks))])
            else:
                toks = self.tokenizer.tokenize(turn + ' [SEP]')
                tmp_tok.append(toks)
                tmp_seg.append([1 for _ in range(len(toks))])
        
        while sum(map(len, tmp_tok))> self.max_history-1:
            tmp_tok.pop(0)
            tmp_seg.pop(0)
            tmp_tok.pop(0)
            tmp_seg.pop(0)
        
        for tok in tmp_tok:
            input_tok.extend(tok)
        for seg in tmp_seg:
            input_seg.extend(seg)
        
        toks = self.tokenizer.tokenize(response)
        toks = toks[:self.max_response-2]
        toks = ['[RSP]'] + toks + ['[SEP]']
        input_tok.extend(toks)
        input_seg.extend([1 for _ in range(len(toks))])
        input_id = tokenizer.convert_tokens_to_ids(input_tok)
        
        return input_tok, input_id, input_seg
            
            
    
    def convert_examples_to_features(self, batch_examples):
        batch_size = len(batch_examples)
        max_seq_len = max([len(exam['input_tok']) for exam in batch_examples])
        matching_label_id = []
        input_ids = []
        input_masks = []
        input_type_ids = []

        for example in batch_examples:
            if max_seq_len > self.max_history+self.max_response:
                print(example)
            matching_label_id.append(example['label'])
            input_id = example['input_id'] + [0 for _ in range(max_seq_len-len(example['input_id']))]
            assert len(input_id) == max_seq_len
            input_ids.append(copy.deepcopy(input_id))
            input_mask = [1 for _ in range(len(example['input_id']))] + \
                [0 for _ in range(max_seq_len-len(example['input_id']))]
            assert len(input_mask) == max_seq_len
            input_masks.append(copy.deepcopy(input_mask))
            input_type_id = example['input_seg'] + [0 for _ in range(max_seq_len - len(example['input_id']))]
            assert len(input_type_id) == max_seq_len
            input_type_ids.append(copy.deepcopy(input_type_id))

        # assert len(input_ids[0]) == max_seq_len*batch_size
        # assert len(input_masks[0]) == max_seq_len*batch_size
        # assert len(input_masks[0]) == max_seq_len*batch_size

        return input_ids, input_masks, input_type_ids, matching_label_id
       


if __name__ == '__main__':
    from transformers import BertTokenizer as tokenizer_class
    from model import BertForMatching
    from config import Config
    import numpy as np

    from transformers import BertConfig
    directory = '/Users/daiyp/Documents/projects/struct-bert-base-chinese-single/single-176k-512'
    tokenizer = tokenizer_class.from_pretrained(directory)
    tokenizer.add_tokens(['[RSP]'])
    
    model = BertForMatching.from_pretrained(directory)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    dataloader = DataLoader(Config(), tokenizer)
    train_dataset = dataloader.get_batch_iterator(dataloader.train_examples, 4)


    for item in train_dataset:
        pass
        
    #     loss, matching_logits = model(**item)
    #     print(loss)
    #     print(matching_logits)
    #     print(input())
    #
        # pprint.pprint(item, width=300)
        # print(input())

        

