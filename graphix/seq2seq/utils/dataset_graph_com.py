import os
import torch
from torch.utils.data import Dataset
import pdb
import re


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, data_training_args, training_args, tokenizer, seq2seq_dataset=None, graph_pedia=None, ):
        
        self.args = data_training_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset
        self.graph_pedia = graph_pedia

        self.conv_sep = " || "
    
    def map_alias(self, example):
        alias_map = {}
        example_list = example.split(' ')
        for i, ex in enumerate(example_list):
            if ex in ['as', 'AS']:
                alias_map[example_list[i + 1]] = example_list[i - 1]
        return alias_map

    def replace_alias(self, example, mapping):
        ex = example
        for k, v in mapping.items():
            ex = ex.replace(k, v)
            if 'as' in example:
                ex = ex.replace(' as ' + v, '')
            elif 'AS' in example:
                ex = ex.replace(' AS ' + v, '')

        return ex

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        question_in = " ".join(self.graph_pedia[index]['raw_question_toks'])
        struct_in_norm = re.sub('  +', ' ', self.graph_pedia[index]['new_struct_in'])
        seq_in = "{} ; {}".format(question_in, struct_in_norm)

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            max_length=self.args.max_source_length,
        )
        # remove alias of sqls:
        # pdb.set_trace()
        seq_out = self.graph_pedia[index]['seq_out']
        alias_map = self.map_alias(seq_out)
        sql_norm = self.replace_alias(seq_out, alias_map)
        # sql_norm_db_id = sql_norm_db_id = "{} | {}".format(raw_item['db_id'], sql_norm)

        tokenized_inferred = self.tokenizer(
            # sql_norm_db_id,
            sql_norm,
            max_length=self.args.max_target_length,
        )
        tokenized_question_and_schemas_input_ids = tokenized_question_and_schemas.data["input_ids"]
        tokenized_question_and_schemas_attention_mask = tokenized_question_and_schemas.data["attention_mask"]
        tokenized_inferred_input_ids = tokenized_inferred.data["input_ids"]
        # # Here -100 will let the model not to compute the loss of the padding tokens.
        # tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        # tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # # Here -100 will let the model not to compute the loss of the padding tokens.
        # tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100
        item = {
            'input_ids': tokenized_question_and_schemas_input_ids,
            'attention_mask': tokenized_question_and_schemas_attention_mask,
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']
        
        item['graph_idx'] = index
            
        
        # assertion:
        if len([a for a in tokenized_question_and_schemas.input_ids if a > 1]) != self.graph_pedia[int(item['graph_idx'])]['graph'].number_of_nodes():
            print('index: {}'.format(index))
            print('len of tokens is {}'.format(len([a for a in tokenized_question_and_schemas.input_ids if a > 1])))
            print('len of nodes is {}'.format(self.graph_pedia[int(item['graph_idx'])]['graph'].number_of_nodes()))
        # assert len([a for a in tokenized_question_and_schemas.input_ids if a > 1]) == self.graph_pedia[int(item['graph_idx'])]['graph'].number_of_nodes()
        
        return item

    def __len__(self):
        return len(self.seq2seq_dataset)