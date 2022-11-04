#!/usr/bin/python
# _*_coding:utf-8_*_
import os
import codecs
import math
from multiprocessing import Pool

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from transformers import AutoTokenizer, AutoConfig

import config
from model.plato.configuration_plato import PlatoConfig


def line_statistics(file_name):
    """
    统计文件行数
    """
    if file_name is None:
        return 0

    content = os.popen("wc -l %s" % file_name)
    line_number = int(content.read().split(" ")[0])
    return line_number


class BertExample():
    def __init__(self, guid, role, text_a, text_b=None, label=None):
        self.guid = guid
        self.role = role
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertFeatures():
    def __init__(self, input_ids, input_mask, segment_ids, role_ids, label_id, turn_ids=None, position_ids=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.role_ids = role_ids
        self.turn_ids = turn_ids
        self.position_ids = position_ids
        self.label_id = label_id
        self.guid = guid

        self.batch_size = len(self.input_ids)

    def show_case(self):
        print('*' * 20 + 'input_ids' + '*' * 20)
        print(self.input_ids) if self.batch_size == 1 else print(self.input_ids[0])
        print('*' * 20 + 'input_mask' + '*' * 20)
        print(self.input_mask) if self.batch_size == 1 else print(self.input_mask[0])
        print('*' * 20 + 'segment_ids' + '*' * 20)
        print(self.segment_ids) if self.batch_size == 1 else print(self.segment_ids[0])
        print('*' * 20 + 'role_ids' + '*' * 20)
        print(self.role_ids) if self.batch_size == 1 else print(self.role_ids[0])
        print('*' * 20 + 'turn_ids' + '*' * 20)
        print(self.turn_ids) if self.batch_size == 1 else print(self.turn_ids[0])
        print('*' * 20 + 'position_ids' + '*' * 20)
        print(self.position_ids) if self.batch_size == 1 else print(self.position_ids[0])
        print('*' * 20 + 'label_id' + '*' * 20)
        print(self.label_id) if self.batch_size == 1 else print(self.label_id[0])


class DataProvider():
    """
    dial2Vec数据提供类
    """
    def __init__(self, args):
        self.tokenizer = None
        self.train_loader = None
        self.num_train_examples = None
        self.clustering_test_loader = None
        self.clustering_dev_loader = None
        self.tss_test_loader = None
        self.num_workers = 20

        self.args = args
        self.logger = args.logger

    def init_data_socket(self):
        """
        初始化数据接口
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.huggingface_mapper[self.args.backbone])

        if self.args.backbone in ['t5']:
            self.tokenizer.cls_token = ''
            self.tokenizer.sep_token = self.tokenizer.eos_token

        if self.args.backbone.lower() == 'plato':
            self.tokenizer_config = PlatoConfig.from_json_file(self.args.config_file)
        elif self.args.backbone.lower() in ['bert', 'roberta', 'todbert', 't5', 'blender', 'unsup_simcse', 'sup_simcse', 'dialoguecse']:
            self.tokenizer_config = AutoConfig.from_pretrained(config.huggingface_mapper[self.args.backbone.lower()])
        else:
            raise NameError('Unknown backbone model: [%s]' % self.args.backbone)

        if self.args.backbone.lower() in ['roberta']:
            self.tokenizer_config.max_seq_length = self.tokenizer_config.max_position_embeddings - 2
        elif self.args.backbone.lower() in ['t5']:
            self.tokenizer_config.max_seq_length = 512
        else:
            self.tokenizer_config.max_seq_length = self.tokenizer_config.max_position_embeddings

        self.labels_list = ["0", "1"]

    def get_tokenizer(self):
        """
        获取分词器
        """
        return self.tokenizer

    def get_labels(self):
        """
        查看标签数量
        """
        return self.labels_list

    def peek_num_train_examples(self):
        """
        查看具有多少训练数据
        """
        self.num_train_examples = line_statistics.line_statistics(self.args.data_dir + "/train.tsv")
        return self.num_train_examples

    def load_data(self, data_file):
        """
        读取数据
        """
        with codecs.open(data_file, "r", "utf-8") as f_in:
            bert_examples = []
            for line in f_in:
                line_array = [s.strip() for s in line.split(config.line_sep_token) if s.strip()]

                role, session, label = line_array[0], line_array[1], line_array[2]
                bert_examples.append(BertExample(guid=None, role=role, text_a=session, label=label))
        return bert_examples

    def load_data_for_simcse(self, data_file):
        """
        为了SimCSE，分句子读取数据。
        :param data_file:
        :return:
        """
        with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
            bert_examples = []
            for index, line in enumerate(f_in):
                line_array = [s.strip() for s in line.split(config.line_sep_token) if s.strip()]

                role, session, label = line_array[0], line_array[1], line_array[2]
                samples = [context.split(config.turn_sep_token) for context in session.split(config.sample_sep_token)]
                for i, r in enumerate(role):
                    ts = []
                    for j in range(len(samples)):
                        ts.append(samples[j][i])                    # 正负样本中对应role的utterance的列表
                    bert_examples.append(BertExample(guid=index,
                                                     role=r,
                                                     text_a=config.sample_sep_token.join(ts),
                                                     label=label))
        return bert_examples

    def convert_examples_worker(self, worker_index, start_index, end_index, examples):
        """
        将examples转换为bert_features的工作线程
        """
        if self.args.backbone in ['bert', 'roberta', 't5', 'blender', 'unsup_simcse', 'sup_simcse']:
            return self.__convert_examples_worker_for_bert(worker_index, start_index, end_index, examples)
        elif self.args.backbone == 'plato':
            return self.__convert_examples_worker_for_plato(worker_index, start_index, end_index, examples)
        elif self.args.backbone == 'todbert':
            return self.__convert_examples_worker_for_todbert(worker_index, start_index, end_index, examples)
        else:
            raise ValueError('Unknown backbone name: [%s]' % self.args.backbone)

    def __convert_examples_worker_for_bert(self, worker_index, start_index, end_index, examples):
        """
        将examples转换为bert_features的工作线程
        """
        features = []
        self.logger.debug("converting_examples, worker_index: %s start: %s end: %s" % (worker_index, start_index, end_index))

        tokenizer = self.get_tokenizer()
        start_token, sep_token, pad_token_id = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token_id

        for data_index, example in enumerate(examples):
            if data_index < start_index or data_index >= end_index:
                continue

            sample_list = example.text_a.split(config.sample_sep_token)
            role_list = [int(r) for r in example.role.split(config.sample_sep_token)] \
                if example.role.find(config.turn_sep_token) != -1 \
                else [int(r) for r in example.role]

            sample_input_ids = []
            sample_segment_ids = []
            sample_role_ids = []
            sample_input_mask = []
            sample_turn_ids = []
            sample_position_ids = []

            for t, s in enumerate(sample_list):
                text_tokens = []
                text_turn_ids = []
                text_role_ids = []

                text_list = s.split(config.turn_sep_token)

                # bert-token:     [cls]  token   [sep]  token
                # roberta-token:   <s>   token   </s>   </s> token
                # t5-token:       token  </s>    token
                # dialogpt-token: token  token   token  token
                # blender-token:  token  </s>    token  </s>
                # segment:          0      0       0      0    0
                # pos:              0      1       2      3    4
                if self.args.backbone in ['bert', 'roberta', 'unsup_simcse', 'sup_simcse']:
                    text_tokens.append(start_token)
                    text_turn_ids.append(0)
                    text_role_ids.append(role_list[0])

                for i, text in enumerate(text_list):
                    tokenized_tokens = self.tokenizer.tokenize(text)
                    text_tokens.extend(tokenized_tokens)
                    text_turn_ids.extend([i] * len(tokenized_tokens))
                    text_role_ids.extend([role_list[i]] * len(tokenized_tokens))

                    if self.args.use_sep_token:
                        if i != (len(text_list) - 1):
                            text_tokens.append(sep_token)
                            text_turn_ids.append(i)
                            text_role_ids.append(role_list[i])
                            if self.args.backbone in ['roberta']:
                                text_tokens.append(sep_token)
                                text_role_ids.append(role_list[i])

                if self.args.backbone in ['t5', 'blender']:
                    text_tokens.append(sep_token)
                    text_turn_ids.append(i)
                    text_role_ids.append(role_list[i])

                text_tokens = text_tokens[:self.tokenizer_config.max_seq_length]
                text_turn_ids = text_turn_ids[:self.tokenizer_config.max_seq_length]
                text_role_ids = text_role_ids[:self.tokenizer_config.max_seq_length]

                text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)

                text_input_ids += [pad_token_id] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_input_mask = [1] * len(text_tokens) + [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_segment_ids = [0] * self.tokenizer_config.max_seq_length
                text_position_ids = list(range(len(text_tokens))) + [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_turn_ids += [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_role_ids += [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))

                assert len(text_input_ids) == self.tokenizer_config.max_seq_length
                assert len(text_input_mask) == self.tokenizer_config.max_seq_length
                assert len(text_segment_ids) == self.tokenizer_config.max_seq_length
                assert len(text_position_ids) == self.tokenizer_config.max_seq_length
                assert len(text_turn_ids) == self.tokenizer_config.max_seq_length
                assert len(text_role_ids) == self.tokenizer_config.max_seq_length

                sample_input_ids.append(text_input_ids)
                sample_turn_ids.append(text_turn_ids)
                sample_role_ids.append(text_role_ids)
                sample_segment_ids.append(text_segment_ids)
                sample_position_ids.append(text_position_ids)
                sample_input_mask.append(text_input_mask)

            n_neg = 9
            label_id = [1] + [0] * n_neg
            bert_feature = BertFeatures(input_ids=sample_input_ids,
                                        input_mask=sample_input_mask,
                                        segment_ids=sample_segment_ids,
                                        role_ids=sample_role_ids,
                                        turn_ids=sample_turn_ids,
                                        position_ids=sample_position_ids,
                                        label_id=label_id,
                                        guid=[example.guid] * (1 + n_neg))

            features.append(bert_feature)
        return features

    def __convert_examples_worker_for_plato(self, worker_index, start_index, end_index, examples):
        """
        将examples转换为bert_features的工作线程
        """
        features = []
        self.logger.debug("converting_examples, worker_index: %s start: %s end: %s" % (worker_index, start_index, end_index))

        for data_index, example in enumerate(examples):
            if data_index < start_index or data_index >= end_index:
                continue

            sample_list = example.text_a.split(config.sample_sep_token)
            role_list = [int(r) for r in example.role.split(config.sample_sep_token)] \
                if example.role.find(config.turn_sep_token) != -1 \
                else [int(r) for r in example.role]

            sample_input_ids = []
            sample_segment_ids = []
            sample_role_ids = []
            sample_input_mask = []
            sample_turn_ids = []
            sample_position_ids = []

            for t, s in enumerate(sample_list):
                text_tokens = []
                text_turn_ids = []
                text_role_ids = []
                text_segment_ids = []

                text_list = s.split(config.turn_sep_token)

                # token: token [eou] token [eou] [bos] token [eos]
                # role:   0     0     1     1     0     0      0
                # turn:   2     2     1     1     0     0      0
                # pos:    0     1     0     1     0     1      2
                bou, eou, bos, eos = "[unused0]", "[unused1]", "[unused0]", "[unused1]"

                # use [CLS] as the latent variable of PLATO
                # text_list[0] = self.args.start_token + ' ' + text_list[0]

                if self.args.use_response == True:   # specify the context and response
                    context, response = text_list[:-1], text_list[-1]
                    word_list = self.tokenizer.tokenize(response)
                    uttr_len = len(word_list)

                    start_token, end_token = bou, eou

                    role_id, turn_id = role_list[-1], 0

                    response_tokens = [start_token] + word_list + [end_token]
                    response_role_ids = [role_id] * (1 + uttr_len + 1)
                    response_turn_ids = [turn_id] * (1 + uttr_len + 1)
                    response_segment_ids = [0] * (1 + uttr_len + 1)                   # not use

                else:
                    context = text_list
                    response_tokens, response_role_ids, response_turn_ids, response_segment_ids = [], [], [], []

                # limit the context length
                context = context[-self.args.max_context_length:]

                for i, text in enumerate(context):
                    word_list = self.tokenizer.tokenize(text)
                    uttr_len = len(word_list)

                    end_token = eou

                    role_id, turn_id = role_list[i], len(context) - i

                    text_tokens.extend(word_list + [end_token])
                    text_role_ids.extend([role_id] * (uttr_len + 1))
                    text_turn_ids.extend([turn_id] * (uttr_len + 1))
                    text_segment_ids.extend([0] * (uttr_len + 1))

                text_tokens.extend(response_tokens)
                text_role_ids.extend(response_role_ids)
                text_turn_ids.extend(response_turn_ids)
                text_segment_ids.extend(response_segment_ids)

                if len(text_tokens) > self.tokenizer_config.max_seq_length:
                    text_tokens = text_tokens[:self.tokenizer_config.max_seq_length]
                    text_turn_ids = text_turn_ids[:self.tokenizer_config.max_seq_length]
                    text_role_ids = text_role_ids[:self.tokenizer_config.max_seq_length]
                    text_segment_ids = text_segment_ids[:self.tokenizer_config.max_seq_length]

                assert (max(text_turn_ids) <= self.args.max_context_length)

                # 制作text_position_id序列
                text_position_ids = []
                text_position_id = 0
                for i, turn_id in enumerate(text_turn_ids):
                    if i != 0 and turn_id < text_turn_ids[i - 1]:   # PLATO
                        text_position_id = 0
                    text_position_ids.append(text_position_id)
                    text_position_id += 1

                # max_turn_id = max(text_turn_ids)
                # text_turn_ids = [max_turn_id - t for t in text_turn_ids]

                text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
                text_input_mask = [1] * len(text_input_ids)

                # Zero-pad up to the sequence length.
                while len(text_input_ids) < self.tokenizer_config.max_seq_length:
                    text_input_ids.append(0)
                    text_turn_ids.append(0)
                    text_role_ids.append(0)
                    text_segment_ids.append(0)
                    text_position_ids.append(0)
                    text_input_mask.append(0)

                assert len(text_input_ids) == self.tokenizer_config.max_seq_length
                assert len(text_turn_ids) == self.tokenizer_config.max_seq_length
                assert len(text_role_ids) == self.tokenizer_config.max_seq_length
                assert len(text_segment_ids) == self.tokenizer_config.max_seq_length
                assert len(text_position_ids) == self.tokenizer_config.max_seq_length
                assert len(text_input_mask) == self.tokenizer_config.max_seq_length

                sample_input_ids.append(text_input_ids)
                sample_turn_ids.append(text_turn_ids)
                sample_role_ids.append(text_role_ids)
                sample_segment_ids.append(text_segment_ids)
                sample_position_ids.append(text_position_ids)
                sample_input_mask.append(text_input_mask)

            label_id = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            bert_feature = BertFeatures(input_ids=sample_input_ids,
                                        input_mask=sample_input_mask,
                                        segment_ids=sample_segment_ids,
                                        role_ids=sample_role_ids,
                                        turn_ids=sample_turn_ids,
                                        position_ids=sample_position_ids,
                                        label_id=label_id)

            features.append(bert_feature)
        return features

    def __convert_examples_worker_for_todbert(self, worker_index, start_index, end_index, examples):
        """
        将examples转换为bert_features的工作线程
        """
        features = []
        self.logger.debug("converting_examples, worker_index: %s start: %s end: %s" % (worker_index, start_index, end_index))

        tokenizer = self.get_tokenizer()
        start_token, sep_token, pad_token_id = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token_id

        for data_index, example in enumerate(examples):
            if data_index < start_index or data_index >= end_index:
                continue

            sample_list = example.text_a.split(config.sample_sep_token)
            role_list = [int(r) for r in example.role.split(config.sample_sep_token)] \
                if example.role.find(config.turn_sep_token) != -1 \
                else [int(r) for r in example.role]

            sample_input_ids = []
            sample_segment_ids = []
            sample_role_ids = []
            sample_input_mask = []
            sample_turn_ids = []
            sample_position_ids = []

            for t, s in enumerate(sample_list):
                text_tokens = []
                text_turn_ids = []
                text_role_ids = []

                text_list = s.split(config.turn_sep_token)

                # token:    [CLS]  [SYS]  token  [USR]  token
                # segment:    0      0      0      0      0
                # pos:        0      1      2      3      4
                text_tokens.append(start_token)
                text_turn_ids.append(0)
                text_role_ids.append(role_list[0])

                for i, text in enumerate(text_list):
                    text_tokens.append('[sys]') if role_list[i] == 0 else text_tokens.append('[usr]')
                    text_turn_ids.append(i)
                    text_role_ids.append(role_list[i])

                    tokenized_tokens = self.tokenizer.tokenize(text)
                    text_tokens.extend(tokenized_tokens)
                    text_turn_ids.extend([i] * len(tokenized_tokens))
                    text_role_ids.extend([role_list[i]] * len(tokenized_tokens))

                text_tokens.append(sep_token)
                text_turn_ids.append(i)
                text_role_ids.append(role_list[i])

                text_tokens = text_tokens[:self.tokenizer_config.max_seq_length]
                text_turn_ids = text_turn_ids[:self.tokenizer_config.max_seq_length]
                text_role_ids = text_role_ids[:self.tokenizer_config.max_seq_length]

                text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)

                text_input_ids += [pad_token_id] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_input_mask = [1] * len(text_tokens) + [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_segment_ids = [0] * self.tokenizer_config.max_seq_length
                text_position_ids = list(range(len(text_tokens))) + [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_turn_ids += [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))
                text_role_ids += [0] * (self.tokenizer_config.max_seq_length - len(text_tokens))

                assert len(text_input_ids) == self.tokenizer_config.max_seq_length
                assert len(text_input_mask) == self.tokenizer_config.max_seq_length
                assert len(text_segment_ids) == self.tokenizer_config.max_seq_length
                assert len(text_position_ids) == self.tokenizer_config.max_seq_length
                assert len(text_turn_ids) == self.tokenizer_config.max_seq_length
                assert len(text_role_ids) == self.tokenizer_config.max_seq_length

                sample_input_ids.append(text_input_ids)
                sample_turn_ids.append(text_turn_ids)
                sample_role_ids.append(text_role_ids)
                sample_segment_ids.append(text_segment_ids)
                sample_position_ids.append(text_position_ids)
                sample_input_mask.append(text_input_mask)

            label_id = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            bert_feature = BertFeatures(input_ids=sample_input_ids,
                                        input_mask=sample_input_mask,
                                        segment_ids=sample_segment_ids,
                                        role_ids=sample_role_ids,
                                        turn_ids=sample_turn_ids,
                                        position_ids=sample_position_ids,
                                        label_id=label_id)

            features.append(bert_feature)
        return features

    def convert_examples_to_features(self, examples):
        """
        将examples转换为bert_features
        """
        worker_results = []
        features = []
        num_workers = self.num_workers

        pool = Pool(processes=num_workers)
        partition_size = math.ceil(len(examples) / num_workers)

        for i in range(num_workers):
            start = i * partition_size
            end = min((i + 1) * partition_size, len(examples))
            worker_results.append(pool.apply_async(self.convert_examples_worker, args=(i, start, end, examples)))
            if end == len(examples):
                break

        for processor in worker_results:
            feature_list = processor.get()
            features.extend(feature_list)

        pool.close()
        pool.join()
        return features

    def get_train_loader(self):
        """
        读取训练数据
        """
        if self.train_loader is not None:
            return self.train_loader

        bert_examples = self.load_data(self.args.data_dir + "/train.tsv")
        bert_features = self.convert_examples_to_features(bert_examples)
        self.num_train_steps = int(len(bert_examples) / self.args.train_batch_size * self.args.num_train_epochs)

        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", self.num_train_examples)
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        self.logger.info("  Num steps = %d", self.num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in bert_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in bert_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in bert_features], dtype=torch.long)
        all_role_ids = torch.tensor([f.role_ids for f in bert_features], dtype=torch.long)
        all_turn_ids = torch.tensor([f.turn_ids for f in bert_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in bert_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in bert_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids,
                                   all_input_mask,
                                   all_segment_ids,
                                   all_role_ids,
                                   all_turn_ids,
                                   all_position_ids,
                                   all_label_ids)

        # train_sampler = SequentialSampler(train_data)
        train_sampler = RandomSampler(train_data) if self.args.local_rank == -1 else DistributedSampler(train_data)
        self.train_loader = DataLoader(train_data,
                                       sampler=train_sampler,
                                       batch_size=self.args.train_batch_size)
        return self.train_loader

    def get_clustering_test_loader(self, mode='test', level='dialogue'):
        """
        读取聚类测试数据
        :param mode:     test/dev
        :param level:    dialogue/sentence
        """
        if level == 'dialogue':
            if mode == 'test' and self.clustering_test_loader is not None:
                return self.clustering_test_loader
            if mode == 'dev' and self.clustering_dev_loader is not None:
                return self.clustering_dev_loader

            bert_examples = self.load_data(self.args.data_dir + "/clustering_%s.tsv" % mode)
        else:
            bert_examples = self.load_data_for_simcse(self.args.data_dir + "/clustering_%s.tsv" % mode)
        bert_features = self.convert_examples_to_features(bert_examples)

        all_input_ids = torch.tensor([f.input_ids for f in bert_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in bert_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in bert_features], dtype=torch.long)
        all_role_ids = torch.tensor([f.role_ids for f in bert_features], dtype=torch.long)
        all_turn_ids = torch.tensor([f.turn_ids for f in bert_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in bert_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in bert_features], dtype=torch.long)

        if level == 'dialogue':
            test_data = TensorDataset(all_input_ids,
                                      all_input_mask,
                                      all_segment_ids,
                                      all_role_ids,
                                      all_turn_ids,
                                      all_position_ids,
                                      all_label_ids)
        else:
            all_guids = torch.tensor([f.guid for f in bert_features], dtype=torch.int)
            test_data = TensorDataset(all_input_ids,
                                      all_input_mask,
                                      all_segment_ids,
                                      all_role_ids,
                                      all_turn_ids,
                                      all_position_ids,
                                      all_label_ids,
                                      all_guids)

        test_sampler = SequentialSampler(test_data)
        if mode == 'test':
            self.clustering_test_loader = DataLoader(test_data,
                                                     sampler=test_sampler,
                                                     batch_size=self.args.test_batch_size)
            return self.clustering_test_loader
        elif mode == 'dev':
            self.clustering_dev_loader = DataLoader(test_data,
                                                     sampler=test_sampler,
                                                     batch_size=self.args.dev_batch_size)
            return self.clustering_dev_loader
        else:
            raise ValueError('Unknown dataset mode: [%s]' % mode)


def main():
    """
    主执行函数
    """
    data_provider = DataProvider(args=None)
    data_provider.init_data_socket()
    for instance in data_provider.get_train_loader():
        print(instance)


if __name__ == "__main__":
    main()
