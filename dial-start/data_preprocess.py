import os
import re
import json
import torch
import random
import pickle
import IPython
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from collections import defaultdict, Counter
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel, set_seed, BertForNextSentencePrediction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
set_seed(3407)
DATASET = {'doc':'doc2dial', '711':'dialseg711'}


def gen_text(args):
    data, topic_data = [], []
    w, k = 2, 5
    for dataset in ['dialseg711', 'doc2dial']:
        todolist = [f'{args.dataroot}/{dataset}/'+i for i in os.listdir(f'{args.dataroot}/{dataset}') if not i.startswith('.')]

        for i in tqdm(todolist):
            dial_name = i.split('/')[-1][:-4]
            cur_dials = open(i).read().split('\n')[:-1]
            dials = [utt for utt in cur_dials if '=======' not in utt]
            dial_len = len(dials)

            for utt_idx in range(dial_len-1):
                context, cur, neg, hard_neg = [], [], [], []
                neg_index = random.choice(list(range(utt_idx-w+1)) + list(range(utt_idx+w+1, dial_len)))  
                negdial = [i for i in open(random.choice(todolist)).read().split('\n')[:-1] if '====' not in i]
                neg_hard_index = random.choice(list(range(len(negdial))))

                mid = utt_idx+1
                l, r = utt_idx, utt_idx+1
                for i in range(args.history):
                    if l > -1:
                        context.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[l]))
                        l -= 1
                    if r < dial_len:
                        cur.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[r]))
                        r += 1
                    if neg_index < dial_len:
                        neg.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[neg_index]))
                        neg_index += 1
                    if neg_hard_index < len(negdial):
                        hard_neg.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', negdial[neg_hard_index]))
                        neg_hard_index += 1

                context.reverse()
                data.append([(context, cur), (context, neg), (context, hard_neg)])
                topic_data.append((dials, mid))

                assert len(dials) > mid

        json.dump(data, open(f'{args.dataroot}/{dataset}_{args.version}.json', 'w'))
        json.dump(topic_data, open(f'{args.dataroot}/{dataset}_topic_data.json', 'w'))


def main(args):
    MAX_LEN = 512
    for dataset in tqdm(['dialseg711', 'doc2dial']):
        data = json.load(open(f'{args.dataroot}/{dataset}_{args.version}.json'))[:50]
        topic_data = json.load(open(f'{args.dataroot}/{dataset}_topic_data.json'))[:50]

        turn_ids, id_inputs, topic_inputs, sample_num_memory, topic_train, topic_num = [], [], [], [len(i) for i in data], [], []
        for i in tqdm(range(len(data))):
            for sample in data[i]:
                context, cur = sample
                if args.history == 2:
                    sent1 = sent2 = ''
                    for sen in context:
                        sent1 += sen + '[SEP]'
                else:
                    sent1 = context[-1]

                sent2 = cur[0]

                encoded_sent1 = tokenizer.encode(sent1, add_special_tokens = True, return_tensors = 'pt')
                encoded_sent2 = tokenizer.encode(sent2, truncation=True, max_length = 256, add_special_tokens = True, return_tensors = 'pt')

                topic_con = tokenizer(context, truncation=True, max_length = 256)
                topic_cur = tokenizer(cur, truncation=True, max_length = 256)
                if args.history == 2:
                    id_input = encoded_sent1[0].tolist()[-257:-1] + encoded_sent2[0].tolist()[1:]  
                    turn_id = [0] * len(encoded_sent1[0].tolist()[-257:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                else:
                    id_input = encoded_sent1[0].tolist()[-257:] + encoded_sent2[0].tolist()[1:]
                    turn_id = [0] * len(encoded_sent1[0].tolist()[-257:]) + [1] * len(encoded_sent2[0].tolist()[1:])

                id_inputs.append(torch.Tensor(id_input))
                topic_inputs.append((topic_con, topic_cur, len(context), len(cur)))
                turn_ids.append(torch.tensor(turn_id))

            topic_train.append(tokenizer(topic_data[i][0], truncation=True, max_length = 512, padding=True, return_tensors='pt'))
            topic_num.append((len(topic_data[i][0]), topic_data[i][1])) 

        id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        turn_ids = pad_sequences(turn_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")

        topic_train_input = [i['input_ids'] for i in topic_train]
        topic_train_mask = [i['attention_mask'] for i in topic_train]
        attention_masks = []
        for sent in tqdm(id_inputs):
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        grouped_inputs, grouped_masks, grouped_topic, grouped_token_type_id = [], [], [], []
        token_type_id = []
        count = 0
        for i in tqdm(sample_num_memory):
            grouped_inputs.append(id_inputs[count: count+i])
            grouped_masks.append(attention_masks[count: count+i])
            grouped_topic.append(topic_inputs[count: count+i])
            grouped_token_type_id.append(turn_ids[count:count+i])
            count += i

        pos_neg_pairs, pos_neg_masks, pos_neg_token_types, topic_pairs = [], [], [], []
        topic_trains, topic_trains_mask, topic_nums = [], [], []
        for i in tqdm(range(len(grouped_inputs))):
            if len(grouped_inputs[i]) == 2:
                pos_neg_pairs.append(grouped_inputs[i])
                pos_neg_token_types.append(grouped_token_type_id[i])
                pos_neg_masks.append(grouped_masks[i])
            else:
                topic_pairs.append([grouped_topic[i][0], grouped_topic[i][1]])
                topic_pairs.append([grouped_topic[i][0], grouped_topic[i][2]])
                
                topic_trains.append(topic_train_input[i])
                topic_trains.append(topic_train_input[i])
                topic_trains_mask.append(topic_train_mask[i])
                topic_trains_mask.append(topic_train_mask[i])
                topic_nums.append(topic_num[i])
                topic_nums.append(topic_num[i])

                pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][1]])
                pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][2]])

                pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][1]])
                pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][2]])

                pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][1]])
                pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][2]])

        train_inputs, train_masks, train_types = torch.tensor(pos_neg_pairs), torch.tensor(pos_neg_masks), torch.tensor(pos_neg_token_types)
        pickle.dump((train_inputs, train_masks, train_types, topic_pairs, topic_trains, topic_trains_mask, topic_nums), 
                    open(f'{args.dataroot}/{dataset}{args.save_name}.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", default='', help='The name of preprocessed data')
    parser.add_argument("--dataroot", default='./data')
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--version", default='2h2cur')
    
    args = parser.parse_args()

    gen_text(args)
    main(args)