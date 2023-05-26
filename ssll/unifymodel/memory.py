import torch
import numpy as np
from eda import *
import torch.nn as nn
from unifymodel.dataset import *

class Curr_Unlabel_Memory(object):
    def __init__(self,keys=None,values=None, questions=None,args=None):
        if keys is None:
            self.keys = []
            self.values = []
            self.questions = []
        else:
            self.keys = keys
            self.values = values
            self.questions = questions

        self.total_keys = len(self.keys)
        if args.unlabel_ratio == -1: # fix unlabel to 2k
            num_unlabel = 2000
        else:
            num_unlabel = args.num_label * args.unlabel_ratio
        self.memory_size = int(args.newmm_size * num_unlabel)
    
    def push(self, keys, values, questions, args=None):
        # print(f'Lengths: key {len(keys)}, values: {len(values)}, questions: {len(questions)}', flush=True)
        for key, value, ques in zip(keys, values, questions):
            if self.total_keys < self.memory_size:
                assert type(self.keys) == list
                # print(f'key {key}, keys {self.keys}', flush=True)
                if key not in self.keys:
                    self.keys.append(key.tolist())
                    self.values.append(value)
                    self.questions.append(ques)
            else:
                drop_idx = random.choice(range(self.memory_size))
                self.keys.pop(drop_idx)
                self.values.pop(drop_idx)
                self.questions.pop(drop_idx)
                self.keys.append(key.tolist())
                self.values.append(value)
                self.questions.append(ques)
            self.total_keys = len(self.keys) # Update the memory size.
                
def create_batch_to_augment_memory(old_task, old_memory, curr_memory, tokz=None, args=None):

    querys = torch.tensor(old_memory['keys'])
    questions = old_memory['questions']
    neighbors = get_neighbors(querys, task_name=None, K=args.back_kneighbors, memory=curr_memory, args=args, questions=questions)
    aug_unlabel_batch = create_batch_from_memory(neighbors, tokz, args=args, task_name=old_task)
    return aug_unlabel_batch


def cosine_similarity(v1, m2):
    # print(v1.shape, m2.shape)
    if len(m2.shape) == 1 and len(v1.shape) == 1:
        cos = nn.CosineSimilarity(dim=0)
    elif len(m2.shape)>1:
        v1 = v1.unsqueeze(0)
        cos = nn.CosineSimilarity(dim=1)
    else:
        print(f'v1 shape {v1.shape}, m2 shape {m2.shape}',flush=True)
    score = cos(v1, m2)
    return score

def get_sentence_embedding(model, batch, args=None):

    model.set_active_adapters(None)
    batch_size = batch['raw_id'].size()[0]
    input_tokens = batch['raw_id'].cuda()
    attn_masks = batch['raw_mask'].cuda()
    outputs = model.transformer.encoder(input_ids=input_tokens, attention_mask=attn_masks, return_dict=True)
    pooled_emb = outputs.last_hidden_state
    sen_emb = torch.mean(pooled_emb, dim=1) # (batch_size, 768)
    # print('sentence emb shape', sen_emb.shape, flush=True)
    # if args.diff_question:
    all_questions = []
    all_values = []
    all_keys = []
    for i in range(batch_size):
        sample_ques = batch['batch_text'][i]['question']
        sample_value = batch['batch_text'][i]['raw_text']
        all_questions.append(sample_ques)
        all_values.append(sample_value)
        # all_keys.append(sen_emb[i].tolist())
        all_keys.append(sen_emb[i,:])
    # return sen_emb, all_values, all_questions
    return all_keys, all_values, all_questions

def construct_memory(task_name, model, batch, memory=None, args=None):

    sen_embs, all_values, all_questions = get_sentence_embedding(model, batch, args=args)

    if memory is None:
        memory = {}
    if task_name not in memory:
        memory[task_name] = {}
        memory[task_name]['keys'] = []
        memory[task_name]['values'] = []
        memory[task_name]['questions'] = []

    batch_size = len(batch['batch_text'])
    for i in range(batch_size):  
        key = sen_embs[i].tolist()
        # value = batch['raw_id'][i] #
        value = batch['batch_text'][i]['raw_text'] # * Store the raw sentence as the value into the memory.
        memory[task_name]['keys'].append(key)
        memory[task_name]['values'].append(value)
        memory[task_name]['questions'].append(all_questions[i])

    return memory

def get_neighbors(querys, task_name=None, K=1, memory=None, args=None, questions=None):
    assert memory is not None

    if task_name is not None: #* forward augment, memory is old unlabel memory
        keys = memory[task_name]['keys']
        values = memory[task_name]['values']
        # questions = memory[task_name]['questions'] # forward augment needs current unlabel questions

    else: #* backward augment, memory is current_unlabel_memory
        keys = memory.keys
        values = memory.values
        # print(f'Number of the memory content is {len(keys)}',flush=True)
        assert questions is not None # should be old memory question
        # if args.diff_question and questions is None:
            # questions = memory.questions
            # questions = memory[task_name]['questions']

    keys_tensor = torch.tensor(keys).cuda()

    retrieved_samples = []
    for q in range(len(querys)):
        query = querys[q]
        query = query.cuda()
        # print('query shape', query.shape, 'key shape', keys_tensor.shape, flush=True)
        # similarity_scores = torch.matmul(keys_tensor, query.T)
        similarity_scores = cosine_similarity(query, keys_tensor)
        top_k = torch.topk(similarity_scores, K)
        top_k_idxs = top_k.indices.tolist()
        if top_k.values[-1] <= args.similarity_tau:
            continue
        # print('Top K largest consine similarity scores:', top_k.values.tolist(), flush=True)
        K_neighbor_keys = [keys[i] for i in top_k_idxs]
        neighbors = [values[i] for i in top_k_idxs] 
        # retrieved_samples.append(neighbors)
        retrieved_samples.append((neighbors, questions[q])) # neighbors is a list of unlabeled inputs, questions[q] is a sentence.
    return retrieved_samples
    
def create_batch_from_memory(samples, tokz, args, task_name):
    input_sen, context_sen, all_sen = [], [], []
    task_prefix = task_name+':'
    raw_sen, ques_sen = [], []

    if args.diff_question:
        for neighbors, question in samples:   
            for sen in neighbors:
                # * Augments retrieved neighbors
                if args.aug_neighbors_from_memory:
                    aug_text_list = eda(sen, num_aug=args.num_aug)
                    for aug_text in aug_text_list:
                        input_text = task_prefix + aug_text + '<QUES>' + question
                        context_sen.append(input_text+'<ANS>')
                        all_sen.append(aug_text + '<QUES>')
                        raw_sen.append(aug_text)
                        ques_sen.append(question)
                        # all_sen.append(aug_text+'<ANS>'+ans_text)
                        # ans_sen.append(ans_text)
                # * Directly use retrieved neighbors
                else:
                    input_text = task_prefix + sen + '<QUES>' + question
                    context_text = input_text+'<ANS>'
                    all_text = sen + '<QUES>'
                    input_sen.append(input_text)
                    context_sen.append(context_text)
                    all_sen.append(all_text)
                    raw_sen.append(sen)
                    ques_sen.append(question)

    else:
        for sen in samples:
            # * Augments retrieved neighbors
            if args.aug_neighbors_from_memory:
                aug_text_list = eda(sen, num_aug=args.num_aug)
                for aug_text in aug_text_list:
                    input_text = task_prefix + aug_text + '<QUES>' + question
                    context_sen.append(input_text+'<ANS>')
                    all_sen.append(aug_text + '<QUES>')
                    raw_sen.append(aug_text)
                    ques_sen.append(question)
                    # all_sen.append(aug_text+'<ANS>'+ans_text)
                    # ans_sen.append(ans_text)
            # * Directly use retrieved neighbors
            else:
                input_text = task_prefix + sen + '<QUES>' + question
                context_text = input_text+'<ANS>'
                all_text = sen + '<QUES>' 
                input_sen.append(input_text)
                context_sen.append(context_text)
                all_sen.append(all_text)
                raw_sen.append(sen)
                ques_sen.append(question)

    batch = []
    batch_list = []
    if len(input_sen)>=1:
        for l in range(len(input_sen)):
            sample_dict = {}
            sample_dict['all_sen'] = all_sen[l]
            sample_dict['input_sen'] = input_sen[l]
            sample_dict['context_sen'] = context_sen[l]
            sample_dict['question'] = ques_sen[l]
            sample_dict['raw_text'] = raw_sen[l]
            batch.append(sample_dict)

        # input_encoding = tokz(input_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
        # all_encoding = tokz(all_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
        # context_encoding =  tokz(context_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
        # raw_text_encoding = tokz(raw_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
        # question_encoding = tokz(ques_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')

        res = {}
        # res["input_id"], res['input_mask'] = input_encoding.input_ids, input_encoding.attention_mask
        # res["all_id"], res['all_mask'] = all_encoding.input_ids, all_encoding.attention_mask
        # res["context_id"], res['context_mask'] = context_encoding.input_ids, context_encoding.attention_mask
        # res["raw_id"], res['raw_mask'] = raw_text_encoding.input_ids, raw_text_encoding.attention_mask
        # res["ques_id"], res['ques_mask'] = question_encoding.input_ids, question_encoding.attention_mask
        res['batch_text'] = batch
    else:
        res = None

    return res

def get_old_center_dict(old_memory, prev_tasks, args=None):
    center_dict = {}
    for prev_task in prev_tasks:
        old_keys = old_memory[prev_task]['keys'] # list of list
        old_keys_tensor = torch.tensor(old_keys).cuda()
        old_center = torch.mean(old_keys_tensor,dim=0)
        center_dict[prev_task] = old_center
    return center_dict

def adapter_selection(prev_tasks, curr_center, old_center_dict, args=None):
    distance_list = []
    for prev_task in prev_tasks:
        dist = cosine_similarity(curr_center, old_center_dict[prev_task])
        distance_list.append(dist)
    print(f'Distances among centers {distance_list}',flush=True)
    max_dist = max(distance_list)
    max_idx = distance_list.index(max_dist)
    adapter_name = prev_tasks[max_idx]
    return adapter_name