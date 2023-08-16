import os
import sys
sys.path.append("..")
import json
import random
import numpy as np
import tqdm
from utils.metrics_hh import create_reward_fn
get_score, reward_batch_size = create_reward_fn()

def split_trans(split):
    if split == 'train' or split == 'test' or split == 'dev':
        return split
    elif split == 'valid':
        return 'dev'
    elif split == 'valid1':
        return 'dev'
    elif split == 'valid2':
        return 'test'
    else:
        raise Exception('guaiguaidigai')

def concat_wo_ranker(prefixes, suffixes):
    #prefixes = [[a,b,c],[d,e,f]]
    #suffixes = [[a,b,c],[d,e,f]]
    training_stage_num = len(prefixes[0])
    batch_size = len(prefixes)
    new_prefixes = sum(prefixes,[])
    new_suffixes = sum(suffixes,[])
    rewards = get_score(new_prefixes, new_suffixes).view(batch_size, training_stage_num).cpu().detach().numpy().tolist() #[batch_size, ranking]

    return prefixes, suffixes, rewards

def reward_model_ranker(prefixes, suffixes):
    #prefixes = [[a,b,c],[d,e,f]]
    #suffixes = [[a,b,c],[d,e,f]]
    training_stage_num = len(prefixes[0])
    batch_size = len(prefixes)
    new_prefixes = sum(prefixes,[])
    new_suffixes = sum(suffixes,[])
    rewards = get_score(new_prefixes, new_suffixes).view(batch_size, training_stage_num).cpu().detach().numpy() #[batch_size, ranking]
    indices = np.argsort(-rewards,axis=1)
    prefixes = [[prefixes[i][index] for index in indices[i]] for i in range(batch_size)]
    suffixes = [[suffixes[i][index] for index in indices[i]] for i in range(batch_size)]
    rewards = [[float(rewards[i][index]) for index in indices[i]] for i in range(batch_size)]
    return prefixes, suffixes, rewards

def extract_train_data(root_dir, if_score, if_rerank, training_stage_num = None, split='train'):
    file_list = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if not file.endswith("json"):
                continue
            file_list.append(os.path.join(root,file))
    
    training_data = []
    for file in tqdm.tqdm(file_list,desc="re-formating"):
        with open(file,'r', encoding='utf-8') as f:
            raw_data = f.readlines()
        for line in raw_data:
            sample = json.loads(line)
            if split_trans(sample['split']) == split:
                new_sample = {'meta': sample['source'], 'prefix':[],'suffix':[]}
                if data_aug:
                    for s in sample['extended']+sample['available']:
                        for suffix in s['target']:
                            assert isinstance(suffix,str)
                            new_sample['prefix'].append(s['prefix'])
                            new_sample['suffix'].append(suffix)
                else:
                    for s in sample['available']:
                        for suffix in s['target']:
                            assert isinstance(suffix,str)
                            new_sample['prefix'].append(s['prefix'])
                            new_sample['suffix'].append(suffix)
                training_data.append(new_sample)
                if training_stage_num == None:
                    training_stage_num = len(new_sample['prefix'])
                assert training_stage_num == len(new_sample['prefix'])
    

    if if_score:
        batch_size = reward_batch_size / 2 # default
        for index in tqdm.tqdm(range(0,len(training_data),batch_size),desc="rewarding"):
            prefixes = []
            suffixes = []
            if len(training_data)-index < batch_size:
                batch_size = len(training_data)-index
            for sub_index in range(batch_size):
                prefixes.append(training_data[index+sub_index]['prefix'])
                suffixes.append(training_data[index+sub_index]['suffix'])
            if if_rerank:
                prefixes, suffixes, rewards = reward_model_ranker(prefixes,suffixes)
            else:
                prefixes, suffixes, rewards = concat_wo_ranker(prefixes,suffixes)
            for sub_index in range(batch_size):
                training_data[index+sub_index]['prefix'] = prefixes[sub_index]
                training_data[index+sub_index]['suffix'] = suffixes[sub_index]
                training_data[index+sub_index]['reward'] = rewards[sub_index]
    else:
        for l in training_data:
            l['reward'] = [1.0] * len(l['suffix'])
    
    for l in training_data:
        l['sft_index'] = 0
    
    return training_data

if __name__ == '__main__':
    root_dir = os.path.join('..','..','data',"preprocessed_data")
    data_aug = False
    os.makedirs(os.path.join('..','..','data','hh_train_len2'), exist_ok=True)
    random.seed(42)
    training_data = extract_train_data(root_dir = os.path.join(root_dir, "hhrlhf"), if_score = True, if_rerank=True, split = 'train') 
    random.shuffle(training_data)
    with open(os.path.join('..','..','data','hh_train_len2','train.json'),'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')

    data_aug = False
    random.seed(42)
    harmless_base_dev_data = extract_train_data(root_dir = os.path.join(root_dir, "hhrlhf", "harmless-base"), if_score = True, if_rerank=False, split = 'dev') 
    random.shuffle(harmless_base_dev_data)
        
    random.seed(42)
    helpful_base_dev_data = extract_train_data(root_dir = os.path.join(root_dir, "hhrlhf", "helpful-base"), if_score = True, if_rerank=False, split = 'dev') 
    random.shuffle(helpful_base_dev_data)

    random.seed(42)
    helpful_online_dev_data = extract_train_data(root_dir = os.path.join(root_dir, "hhrlhf", "helpful-online"), if_score = True, if_rerank=False, split = 'dev') 
    random.shuffle(helpful_online_dev_data)

    random.seed(42)
    helpful_rejection_dev_data = extract_train_data(root_dir = os.path.join(root_dir, "hhrlhf", "helpful-rejection-sampled"), if_score = True, if_rerank=False, split = 'dev') 
    random.shuffle(helpful_rejection_dev_data)
    
    total_dev_data = harmless_base_dev_data + helpful_base_dev_data + helpful_online_dev_data + helpful_rejection_dev_data
    random.shuffle(total_dev_data)
    total_dev_data = total_dev_data[:280]
    with open(os.path.join('..','..','data','hh_dev','sampled_dev.json'),'w', encoding='utf-8') as f:
        for sample in total_dev_data:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')