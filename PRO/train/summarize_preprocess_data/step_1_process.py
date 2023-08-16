import json
import re
import pprint
import os
import tqdm
import random
random.seed(42)

def gen_global_index():
    index = 0
    while True:
        yield index
        index += 1

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
    
def summarize_from_feedback_preprocess(path,index_generator):
    files = os.listdir(path)
    files = [filename for filename in files if filename.endswith('.json')]
    target_samples = {
        'train':[],
        'dev':[],
        'test':[]
    }

    for filename in files:
        with open(os.path.join(path,filename),'r', encoding="utf-8") as f:
            raw = f.readlines()
    
        data = []
        for line in raw:
            line = json.loads(line)
            data.append(line)

        samples = []
        bar = tqdm.tqdm(data)
        for index,sample in enumerate(bar):
            bar.set_description(os.path.join(path,filename))
            assert len(sample['summaries']) == 2
            if 'post' in sample['info']:
                prefix = "SUBREDDIT: r/{}\nTITLE: {}\nPOST: {}\nTL;DR:".format(sample['info']['subreddit'], sample['info']['title'],sample['info']['post']).strip()
                one_sample = {
                    'available': [
                        {
                            'id':next(index_generator), 
                            'prefix': prefix,
                            'target_num':2,
                            'target':[
                                " {}".format(sample['summaries'][sample['choice']]['text'].strip()),
                                " {}".format(sample['summaries'][1-sample['choice']]['text'].strip()),
                            ]
                        },
                    ],
                    'split': split_trans(sample['split']),
                    'source': {
                        'path': os.path.join(path,filename),
                        'line_num': index+1,
                    }
                }
                target_samples[one_sample['split']].append(one_sample)
            else:
                prefix = "Article: {}\nTL;DR:".format(sample['info']['article'])
                pass
                
    os.makedirs(path.replace('raw_data','preprocessed_data'), exist_ok=True)
    
    true_dev_index = random.sample(list(range(len(target_samples['dev']))),1000)
    true_dev = []
    for index, sample in enumerate(target_samples['dev']):
        if index in true_dev_index:
            sample['split'] = 'dev'
            true_dev.append(sample)
        else:
            sample['split'] = 'train'
            target_samples['train'].append(sample)
    target_samples['dev'] = true_dev

    with open(os.path.join(path.replace('raw_data','preprocessed_data'), "train.json"), 'w', encoding='utf-8') as f:
        for sample in target_samples['train']:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')
        print("{}: {}".format(os.path.join(path.replace('raw_data','preprocessed_data'),"train.json"),len(target_samples['train'])))

    with open(os.path.join(path.replace('raw_data','preprocessed_data'), "dev.json"), 'w', encoding='utf-8') as f:
        for sample in target_samples['dev']:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')
        print("{}: {}".format(os.path.join(path.replace('raw_data','preprocessed_data'),"dev.json"),len(target_samples['dev'])))

    with open(os.path.join(path.replace('raw_data','preprocessed_data'), "test.json"), 'w', encoding='utf-8') as f:
        for sample in target_samples['test']:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')
        print("{}: {}".format(os.path.join(path.replace('raw_data','preprocessed_data'),"test.json"),len(target_samples['test'])))

if __name__ == "__main__":
    global_index_generator = gen_global_index()

    summarize_from_feedback_preprocess(
        os.path.join('..','..','data','raw_data','summarize_from_feedback','comparisons'),
        global_index_generator
    )