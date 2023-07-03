import json
import re
import pprint
import os
import tqdm

def gen_global_index():
    index = 0
    while True:
        yield index
        index += 1

Roles = {
    "Human": "<|prompter|>",
    "Assistant": "<|assistant|>"
}

def hhrlhf_preprocess(path,filename,index_generator,split='train'):
    with open(os.path.join(path,filename),'r', encoding='utf-8') as f:
        raw = f.readlines()
    
    data = []
    for line in raw:
        line = json.loads(line)
        data.append(line)
    
    # Thank OpenAssistant for their helpful public code:
    # https://github.com/LAION-AI/Open-Assistant/blob/c6591bd04dd337c716d097b2d267b92403850396/model/model_training/custom_datasets/rank_datasets.py
    def _split_dialogue(text: str):
        lines = text.split("\n\n")

        dialogue: list[tuple[str, str]] = []

        # go over messages and combine consecutive messages from the
        # same speaker (OA v1 expects alternating roles)
        role = None
        messages = []
        for line in lines:
            if line.startswith("Human:"):
                speaker = "Human"
                message = line[7:]
            elif line.startswith("Assistant:"):
                speaker = "Assistant"
                message = line[11:]
            else:
                continue
            if role != speaker:
                if role is not None:
                    dialogue.append((Roles[role], "\n".join(messages)))
                    messages = []
                role = speaker
            messages.append(message.strip())

        if role is not None and len(messages) > 0:
            dialogue.append([Roles[role], "\n".join(messages)])

        return dialogue

    if split == "train" or split == "dev":
        samples = []
        bar = tqdm.tqdm(data)
        for index, sample in enumerate(bar):
            bar.set_description(os.path.join(path,filename))
            if "Assistant" not in sample["chosen"]:
                print("Flag1", index+1)
                continue
            chosen = _split_dialogue(sample["chosen"]) # [(Role, Post), (Role, Post), ... ]
            rejected = _split_dialogue(sample["rejected"])
            assert rejected[0][0] == "<|prompter|>" and chosen[0][0] == "<|prompter|>"

            # only very few items have non matching lengths
            if len(rejected) == len(chosen):
                assert chosen[-1][0] == rejected[-1][0]

                prefix = [role+text for role, text in chosen[:-1]] # need to be concated with [EOS] in practice
                prefix.append(chosen[-1][0])
                good_reply = chosen[-1][1]  # last part of dialog, the text
                bad_reply = rejected[-1][1]  # last part of dialog, the text
                
                one_sample = {
                    'extended':[
                        {'id':next(index_generator), 'prefix': prefix, 'target_num':2, 'target':[]}
                    ],
                    'available':[
                        {'id':next(index_generator), 'prefix': prefix, 'target_num':2, 'target':[good_reply, bad_reply]}
                    ],
                    'available_for_test':[],
                    'split': split,
                    'source':{
                        'path': os.path.join(path,filename),
                        'line_num': index+1,
                        'task': "dialogue"
                    }
                }
                samples.append(one_sample)
    else:
        samples = []
        bar = tqdm.tqdm(data)
        for index, sample in enumerate(bar):
            bar.set_description(os.path.join(path,filename))
            assert "Assistant" in sample["chosen"] and "Assistant" in sample["rejected"]
            chosen = _split_dialogue(sample["chosen"]) # [(Role, Post), (Role, Post), ... ]
            rejected = _split_dialogue(sample["rejected"]) # [(Role, Post), (Role, Post), ... ]
            assert chosen[0][0] == "<|prompter|>" and rejected[0][0] == "<|prompter|>"
            
            # prepare chosen sample
            prefix = []
            step = 0
            for role, text in chosen:
                step += 1
                if role == "<|prompter|>":
                    prefix.append([role, text])
                elif role == "<|assistant|>":
                    temp_prefix = [temp_role+temp_text for temp_role, temp_text in prefix]
                    temp_prefix.append(role) # need to be concated with [EOS] in practice
                    temp_reply = text # last part of dialog, the text
                    chosen_sample = {
                        'extended':[
                            {'id':next(index_generator), 'prefix': temp_prefix,'target_num':2, 'target':[]}
                        ],
                        'available':[],
                        'available_for_test':[{
                            'id': next(index_generator), 
                            'prefix': temp_prefix,
                            'target_num': 1,
                            'target':[temp_reply]
                        }],
                        'split': split,
                        'source':{
                            'path': os.path.join(path,filename),
                            'line_num': index+1,
                            'task': 'dialogue',
                            'selected': 'chosen',
                            'completed': step == len(chosen)
                        },
                    }
                    if chosen_sample['source']['completed']:
                        samples.append(chosen_sample)
                    prefix.append([role, text])
                else:
                    raise Exception("???")
            # prepare rejected sample
            prefix = []
            step = 0
            for role, text in rejected:
                step += 1
                if role == "<|prompter|>":
                    prefix.append([role, text])
                elif role == "<|assistant|>":
                    temp_prefix = [temp_role+temp_text for temp_role, temp_text in prefix]
                    temp_prefix.append(role) # need to be concated with [EOS] in practice
                    temp_reply = text # last part of dialog, the text
                    rejected_sample = {
                        'extended':[
                            {'id':next(index_generator), 'prefix': temp_prefix,'target_num':2, 'target':[]}
                        ],
                        'available':[],
                        'available_for_test':[{
                            'id': next(index_generator), 
                            'prefix': temp_prefix,
                            'target_num': 1,
                            'target':[temp_reply]
                        }],
                        'split': split,
                        'source':{
                            'path': os.path.join(path,filename),
                            'line_num': index+1,
                            'task': 'dialogue',
                            'selected': 'rejected',
                            'completed': step == len(rejected)
                        },
                    }
                    if rejected_sample['source']['completed']:
                        samples.append(rejected_sample)
                    prefix.append([role, text])
                else:
                    raise Exception("???")

    os.makedirs(path.replace('raw_data','preprocessed_data'), exist_ok=True)
    with open(os.path.join(path.replace('raw_data','preprocessed_data'),"{}.json".format(split)),'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample,ensure_ascii=False)+'\n')
    print("{}: {}".format(os.path.join(path.replace('raw_data','preprocessed_data'),"{}.json".format(split)),len(samples)))
    return samples

if __name__ == "__main__":
    # get a global index generator
    global_index_generator = gen_global_index()

    # prepare to post-processing
    res = {
        'hhrlhf':[],
    }

    prompts = {
        'hhrlhf': '<prefix>',
        'summarize':'<prefix>',
        'webgpt':'<prefix>',
        'tldr':'<prefix>',
    }
    # process raw datasets
    # hhrlhf
    res['hhrlhf'] = [
        hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','harmless-base'),'train.jsonl',global_index_generator,split='train'),
        # hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','harmless-base'),'test.jsonl',global_index_generator,split='test'),
        hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-base'),'train.jsonl',global_index_generator,split='train'),
        # hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-base'),'test.jsonl',global_index_generator,split='test'),
        hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-online'),'train.jsonl',global_index_generator,split='train'),
        # hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-online'),'test.jsonl',global_index_generator,split='test'),
        hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-rejection'),'train.jsonl',global_index_generator,split='train'),
        # hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-rejection'),'test.jsonl',global_index_generator,split='test'),
    ]
    hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','harmless-base'),'test.jsonl',global_index_generator,split='dev')
    hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-base'),'test.jsonl',global_index_generator,split='dev')
    hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-online'),'test.jsonl',global_index_generator,split='dev')
    hhrlhf_preprocess(os.path.join('..','..','data','raw_data','hhrlhf','helpful-rejection'),'test.jsonl',global_index_generator,split='dev')
    
    global_prefixes = []
    global_extended_samples = 0
    for key in res:
        for dataset in res[key]:
            for sample in dataset:
                for sub_sample in sample['extended']:
                    prefix = "".join(sub_sample['prefix'])
                    prefix = prefix.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ")
                    prefix = prompts[key].replace('<prefix>', prefix)
                    global_prefixes.append(
                        {
                            'id': sub_sample['id'],
                            'prefix': prefix,
                            'target_num': sub_sample['target_num'],
                            'target': []
                        }
                    )
                    global_extended_samples += sub_sample['target_num']
    
    
    print('Total Num: {}'.format(len(global_prefixes)))
    print('Total Extended Num: {}'.format(global_extended_samples))