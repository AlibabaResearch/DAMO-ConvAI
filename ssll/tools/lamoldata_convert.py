import os
import json
import sys

ori_dir = 'lamol_data'
task_names = ['ag','dbpedia','yelp','yahoo']
for task in task_names:
    dir_path = os.path.join('data_lamol','TC',task)
    if not os.path.exists(task) and not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    # Convert train-data
    train_file = os.path.join(ori_dir,task+'_to_squad-train-v2.0.json')
    with open(train_file,'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    train_outfile = os.path.join(dir_path,'train.json')
    with open(train_outfile,'w', encoding='utf-8') as fw:
        for sample in data:
            sample_dict = {}
            x = sample['paragraphs'][0]['context'] + ' '  + sample['paragraphs'][0]['qas'][0]['question']
            y = sample['paragraphs'][0]['qas'][0]['answers'][0]['text']
            sample_dict['input'] = x
            sample_dict['output'] = y
            sample_dict['question'] = sample['paragraphs'][0]['qas'][0]['question'] 
            sample_dict['text_wo_question'] = sample['paragraphs'][0]['context'] 
            print(json.dumps(sample_dict, ensure_ascii=False), file=fw)

    # Convert test-data
    test_file = os.path.join(ori_dir,task+'_to_squad-test-v2.0.json')
    with open(test_file,'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    test_outfile = os.path.join(dir_path,'test.json')
    with open(test_outfile,'w', encoding='utf-8') as fw:
        for sample in data:
            sample_dict = {}
            x = sample['paragraphs'][0]['context'] + ' '  + sample['paragraphs'][0]['qas'][0]['question']
            y = sample['paragraphs'][0]['qas'][0]['answers'][0]['text']
            sample_dict['input'] = x
            sample_dict['output'] = y
            sample_dict['question'] = sample['paragraphs'][0]['qas'][0]['question'] 
            sample_dict['text_wo_question'] = sample['paragraphs'][0]['context'] 
            print(json.dumps(sample_dict, ensure_ascii=False), file=fw)
    print('Finishing dealing with ',task,flush=True)


