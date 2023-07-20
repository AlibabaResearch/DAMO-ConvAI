import os
import json
import time
import json
import bisect
import openai
from functools import partial
import random
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import argparse
from collections import defaultdict
random.seed(42)

# please put your key here
keys = []

class ChatGPT:
    def __init__(self, args):
        self.fin = args.fin
        self.fout = args.fout
        self.n_workers = args.n_workers
        self.multithread = args.multithread
        self.n_samples = args.n_samples
        self.basenames = []
        self.key_index = 0
        self.check = args.check
        self.args = args
    

    def construct_db_prompt(self, db):
        dic = {}
        for domain in db:
            dic[domain] = {}
            for slot in db[domain]:
                dic[domain][slot] = str(db[domain][slot])
        result = str(dic)
        return result

    def construct_current_input(self, dialog, dialogue_index, turn_index):
        # prompt = dialog['log'][turn_index]['prompt'] + ' \n'
        dst_key = dialogue_index + '-' + str(turn_index)
        prompt = 'Please continue the dialogue as a task-oriented dialogue system called SYSTEM. The answer of SYSTEM should follow the DATABASE provided in json format and answer the USER’s last utterance.'
        prompt +='''\nSYSTEM can recommend and inform the contents in the DATABASE according to the utterance of the USER and return the name of the entity when it comes to restaurants, hotels and attractions, and the trainid when it comes to trains. \nBut only when the USER requests information about an entity in the DATABASE, such as a phone number, should SYSTEM inform the corresponding content.'''
 
        
        dial_db = db_query[dialogue_index]
        # print(dialogue_index, dial_db)
        db_key = bisect.bisect_left(dial_db, int(turn_index))
        # print(dialogue_index, turn_index, dial_db[db_key])
        db_key = dialogue_index + '-' + str(dial_db[db_key])
        db = db_result[db_key]
        
        db = self.construct_db_prompt(db) if db else ''
        prompt_db = prompt + '\n \n' + 'DATABASE:\n' + db + ' \n\nDIALOGUE CONTENT:\n'
        
        for turn_number in range(turn_index):
            if turn_number % 2 == 0:
                prompt_db = prompt_db + 'USER: ' +  dialog['log'][turn_number]['text'] + '\n'
            else:
                prompt_db = prompt_db + 'SYSTEM: ' +  dialog['log'][turn_number]['result'].replace('\n', '') + ' \n'
        
        prompt_db = prompt_db + 'SYSTEM:'
        return prompt_db
        
    def process(self, dialog, dialogue_index, dialogue_number):
        openai.api_key = random.choice(keys)
        
        for turn_index in tqdm(range(len(dialog['log']))):
            # and 'result' not in dialog['log'][turn_index+1]
            if turn_index % 2 != 0 :
                try:
                    gpt_input = self.construct_current_input(dialog, dialogue_index, turn_index)
                    #print(gpt_input)
                except:
                    print(dialogue_index, turn_index)
                    continue
                    
                # print(gpt_input)
                # print(dst_key, turn_dst[dst_key], )
                result = ''
                while not result:
                    try:
                        if self.args.instruction:
                            completion = openai.Completion.create(
                                model="text-davinci-003",
                                prompt=gpt_input,
                                temperature=0,
                                max_tokens=1024,
                                top_p=1,
                            )
                            result = completion["choices"][0]['text'].split('USER')[0]
                        else:
                            completion = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": gpt_input}]
                            )
                            result = completion.choices[0].message['content'].split('USER')[0]
                            
                    except Exception as e:
                        print(openai.api_key, ':', e)
                        openai.api_key = random.choice(keys)
                        time.sleep(3)
                
                dialog['log'][turn_index]['result'] = result
                print(dialogue_index+'-'+str(turn_index), result)
            
        # # bar.update()
        return [dialog, dialogue_number]

    def run(self):
        with open(self.fin, 'r') as fp:
            # print(self.fin)
            # lines = fp.read().split('\n')
            json_files = json.load(fp)
        if self.multithread:
            # bar = tqdm(total=len(lines))
            dialogue_lines = [json_files[key] for key in json_files.keys()]
            dialogue_indexs = [key for key in json_files.keys()]
            dialogue_numbers = range(0,1000)
            bar = tqdm(total=len(dialogue_indexs))
            
            list_all = []
            for index in range(0,len(dialogue_indexs)):
                temp_instance = [dialogue_lines[index], dialogue_indexs[index], dialogue_numbers[index]]
                list_all.append(temp_instance)
                
            def process_with_bar(list_all):
                bar.update()
                # list_all_1[0].update()
                return self.process(list_all[0],list_all[1],list_all[2])
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                results = [executor.map(process_with_bar, list_all)]

            for item in results:
                for instance in item:
                    dialogue_number_index = instance[1]
                    dialogue_name_index = dialogue_indexs[dialogue_number_index]
                    json_files[dialogue_name_index] = instance[0]
                
        else:
            results = []
            dialogue_lines = [json_files[key] for key in json_files.keys()]
            dialogue_indexs = [key for key in json_files.keys()]
            dialogue_numbers = range(0,1000)
            # bar = tqdm(total=len(dialogue_indexs))

            list_all = []
            for index in range(0,len(dialogue_indexs)):
                temp_instance = [dialogue_lines[index], dialogue_indexs[index], dialogue_numbers[index]]
                list_all.append(temp_instance)

            for dialog in tqdm(list_all):
                results.append(self.process(dialog[0],dialog[1],dialog[2]))
                # break

            for instance in results:
                dialogue_number_index = instance[1]
                dialogue_name_index = dialogue_indexs[dialogue_number_index]
                json_files[dialogue_name_index] = instance[0]

            
        with open(args.fout, 'w') as f:
            json.dump(json_files, f, sort_keys=False, indent=2)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', action='store_true', help='use 003 or not')
    parser.add_argument('--fin', type=str, help='Input file path')
    parser.add_argument('--fout', type=str, default=None, help='Output file path')
    parser.add_argument('--model', type=str, default=None, help='Output file path')
    parser.add_argument('--n_workers', type=int, default=30, help='Number of workers')
    parser.add_argument('--multithread', action="store_true", help='Use multiple threads to process the input file')
    parser.add_argument('--n_samples', type=int, default=0, help='Number of workers')
    parser.add_argument('--check', action="store_true", help='Check the format for role')
    args = parser.parse_args()

    model_name = '003' if args.instruction else ''
    print('use_003:', args.instruction)

    setting = '_policy' if args.model == 'gt' else '_e2e'

    if args.fout is None:
        args.fout = args.fin.replace('.json', '') + setting + '_out.json'
    
    db_query = json.load(open(f'./{args.model}_db_query.json'))
    db_result = json.load(open(f'./{args.model}_db_result.json'))
    
    chat_gpt = ChatGPT(args)
    chat_gpt.run()

