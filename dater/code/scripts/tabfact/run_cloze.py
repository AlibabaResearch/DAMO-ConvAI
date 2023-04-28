import pandas as pd
import numpy as np
import argparse
import os
import time
import json
import copy
from typing import List, Dict
import random
import multiprocessing

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../..")
print(ROOT_DIR)
import sys
sys.path.append('../..')
from gloc.generation.generator import Generator
from gloc.utils import dict2df

def worker(
    pid: int,
    args,
    generator: Generator,
    g_eids: List,
    dataset: List[Dict],
    tokenizer
):
    """
    A worker process for annotating.
    """
    generation_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            data_item = dataset[g_eid]
            generation_dict[g_eid] = {
                'generations': [],
                'data_item': copy.deepcopy(data_item)
            }
            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            # print(data_item)
            # print(data_item['statement'])
            # print(data_item['table_caption'])
            # print(type(dict2df(data_item['table_text'])))
            generate_prompt = generator.build_generate_prompt(
                data_item={
                    'question': data_item['statement'],
                    'title': data_item['table_caption'] if 'table_caption' in data_item.keys() else None,
                    'table': dict2df(data_item['table_text'])
                },
                num_rows = args.num_rows,
                select_type=args.select_type
            )
            # print("there!!!!!!!!!")
            prompt = few_shot_prompt + '\n\n' + generate_prompt

            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:  # TODO: Add shrink rows
                n_shots -= 1
                assert n_shots >= 0
                few_shot_prompt = generator.build_few_shot_prompt_from_file(
                    file_path=args.prompt_file,
                    n_shots=n_shots
                )
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            # print("*"*80)
            # print(prompt)
            built_few_shot_prompts.append((g_eid,prompt))


            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{data_item['statement']}")
            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                continue
            
            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[1], reverse=True)
                generation_dict[eid]['generations'] = g_pairs
            
            built_few_shot_prompts = []
        except Exception as e:
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{data_item['statement']} generation error: {e}")
    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[1], reverse=True)
            generation_dict[eid]['generations'] = g_pairs
    
    return generation_dict

def main():
    def twoD_list_transpose(arr):
        return [[arr[i][j] for i in range(len(arr))] for j in range(len(arr[0]))]
    def filter_col(table,pred_col):
        table = twoD_list_transpose(table)
        new_table = []
        for cols in table:
            if cols[0] in pred_col:
                new_table.append(copy.deepcopy(cols))
        if len(new_table) == 0:
            new_table = table
        new_table = twoD_list_transpose(new_table)
        return new_table
    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    # Load dataset
    
    dataset = []

    start_time = time.time()
    if args.dataset == 'tab_fact' and args.dataset_split == 'cloze':

        with open('../../datasets/tabfact/raw_dataset/small_test.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                dic = json.loads(line)
                dataset.append(dic)


    # Load openai keys
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

     # Annotate
    generator = Generator(args, keys=keys)
    # Map data to different processing
    # dataset = random.sample(dataset,100)
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)

    print('\n******* Annotating *******')
    print(len(dataset))
    g_dict = dict()
    worker_results = []

    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='../../utils_file/gpt2')
        worker_results.append(pool.apply_async(worker, args=(
            pid,
            args,
            generator,
            generate_eids_group[pid],
            dataset,
            tokenizer
        )))

        # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results
    save_file_name = f'gloc_{args.select_type}_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path or name
    parser.add_argument('--dataset', type=str, default='tab_fact',
                    choices=['wikitq', 'tab_fact'])
    parser.add_argument('--dataset_split', type=str, default='cloze', choices=['train', 'validation', 'test', 'gold','cloze'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/tabfact/cloze.txt')
    parser.add_argument('--save_dir', type=str, default='results/tabfact/cloze')
    
    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=20)
    

    # Prompt options 
    parser.add_argument('--select_type', type=str, default='cloze',
                        choices=['col', 'row', 'all','cloze','cplex'])
    #########################
    #######################
    parser.add_argument('--num_rows', type=int, default=3)
    parser.add_argument('--n_shots', type=int, default=14)
    parser.add_argument('--seed', type=int, default=42)

    # CodeX options
    parser.add_argument('--engine', type=str, default="code-davinci-002")
    parser.add_argument('--n_parallel_prompts', type=int, default=2)

    parser.add_argument('--max_generation_tokens', type=int, default=80)
    parser.add_argument('--max_api_total_tokens', type=int, default=8001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=20)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')
    
    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')
    
    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')

    # if args.select_type == 'col':
    #     assert 'col' in args.prompt_file 

    # # if args.select_type == 'row':
    # #     assert 'row' in args.prompt_file

    # if args.select_type == 'all':
    #     assert 'all' in args.prompt_file

    
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    
    main()


