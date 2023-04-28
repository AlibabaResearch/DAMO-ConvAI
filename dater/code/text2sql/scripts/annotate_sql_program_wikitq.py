"""
Multiprocess annotating text2sql programs.
"""

import time
import json
import argparse
import copy
import os
import random
from typing import List
import platform
import multiprocessing

from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        dataset,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            g_data_item = dataset[g_eid]
            g_dict[g_eid] = {
                'generations': [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,)
            )
            generate_prompt = generate_prompt.split('Q:',-1)[0]
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            sub_q = g_data_item['sub_q']
            sub_q = sub_q.strip().split('\n')
            q_len = len(sub_q)
            for idx,q in enumerate(sub_q):
                prompt += 'Q'+str(idx+1)+': ' +q+'\n'
            prompt += 'NeuralSQL1:'
            print(prompt)
            # Ensure the input length fit Codex max input tokens by shrinking the n_shots
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:  # TODO: Add shrink rows
                n_shots -= 1
                assert n_shots >= 0
                few_shot_prompt = generator.build_few_shot_prompt_from_file(
                    file_path=args.prompt_file,
                    n_shots=n_shots
                )
                prompt = few_shot_prompt + "\n\n" + generate_prompt

            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))
            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                g_dict[eid]['generations'] = g_pairs

            built_few_shot_prompts = []
        except Exception as e:
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")

    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'] = g_pairs

    return g_dict


def main():
    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    if args.dataset == "wikitq":
        st2subq={}
        with open(os.path.join(ROOT_DIR, "templates", f"wikitq_{args.dataset_split}_decomposed.jsonl"), "r") as f:
            lines = f.readlines()
            for line in lines:
                dic = json.loads(line)
                st = dic['data_item']['statement']
                subq = dic['generations'][0][0]
                st2subq[st] = subq
        new_dataset = []
        for d in dataset:
            st = d['question']
            if st in st2subq.keys():
                d['sub_q'] = st2subq[st]
                if args.parsing_num_rows < len(d['table']['rows']):
                    d['table']['rows'] = random.sample(d['table']['rows'],k=args.parsing_num_rows)
                new_dataset.append(d)
        dataset = new_dataset
        
        
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    # Annotate
    generator = Generator(args, keys=keys)
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    g_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
        worker_results.append(pool.apply_async(worker_annotate, args=(
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
    save_file_name = f'sql_program_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'tab_fact'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_parsing.txt')
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=20)

    # Binder program generation options
    parser.add_argument('--parsing_num_rows', type=int, default=1000)

    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'create_table_select_3_full_table_w_all_passage_image',
                                 'create_table_select_3_full_table_w_gold_passage_image',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='nsql',
                        choices=['nsql', 'sql', 'answer', 'npython', 'python'])
    parser.add_argument('--n_shots', type=int, default=14)
    parser.add_argument('--seed', type=int, default=42)

    # Codex options
    parser.add_argument('--engine', type=str, default="code-davinci-002")
    parser.add_argument('--n_parallel_prompts', type=int, default=2)
    parser.add_argument('--max_generation_tokens', type=int, default=512)
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
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
