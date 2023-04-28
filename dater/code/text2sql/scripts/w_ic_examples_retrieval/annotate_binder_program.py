"""
Multiprocess annotating text2sql programs with sample selection.
"""

import time
import json
import argparse
import copy
import os

from typing import List
import platform
import multiprocessing
import pandas as pd

from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        dataset,
        dataset_for_retrieve,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []
    # Hard encode, only work for test
    with open(os.path.join(ROOT_DIR, "scripts", "w_ic_examples_retrieval",
                           "tab_fact_in_context_examples_test_from_train_ids.json"), "r") as f:
        tab_fact_in_context_examples_test_from_train_ids = json.load(f)

    with open(os.path.join(ROOT_DIR, "scripts", "w_ic_examples_retrieval",
                           "nsql_annotations_tab_fact_train.json"), "r") as f:
        nsql_annotations_tab_fact_train = json.load(f)

    tab_fact_in_context_examples_test = {}
    for _id, ids in tab_fact_in_context_examples_test_from_train_ids.items():
        nsqls_for_the_examples = [{"nid": nid, "nsql": nsql_annotations_tab_fact_train[nid]} for nid in ids]
        tab_fact_in_context_examples_test[_id] = nsqls_for_the_examples

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
            all_nsqls = tab_fact_in_context_examples_test[str(g_eid)]

            few_shot_prompts = []
            for in_context_example in all_nsqls[:n_shots]:
                ic_eid = int(in_context_example['nid'])
                ic_data_item = dataset_for_retrieve[ic_eid]
                if not isinstance(ic_data_item['table'], pd.DataFrame):
                    db = NeuralDB(
                        tables=[{'title': ic_data_item['table']['page_title'], 'table': ic_data_item['table']}]
                    )
                    ic_data_item['table'] = db.get_table_df()
                    ic_data_item['title'] = db.get_table_title()
                    ic_data_item['nsql'] = in_context_example['nsql']
                ic_prompt = generator.prompt_builder.build_one_shot_prompt(
                    **ic_data_item,
                    prompt_type=('question', 'nsql')
                )
                few_shot_prompts.append(ic_prompt)

            few_shot_prompt = """Generate SQL given the statement and table to verify the statement correctly.
            If statement-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar QA("map@").
            If mapping to a new column still can not answer the statement with valid SQL, turn to an end-to-end solution by a new grammar QA("ans@"). This grammar aims to solve all the rest of complex statements or tables.
            """ + "\n" + "\n\n\n".join([_.strip() for _ in few_shot_prompts])

            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=('nsql',)
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt

            # Ensure the input length fit Codex max input tokens by shrinking the n_shots
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:  # TODO: Add shrink rows
                n_shots -= 1
                assert n_shots >= 0
                few_shot_prompt = "\n\n\n".join(few_shot_prompt.split("\n\n\n")[:-1])
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
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    # Load dataset for retrieve
    dataset_for_retrieve = load_data_split("tab_fact", "train")

    # For TabFact test split, we load the small test set (about 2k examples) to test,
    # since it is expensive to test on full set
    if args.dataset == "tab_fact" and args.dataset_split == "test":
        with open(os.path.join(ROOT_DIR, "utils", "tab_fact", "small_test_id.json"), "r") as f:
            small_test_ids_for_iter = json.load(f)
        dataset = [data_item for data_item in dataset if data_item['table']['id'] in small_test_ids_for_iter]

    # Load openai keys
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
            dataset_for_retrieve,
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results
    # "_".join(["{}={}".format(k, str(args.__dict__[k])) for k in args.__dict__ if k not in ['api_keys_file', 'prompt_file', 'save_dir', 'stop_tokens']])
    save_file_name = f'binder_program_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='tab_fact',
                        choices=['wikitq', 'tab_fact', 'mmqa'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # Binder program generation options
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
