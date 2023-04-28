"""
Multiprocess annotating text2sql programs.
"""

import time
import json
import argparse
import copy
import os

from typing import List
import platform
import multiprocessing

from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB
from utils.mmqa.qpmc import Question_Passage_Match_Classifier
from utils.mmqa.qimc import Question_Image_Match_Classifier

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


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
    qpmc = Question_Passage_Match_Classifier()
    qimc = Question_Image_Match_Classifier()
    g_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            g_data_item = dataset[g_eid]
            g_dict[g_eid] = {
                'generations': dict(),
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            table = g_data_item['table']
            header, rows, rows_with_links = table['header'][0], table['rows'][0], table['rows_with_links'][0]

            if args.prompt_style == "create_table_select_3_full_table_w_all_passage_image":
                # Get the retrieved passages & images and re-assign to the neuraldb,
                # then when running "xxxaxllxxx" prompt_style, it will only show the retrieved parts

                # Get the retrieved passages
                new_passages = {"id": [], "title": [], "text": [], "url": []}
                for _id, title, text, url in zip(g_data_item['passages']['id'],
                                                 g_data_item['passages']['title'],
                                                 g_data_item['passages']['text'],
                                                 g_data_item['passages']['url']):
                    if qpmc.judge_match(question=g_data_item['question'], passage=text):
                        new_passages["id"].append(_id)
                        new_passages["title"].append(title)
                        new_passages["text"].append(text)
                        new_passages["url"].append(url)

                g_data_item['passages'] = new_passages

                # Get the retrieved images
                new_images = {"id": [], "title": [], "pic": [], "url": [], "path": []}
                for _id, title, pic, url, path in zip(g_data_item['images']['id'],
                                                      g_data_item['images']['title'],
                                                      g_data_item['images']['pic'],
                                                      g_data_item['images']['url'],
                                                      g_data_item['images']['path']):
                    if qimc.judge_match(g_data_item['id'], g_data_item['question'], pic):
                        new_images["id"].append(_id)
                        new_images["title"].append(title)
                        new_images["pic"].append(pic)
                        new_images["url"].append(url)
                        new_images["path"].append(path)

                g_data_item['images'] = new_images
            else:
                assert args.prompt_style == "create_table_select_3_full_table_w_gold_passage_image"

            db = NeuralDB([{
                "title": "{} ({})".format(table['title'][0], table['caption'][0]),
                "table": {"header": header, "rows": rows, "rows_with_links": rows_with_links}
            }],
                passages=[{"id": _id, "title": title, "text": text} for _id, title, text in
                          zip(g_data_item['passages']['id'], g_data_item['passages']['title'],
                              g_data_item['passages']['text'])],
                images=[{"id": _id, "title": title, "pic": pic} for _id, title, pic in
                        zip(g_data_item['images']['id'], g_data_item['images']['title'], g_data_item['images']['pic'])])
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
            prompt = few_shot_prompt + "\n\n" + generate_prompt

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
    parser.add_argument('--dataset', type=str, default='mmqa',
                        choices=['mmqa'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/prompt_wikitq_v3.txt')
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # Binder program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table_w_all_passage_image',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'create_table_select_3_full_table_w_all_passage_image',
                                 'create_table_select_3_full_table_w_gold_passage_image',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='nsql',
                        choices=['nsql', 'sql', 'answer', 'npython', 'python'])
    parser.add_argument('--n_shots', type=int, default=18)
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
