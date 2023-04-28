"""
Multiprocess executing text2sql programs.
"""

import json
import argparse
import platform, multiprocessing
import os
import time

from nsql.nsql_exec import Executor, NeuralDB
from utils.normalizer import post_process_sql
from utils.utils import load_data_split, majority_vote
from utils.evaluator import Evaluator

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


def worker_execute(
        pid,
        args,
        dataset,
        nsql_dict,
        keys
):
    """
    A worker process for execution.
    """
    result_dict = dict()
    n_total_samples, n_correct_samples = 0, 0
    for eid, data_item in enumerate(dataset):
        eid = str(eid)
        if eid not in nsql_dict:
            continue
        print(f"Process#{pid}: eid {eid}, wtq-id {data_item['id']}")
        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text'].split(" | ")
        n_total_samples += 1
        # Load table
        table = data_item['table']
        header, rows, rows_with_links = table['header'][0], table['rows'][0], table['rows_with_links'][0]
        title = table['title'][0]

        executor = Executor(args, keys)
        # Execute
        exec_answer_list = []
        nsql_exec_answer_dict = dict()
        for idx, (nsql, logprob) in enumerate(nsql_dict[eid]['nsqls']):
            print(f"Process#{pid}: eid {eid}, original_id {data_item['id']}, executing program#{idx}, logprob={logprob}")
            try:
                if nsql in nsql_exec_answer_dict:
                    exec_answer = nsql_exec_answer_dict[nsql]
                else:
                    db = NeuralDB([{
                        "title": "{} ({})".format(table['title'][0], table['caption'][0]),
                        "table": {"header": header, "rows": rows, "rows_with_links": rows_with_links}
                    }],
                        passages=[{"id": _id, "title": title, "text": text} for _id, title, text in
                                  zip(data_item['passages']['id'], data_item['passages']['title'],
                                      data_item['passages']['text'])],
                        images=[{"id": _id, "title": title, "pic": pic} for _id, title, pic in
                                zip(data_item['images']['id'], data_item['images']['title'],
                                    data_item['images']['pic'])])

                    nsql = post_process_sql(
                        sql_str=nsql,
                        df=db.get_table_df(),
                        process_program_with_fuzzy_match_on_db=args.process_program_with_fuzzy_match_on_db,
                        table_title=title
                    )
                    exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)
                    if isinstance(exec_answer, str):
                        exec_answer = [exec_answer]
                    nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                print(f"Process#{pid}: Execution error {e}")
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)
            # Store tmp execution answers
            if nsql_dict[eid].get('exec_answers', None) is None:
                nsql_dict[eid]['exec_answers'] = []
            nsql_dict[eid]['exec_answers'].append(exec_answer)
        # Majority vote to determine the final prediction answer
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=nsql_dict[eid]['nsqls'],
            pred_answer_list=exec_answer_list,
            allow_none_and_empty_answer=args.allow_none_and_empty_answer,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )
        # Evaluate
        result_dict[eid]['pred_answer'] = pred_answer
        result_dict[eid]['nsql'] = pred_answer_nsqls
        gold_answer = data_item['answer_text'].split(" | ")
        score = Evaluator().evaluate(
            pred_answer,
            gold_answer,
            dataset=args.dataset,
            question=result_dict[eid]['question']
        )
        n_correct_samples += score
        print(f'Process#{pid}: pred answer: {pred_answer}')
        print(f'Process#{pid}: gold answer: {gold_answer}')
        if score == 1:
            print(f'Process#{pid}: Correct!')
        else:
            print(f'Process#{pid}: Wrong.')
        print(f'Process#{pid}: Accuracy: {n_correct_samples}/{n_total_samples}')

    return result_dict


def main():
    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
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

    # Load programs and process as a unified format
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    nsql_dict = dict()
    for eid, data_dict in data.items():
        if data[eid]['generations']:
            nsqls = data[eid]['generations']
        else:
            nsqls = [['<dummy program>', 0.]]
        nsql_dict[eid] = {'nsqls': nsqls}

    # Split by processes
    nsql_dict_group = [dict() for _ in range(args.n_processes)]
    for idx, eid in enumerate(nsql_dict.keys()):
        nsql_dict_group[idx % args.n_processes][eid] = nsql_dict[eid]

    # Execute programs
    result_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        worker_results.append(pool.apply_async(worker_execute, args=(
            pid,
            args,
            dataset,
            nsql_dict_group[pid],
            keys
        )))

    # Merge worker results
    for r in worker_results:
        worker_result_dict = r.get()
        result_dict.update(worker_result_dict)
    pool.close()
    pool.join()
    n_correct_samples = 0
    for eid, item in result_dict.items():
        pred_answer, gold_answer = item['pred_answer'], item['gold_answer']
        n_correct_samples += Evaluator().evaluate(
            pred_answer,
            gold_answer,
            dataset=args.dataset,
            question=result_dict[eid]['question']
        )
    print(f'Overall Accuracy: {n_correct_samples}/{len(result_dict)}')

    # Save program executions
    with open(os.path.join(args.save_dir, args.output_program_execution_file), 'w') as f:
        json.dump(result_dict, f)

    print(f'Done. Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='tab_fact',
                        choices=['wikitq', 'tab_fact'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/prompts/qa_retrieve_pool.txt')
    parser.add_argument('--input_program_file', type=str,
                        default='binder_program_tab_fact_validation.json')
    parser.add_argument('--output_program_execution_file', type=str,
                        default='binder_program_execution_tab_fact_validation.json')

    # Multiprocess options
    parser.add_argument('--n_processes', type=str, default=4)

    # Execution options
    parser.add_argument('--use_majority_vote', action='store_false',
                        help='Whether use majority vote to determine the prediction answer.')
    parser.add_argument('--allow_none_and_empty_answer', action='store_true',
                        help='Whether regarding none and empty executions as a valid answer.')
    parser.add_argument('--allow_error_answer', action='store_true',
                        help='Whether regarding error execution as a valid answer.')
    parser.add_argument('--answer_placeholder', type=str, default='<error|empty>',
                        help='Placeholder answer if execution error occurs.')
    parser.add_argument('--vote_method', type=str, default='simple',
                        choices=['simple', 'prob', 'answer_biased'])
    parser.add_argument('--answer_biased', type=int, default=None,
                        help='The answer to be biased w. answer_biased_weight in majority vote.')
    parser.add_argument('--answer_biased_weight', type=float, default=None,
                        help='The weight of the answer to be biased in majority vote.')
    parser.add_argument('--process_program_with_fuzzy_match_on_db', action='store_false',
                        help='Whether use fuzzy match with db and program to improve on program.')

    # Debugging options
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
