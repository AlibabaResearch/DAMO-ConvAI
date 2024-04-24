# this file remove the instance field from user oriented instructions and add the input field to the instruction field
from io_utils import load_jsonlines
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="datasets/self-instruct-val", type=str)
parser.add_argument("--save_file_name", default="datasets/self-instruct-val(processed)", type=str)
args = parser.parse_args()

tasks = load_jsonlines(args.file_path)
tasks_processed = []

for task in tasks:
    if 'instances' in task:
        task_input = task['instances'][0]['input']
        if task_input:
            task['instruction'] += f"\n\n### Input:\n{task_input}"
        del task['instances']
    elif "context" in task:
        context = task['context']
        if context:
            task['instruction'] += f"\n\n### Input:\n{task['context']}"
        del task['context']
    
    tasks_processed.append(task)

Dataset.from_list(tasks_processed).to_json(args.save_file_name)