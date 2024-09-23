import json
from tqdm import tqdm
import random
import uuid
from pathlib import Path
import glob
from .token_length import token_length
import io


file_handle_cache = {}

def close_cached_files():
    for file, handle in file_handle_cache.items():
        if isinstance(handle, io.IOBase):
            handle.close()
    file_handle_cache.clear()


def get_content(args, item, doc_name, idx):
    global file_handle_cache
    doc_type, doc_level = item['type'], item['level']
    docPath = Path(args.doc_path) / doc_type

    if doc_type == 'financial':
        if str(doc_level).strip() != '4':
            _file = glob.glob(f"{docPath}/*2024-{doc_name}*.txt")[0]
        else:
            _file = glob.glob(f"{docPath}/*{doc_name}*.txt")[0]
        try:
            with open(_file, 'r') as txt_file:
                _doc_name = Path(_file).stem.split('-')[-1]
                doc = f"《{_doc_name}》\n" + txt_file.read() + "\n\n"
        except IOError:
            print(f"Error: File {_file} could not be opened.")

    elif doc_type == 'paper':
        path = docPath / doc_name
        try:
            with open(path, 'r') as txt_file:
                content = txt_file.read()
                doc_name = content.split('\n', 1)[0].strip("#").strip()
                doc = f"{doc_name}\n" + content + "\n\n"
        except IOError:
            print(f"Error: File {path} could not be opened.")

    elif doc_type == 'legal':
        _file = docPath / "legal.json"
        if _file in file_handle_cache:
            legal_js = file_handle_cache[_file]
            # txt_file.seek(0)
        else:
            with open(_file, 'r') as txt_file:
                legal_js = json.load(txt_file)
                file_handle_cache[_file] = legal_js

        if doc_level == 4 and ('阅读以上判决文书，我将给你若干份判决结果：' in item['instruction']):
            content = legal_js[doc_name]["content"]
        else:
            content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]
        doc = f"《判决文书{idx + 1}》\n" + content + "\n\n"

    else:
        raise "doc_type not valid!"

    return doc


def get_contents(args, item, doc_names):
    contents = []
    for idx, doc_name in enumerate(doc_names):
        content = get_content(args, item, doc_name, idx)
        contents.append(content)
    return contents


def get_doc_str(args, item, prompt_template):
    len_prompt_template = token_length(prompt_template) - token_length("{docs}")
    is_shuffle = item.get("shuffle_doc", True)

    docs = item['doc'] if not args.rag else item["recall_chunks"][:args.rag_num]
    docs_list = []

    if args.rag:
        for doc in docs:
            if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(doc) > args.max_length:
                continue
            docs_list.append(doc)
    else:
        # read content from given doc names
        contents = get_contents(args, item, docs)
        # shuffle
        if is_shuffle and item['type'] == 'financial':
            random.shuffle(contents)
        for content in contents:
            if len_prompt_template + sum(token_length(s) for s in docs_list) + token_length(content) > args.max_length:
                continue
            docs_list.append(content)

    # shuffle
    if is_shuffle:
        random.shuffle(docs_list)
    docs_str = "".join(docs_list)
    return docs_str


def get_generate_prompt(args, item):
    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)
    doc_str = get_doc_str(args, item, prompt_template)
    prompt_template = prompt_template.replace("{docs}", doc_str)
    item['docs'] = doc_str
    item['prompt'] = prompt_template
    return item


def get_generate_prompts(args):
    prompts = []
    with open(args.input_path, 'r') as file:
        lines = file.readlines()

        if args.shuffle_prompts:
            random.shuffle(lines)
        # debug num samples
        if args.debug_num and args.debug_num > 0:
            lines = lines[:args.debug_num]
        if args.ratio != 1:
            random.shuffle(lines)
            lines = lines[int(len(prompts) * args.ratio):]

        for line in tqdm(lines, desc="gen_prompts"):
            item = json.loads(line)
            doc_type, set_level, level = item['type'], item['set'], item['level']
            # filter
            if args.domain.strip():
                domains = args.domain.strip().split(",")
                domains = list(map(lambda x: x.strip(), domains))
                if doc_type not in domains:
                    continue
            if args.debug_set.strip():
                sets = args.debug_set.strip().split(",")
                sets = list(map(int, sets))
                if set_level not in sets:
                    continue
            if args.debug_level.strip():
                levels = args.debug_level.strip().split(",")
                levels = list(map(int, levels))
                if level not in levels:
                    continue

            prompt = get_generate_prompt(args, item)
            prompts.append(prompt)
    close_cached_files()
    return prompts


def get_evaluate_prompts(args, tag):
    prompt = '''[Question]
{}

[Gold Answer]
{}

[The Start of Assistant's Predicted Answer]
{}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:'''
    prompts = []
    lines = open(args.output_path).readlines()
    for line in lines:
        line = json.loads(line.strip())
        line.pop('docs', '')
        doc_type, question, instruction = line['type'], line['question'], line['instruction']
        prompt_template = line['prompt_template']
        if doc_type != "paper":
            prompt_template = prompt_template.replace("{docs}", "")
        question = prompt_template.replace("{question}", question).replace("{instruction}", instruction)
        answer = line['answer']
        predict = line[tag]
        line['prompt'] = prompt.format(question, answer, predict)
        prompts.append(line)
    return prompts
