import json
import os
import os.path as osp
import re
from tqdm import tqdm
import shutil
import multiprocessing

# 匹配[]，删除额外的文字
def extract_nested_brackets(text):
    start_idx = text.find('[')
    if start_idx == -1:
        return None  

    bracket_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return text[start_idx:i+1]
    
    return None

# 处理multi-choice中多次换行导致json无法读取
def format_multiline_fields(text):
    fields = ["answer", "question"]

    for field in fields:
        pattern = r'("{}":\s*")(.*?)(?<!\\)"'.format(field)
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            original_content = match[1]
            formatted_content = original_content.replace('\n', '\\n').replace('                        ', '')

            full_match = '"{}": "{}"'.format(field, original_content)
            formatted_match = '"{}": "{}"'.format(field, formatted_content)
            text = text.replace(full_match, formatted_match)
    return text

# 删除注释
def remove_comments(text):
    # 使用正则表达式移除所有注释
    return re.sub(r'//.*$', '', text, flags=re.MULTILINE)

def remove_answer_comma(text):
    cleaned_content = re.sub(r'("answer":\s*"[^"]*"),\s*(?=\n\s*["}])', r'\1', text, flags=re.MULTILINE)
    return cleaned_content

def prefix_del(path, path_corrected):
    # delete ```json and ``` str.
    for filename in tqdm(os.listdir(path), desc="Data Processing...", total=len(os.listdir(path))):
        file_path = os.path.join(path, filename)
        try:
            json.load(open(osp.join(path_corrected, filename), "r"))
            continue
        except:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            content = re.sub(r'.*```json', '', content, flags=re.DOTALL)
            content = re.sub(r'```.*', '', content, flags=re.DOTALL)
            # cleaned_content = content.replace("```json", "").replace("```", "").strip()

            # 移除 [ 前的英文字母和换行，以及 ] 后的英文字母和换行
            # 去除开头多余的字母和换行符
            content = re.sub(r'^[A-Za-z\s]*\n*$$', '[', content, flags=re.DOTALL)

            # 去除结尾多余的换行符和字母
            cleaned_content = re.sub(r']\n*[A-Za-z\s]*$', ']', content, flags=re.DOTALL)

            file_path_corrected = os.path.join(path_corrected, filename)
            with open(file_path_corrected, 'w', encoding='utf-8') as file:
                file.write(cleaned_content.strip())

def prefix_del_parallel(task):
    # delete ```json and ``` str.
    path, path_corrected, filename = task
    # for filename in tqdm(os.listdir(path), desc="Data Processing...", total=len(os.listdir(path))):
    file_path = os.path.join(path, filename)
    try:
        json.load(open(osp.join(path_corrected, filename), "r"))
    except:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        content = re.sub(r'.*```json', '', content, flags=re.DOTALL)
        content = re.sub(r'```.*', '', content, flags=re.DOTALL)
        # cleaned_content = content.replace("```json", "").replace("```", "").strip()

        # 移除 [ 前的英文字母和换行，以及 ] 后的英文字母和换行
        # 去除开头多余的字母和换行符
        content = re.sub(r'^[A-Za-z\s]*\n*$$', '[', content, flags=re.DOTALL)

        # 去除结尾多余的换行符和字母
        cleaned_content = re.sub(r']\n*[A-Za-z\s]*$', ']', content, flags=re.DOTALL)

        file_path_corrected = os.path.join(path_corrected, filename)
        with open(file_path_corrected, 'w', encoding='utf-8') as file:
            file.write(cleaned_content.strip())

# 逐字段判断
def field_process(path):
    # field corrections
    try:
        flag = False
        with open(path, "r") as file:
            data = json.load(file)
        
        # 1. data is not list
        if not isinstance(data, list):
            data = [data]
            flag = True
        for index, d in enumerate(data):
            # 2. answers -> answer 
            if "answer" not in d and "answers" in d:
                d['answer'] = d.pop('answers')
                flag = True

            # 3. answer field is not a str but list/dict
            if isinstance(d["answer"], list):
                d['answer'] = ' '.join(str(s) for s in d['answer'])
                flag = True

            if isinstance(d['answer'], dict):
                d['answer'] = ' '.join(str(v) for k, v in d['answer'].items())
                flag = True
  
        json.dump(data, open(path, "w"), indent=4) if flag else None
    
    except:

        with open(path, 'r', encoding='utf-8') as file:
            data = file.read().strip()
        # print(file)
        if data != "" and data[0] != "[":
            data = "[\n" + data + "\n]" 

        data = re.sub(r'\},\s+\]', '}]', data).replace('”', '"').replace('“', '"').replace('"""', '"')
        data = format_multiline_fields(data)
        data = remove_comments(data)
        data = remove_answer_comma(data)
        # match = extract_nested_brackets(data)
        # data = match if match is not None else data

        try:
            data = json.loads(data)
            json.dump(data, open(path, "w"), indent=4)
        except:
            pass
        
        # pass

def check_json_format(path):
    # type check
    try:
        with open(path, "r") as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            return False, "JSON data is not a list"
        
        for item in data:
            if not isinstance(item, dict):
                return False, "One of the items in the JSON data is not a dictionary"
            
            required_fields = ["objects", "skills", "format", "question", "steps", "answer"]

            for field in required_fields:
                if field not in item:
                    return False, f"Missing required field: {field}"

            for field in required_fields:
                if field in ["format", "question", "answer"] and not isinstance(item[field], str):
                    return False, f"The 'answer' field is not a str: {item['answer']}"

                if field in ["objects", "skills", "steps"] and not isinstance(item[field], list):
                    return False, f"The 'steps' field is not a list: {item['question']}"
                
                if field == "steps":
                    for step in item[field]:
                        if not isinstance(step, dict):
                            return False, f"One of the steps is not a valid tuple: {item['question']}"
                
        return True, "JSON data is valid"
    
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {str(e)}"


def wrapper_function(task):
    current_path = task
    # 处理前check
    correct, msg = check_json_format(current_path)
    # [pre, post]= ["\033[1;32m", "\033[0m"] if correct else ["\033[1;31m", "\033[0m"] # ["", ""]
    # print(pre+msg+post, "|", current_path) # if not correct else None
    field_process(current_path) if not correct else None

    # 处理后check
    correct, msg = check_json_format(current_path)
    # correct_sample += 1 if correct else 0
    # [pre, post]= ["\033[1;32m", "\033[0m"] if correct else ["\033[1;31m", "\033[0m"] # ["", ""]
    # print(pre+msg+post, "|", current_path) # if not correct else None


def func_4_score(path, data_path, data_path_corrected, round_n=None):
    
    data_path = osp.join(path, data_path)
    data_path_corrected = osp.join(path, data_path_corrected)

    files = os.listdir(data_path)

    tasks = list(zip([data_path]*len(files), [data_path_corrected]*len(files), files))
    with multiprocessing.Pool(processes=100) as pool:
        for _ in tqdm(pool.imap(prefix_del_parallel, tasks), total=len(tasks), desc="Post-Process-Score-Round{}".format(round_n)):
            pass

    # prefix_del(data_path, data_path_corrected)

def func_4_qa(path, data_path, data_path_corrected, round_n=None):

    data_path = osp.join(path, data_path)
    data_path_corrected = osp.join(path, data_path_corrected)

    files = os.listdir(data_path)

    tasks = list(zip([data_path]*len(files), [data_path_corrected]*len(files), files))
    with multiprocessing.Pool(processes=100) as pool:
        for _ in tqdm(pool.imap(prefix_del_parallel, tasks), total=len(tasks), desc="Post-Process-QA-Round{}".format(round_n)):
            pass

    paths = []
    for json_data in os.listdir(data_path_corrected):
        paths.append(os.path.join(data_path_corrected, json_data))

    with multiprocessing.Pool(processes=100) as pool:
        for _ in tqdm(pool.imap(wrapper_function, paths), total=len(paths)):
            pass

    # prefix_del(data_path, data_path_corrected)

    # correct_sample = 0
    # for json_data in os.listdir(data_path_corrected):
    #     current_path = os.path.join(data_path_corrected, json_data)

    #     # 处理前check
    #     correct, _ = check_json_format(current_path)
    #     # [pre, post]= ["\033[1;32m", "\033[0m"] if correct else ["\033[1;31m", "\033[0m"] # ["", ""]
    #     # print(pre+msg+post, "|", current_path) # if not correct else None
    #     field_process(current_path) if not correct else None

    #     # 处理后check
    #     correct, msg = check_json_format(current_path)
    #     correct_sample += 1 if correct else 0
    #     [pre, post]= ["\033[1;32m", "\033[0m"] if correct else ["\033[1;31m", "\033[0m"] # ["", ""]
    #     # print(pre+msg+post, "|", current_path) # if not correct else None
            
    # # # json_transfer(data_path)
    # print("Correct ratio: {:.8f}".format(correct_sample / len(os.listdir(data_path_corrected))))

if __name__ == '__main__':
    
    # path = "/mnt/workspace/workgroup/haonan/evolution-code/evolution/single_imgs/multi_round_v1_single_imgs_persona_top50_gpt4o_mini/round1"
    path = "/mnt/workspace/workgroup/haonan/evolution-code/evolution/single_imgs/multi_round_v1_single_imgs_science_30k_mini/round1"
    data_path = osp.join(path, "gen_qa")
    data_path_corrected = osp.join(path, "gen_qa_corrected")
    # data_path = osp.join(path, "ini_object_v1_single_img_153k")
    # data_path_corrected = osp.join(path, "ini_object_v1_single_img_153k_corrected")

    files = os.listdir(data_path)

    tasks = list(zip([data_path]*len(files), [data_path_corrected]*len(files), files))
    with multiprocessing.Pool(processes=100) as pool:
        for _ in tqdm(pool.imap(prefix_del_parallel, tasks), total=len(tasks)):
            pass

    # if os.path.exists(data_path_corrected):
    #     shutil.rmtree(data_path_corrected)
    #     os.mkdir(data_path_corrected)

    if "score_" not in data_path_corrected:
        correct_sample = 0
        paths = []
        for json_data in tqdm(os.listdir(data_path_corrected)):
            paths.append(os.path.join(data_path_corrected, json_data))

        tasks = paths
        with multiprocessing.Pool(processes=100) as pool:
            for _ in tqdm(pool.imap(wrapper_function, tasks), total=len(tasks)):
                pass
        
