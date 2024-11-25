import os
import sys
sys.path.append("/mnt/data/haonan/code/mmevol_sft_data")
from base import BaseAPI
import numpy as np
from tqdm import tqdm
import multiprocessing
import os.path as osp
import requests
import cv2
import jsonlines
import os.path as osp
import shutil
import time
import json
from utils import data_process
from score_process import difficulty_scoring_v123
from uuid import uuid4
import base64

from prompt import df_prompt_format_latest_v1, df_prompt_cot_atom_latest_v1, bf_prompt_in_breath_latest_v1, ini_prompt_breath, ini_prompt_format, ini_prompt_cot


APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4o-2024-05-13': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-0613': 8192,
        'gpt-4-32k-0613': 32768,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-instruct': 4096,
        'gpt-3.5-turbo-0613': 4096,
        'gpt-3.5-turbo-16k-0613': 16385,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret

class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-4o-2024-05-13',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = 4096
        self.temperature = 1

        if 'step-1v' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        else:
            env_key = os.environ.get('OPENAI_API_KEY', '')
            if key is None:
                key = env_key
            # assert isinstance(key, str) and key.startswith('sk-'), (
            #     f'Illegal openai_key {key}. '
            #     'Please set the environment variable OPENAI_API_KEY to your openai key. '
            # )
        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if api_base is None:
            if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                print('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                api_base = os.environ['OPENAI_API_BASE']
            else:
                api_base = 'OFFICIAL'

        assert api_base is not None

        if api_base in APIBASES:
            self.api_base = APIBASES[api_base]
        elif api_base.startswith('http'):
            self.api_base = api_base
        else:
            print('Unknown API Base. ')
            sys.exit(-1)

        self.api_base="http://47.88.8.18:8088/api/ask"
        # self.api_base = "http://47.88.8.18:8088/api/ask?tenant=gpt-4o-mini"
        
        # m6
        # self.key = "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImZhbnpoaWhhby5memhAYWxpYmFiYS1pbmMuY29tIiwicGFzc3dvcmQiOiIwOGU3NTk1ZjgyYTk4ZGY2NDRjMmI0NDM4NzM1Y2Y4Y2U0NDBmMWNjIiwiZXhwIjoyMDA5NjA5NTM4fQ.CmIJOx7fvERV2PP7eQ3sZVLhtO1aRB2B5DU7BIETVC8"
        
        # coai
        # self.key = "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjI1ODczMCIsInBhc3N3b3JkIjoiMjU4NzMwMTIzIiwiZXhwIjoyMDE5NTUwNzAxfQ.JuqnTa7yauGkSzWkBiEig1K_rxvfAYTXS9F9_m-h4q8" 
        
        # norm
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJ1c2VybmFtZSI6IjQ0MzQ1NSIsInBhc3N3b3JkIjoiNDQzNDU1MTIzIiwiZXhwIjoyMDMxNzA1NTA3fQ.7g4a6t9dKcRXVRa7MwQb5m2oirFu1OxjXhWbNM0w50s"
        
        # self.model="gpt-4o-2024-08-06"
        self.model = "gpt-4o-mini"
        print(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_inputs(self, inputs):
        input_msgs = []
        self.system_prompt = "You are GPT-4, a large language model trained by OpenAI."
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
            input_msgs.append(dict(role='user', content=content_list))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            input_msgs.append(dict(role='user', content=text))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            print(
                'Less than 100 tokens left, '
                'may exceed the context window with some additional meta symbols. '
            )
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            seed=3407,
            # tenant='coai',
            # response_format="json_object",
            **kwargs)
        response = requests.post(self.api_base, headers=headers, data=json.dumps(payload), timeout=self.timeout * 11)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['data']['response']['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except:
            enc = tiktoken.encoding_for_model('gpt-4')
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += 85
                if self.img_detail == 'high':
                    img = Image.open(item['value'])
                    npatch = np.ceil(img.size[0] / 512) * np.ceil(img.size[1] / 512)
                    tot += npatch * 170
        return tot

class GPT4V(OpenAIWrapper):
    
    def generate(self, task, dataset=None):
        message, path, raw_path, evo_direction = task
        with open(path, 'w') as file:
            file.write(super(GPT4V, self).generate(message))
        json.dump(evo_direction, open(raw_path, "w"), indent=4)

def boxes2str(boxes):
    res = ''
    for i, box in enumerate(boxes):
        # if box['confidence'] >= 0.1:
        if i > 0:
            res += '\n'
        bbox_str = ', '.join(f"{coord:.3f}" for coord in box['bbox'])
        res = res + f"{box['label']}: [{bbox_str}]"
    return res

def evolution_parallel(data_path, img_path, gen_save_path, raw_save_path, round_n, root_path):    
    
    # Read Data
    # with open(data_path, 'r') as file:
    #     data = json.load(file)
    meta_info_list = os.listdir(data_path)
    
    run_idx = ['0_34140_tabmwp_tables_34140.png']
    c = 0
    messages, paths, raw_paths, evo_direction = [], [], [], []
    prompt_candidate = [df_prompt_format_latest_v1, df_prompt_cot_atom_latest_v1, bf_prompt_in_breath_latest_v1]
    # prompt_candidate = [ini_prompt_format, ini_prompt_cot, ini_prompt_breath]
    
    for meta_file in tqdm(meta_info_list, desc="Collect-Data-Round-{}".format(round_n)):
        
        if c>=50:
            break
        c+=1

        d = json.load(open(os.path.join(data_path, meta_file)))
        hash_id, fig, boxes = d['hash_id'], d['image'], boxes2str(d['det'])
        # if hash_id not in run_idx:
        #     continue
        caption = [] if len(d['caption']) == 0 else d['caption'][1]['value'].lower()
        fig = os.path.join(img_path, fig)
        
        if 'conversations_v2' in d and len(d['conversations_v2']) != 0:
            qa = d['conversations_v2']
        elif 'conversations_v1' in d and len(d['conversations_v1']) != 0:
            qa = d['conversations_v1']
        else:
            qa = d['conversations_v0']

        try:
            json.load(open(osp.join(root_path, "round{}/gen_qa_corrected/{}.json".format(round_n, hash_id)), "r"))
            continue

        except:
            values = {
                'given_paragraph': caption,
                'given_location': boxes,
                'given_qa': qa
            }
            prompt= np.random.choice(prompt_candidate, size=1, p=[0.25, 0.25, 0.5])[0]

            if prompt_candidate.index(prompt) == 0:
                evo_direction.append("df_format")
            elif prompt_candidate.index(prompt) == 1:
                evo_direction.append("df_cot")
            elif prompt_candidate.index(prompt) == 2:
                evo_direction.append("bf_in_breath")
            else:
                evo_direction.append("error")

            prompt = prompt.format(**values)

            message = [
                {"type": "image", "value": fig},
                {"type": "text", "value": prompt}
            ]

            messages.append(message)
            paths.append(gen_save_path.format(hash_id))
            raw_paths.append(raw_save_path.format(hash_id))
            
    gpt4v = GPT4V()
    tasks = list(zip(messages, paths, raw_paths, evo_direction))

    with multiprocessing.Pool(processes=10) as pool:
        for _ in tqdm(pool.imap(gpt4v.generate, tasks), total=len(messages), desc="Evo-Round-{}".format(round_n)):
            pass
    
    return messages

def filter_round1(meta_data, conversation_v1_path):

    meta_data_files = os.listdir(meta_data)

    if os.path.exists(osp.join(conversation_v1_path, "filtered_qa")):
        shutil.rmtree(osp.join(conversation_v1_path, "filtered_qa"))
        os.mkdir(osp.join(conversation_v1_path, "filtered_qa"))

    count = 0
    for mdf in tqdm(meta_data_files, desc="Filter-Round-1"):

        meta_sample = json.load(open(osp.join(meta_data, mdf), "r"))
        qa_path = osp.join(conversation_v1_path, "gen_qa_corrected", mdf)
        score_path = osp.join(conversation_v1_path, "score_gpt4_mini_corrected", mdf)
        
        try:
            data = json.load(open(qa_path, "r"))
            score = json.load(open(score_path, "r"))

            temp = []
            for index, i in enumerate(data):
                if score[index]['improved'] == "yes":
                    i["score"] =  score[index]['score']
                    temp.append(i)
            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": temp,
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v1_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
            count += 1
        except:
            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": [],
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v1_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
    print("{} of {} are evolved".format(count, len(meta_data_files)))

def filter_round2(meta_data, conversation_v2_path):

    meta_data_files = os.listdir(meta_data)
    if os.path.exists(osp.join(conversation_v2_path, "filtered_qa")):
        shutil.rmtree(osp.join(conversation_v2_path, "filtered_qa"))
        os.mkdir(osp.join(conversation_v2_path, "filtered_qa"))

    count = 0
    for mdf in tqdm(meta_data_files, desc="Filter-Round-2"):

        meta_sample = json.load(open(osp.join(meta_data, mdf), "r"))
        qa_path = osp.join(conversation_v2_path, "gen_qa_corrected", mdf)
        score_path = osp.join(conversation_v2_path, "score_gpt4_mini_corrected", mdf)
        
        try:
            data = json.load(open(qa_path, "r"))
            score = json.load(open(score_path, "r"))
            temp = []
            for index, i in enumerate(data):
                if score[index]['improved'] == "yes":
                    i["score"] =  score[index]['score']
                    temp.append(i)

            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": meta_sample["conversations_v1"],
                "conversations_v2": temp,
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v2_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
            count += 1
        except:
            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": meta_sample["conversations_v1"],
                "conversations_v2": [],
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v2_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
    print("{} of {} is evolved".format(count, len(meta_data_files)))

def filter_round3(meta_data, conversation_v3_path):

    meta_data_files = os.listdir(meta_data)

    if os.path.exists(osp.join(conversation_v3_path, "filtered_qa")):
        shutil.rmtree(osp.join(conversation_v3_path, "filtered_qa"))
        os.mkdir(osp.join(conversation_v3_path, "filtered_qa"))

    count = 0

    for mdf in tqdm(meta_data_files, desc="Filter-Round-3"):

        meta_sample = json.load(open(osp.join(meta_data, mdf), "r"))
        qa_path = osp.join(conversation_v3_path, "gen_qa_corrected", mdf)
        score_path = osp.join(conversation_v3_path, "score_gpt4_mini_corrected", mdf)
        
        try:
            data = json.load(open(qa_path, "r"))
            score = json.load(open(score_path, "r"))
            temp = []
            for index, i in enumerate(data):
                if score[index]['improved'] == "yes":
                    i["score"] =  score[index]['score']
                    temp.append(i)

            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": meta_sample["conversations_v1"],
                "conversations_v2": meta_sample["conversations_v2"],
                "conversations_v3": temp,
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v3_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
            count += 1
        except:
            d = {
                "id": meta_sample["id"],
                "image": meta_sample["image"],
                "conversations_v0": meta_sample["conversations_v0"],
                "conversations_v1": meta_sample["conversations_v1"],
                "conversations_v2": meta_sample["conversations_v2"],
                "conversations_v3": [],
                "topic_name": meta_sample["topic_name"] if "topic_name" in meta_sample else [],
                "caption": meta_sample["caption"],
                "det": meta_sample["det"],
                "hash_id": meta_sample["hash_id"]
            }
            json.dump(d, open(osp.join(conversation_v3_path, "filtered_qa", "{}.json".format(meta_sample['hash_id'])), "w"), indent=4)
    print("{} of {} is evolved".format(count, len(meta_data_files)))


if __name__=='__main__':

    final_save_path = "/mnt/data/haonan/code/mmevol_sft_data/datasets/seed_data_1k_demo_evo.json"
    root_path = '/mnt/data/haonan/code/mmevol_sft_data/evolution/multi_round_single_imgs_1k_mini'
    img_path = '/mnt/workspace/lr/datasets'

    for round_n in [1,2,3]:
        if round_n == 1: 
            seed_data_path = "/mnt/data/haonan/code/mmevol_sft_data/datasets/meta_data"
        else:
            seed_data_path = osp.join(root_path, "round{}".format(round_n-1), "filtered_qa")
        
        round_path = osp.join(root_path, "round{}".format(round_n))
        gen_save_path = osp.join(round_path, "gen_qa/{}.json")
        raw_save_path = osp.join(round_path, "evo_path/{}.json")
        
        # patience = 0
        # while True:
            
        #     # evol
        #     num_messages = evolution_parallel(seed_data_path, img_path, gen_save_path, raw_save_path=raw_save_path, round_n=round_n, root_path=root_path)

        #     # post-process
        #     data_process.func_4_qa(
        #         path = round_path,
        #         data_path = osp.join(round_path, "gen_qa"),
        #         data_path_corrected = osp.join(round_path, "gen_qa_corrected"),
        #         round_n = round_n
        #     )
        #     patience += 1
        #     if len(num_messages) < 50 or patience >= 5:
        #         print("Round: {} QA Evo Finished".format(round_n))
        #         break
            
        patience = 0
        while True:
            # score
            scores = difficulty_scoring_v123.score_parallel(seed_data_path, osp.join(root_path, "round{}".format(round_n), "score_gpt4_mini/{}.json"), round_n, root_path)
            # score-process
            data_process.func_4_score(
                path=round_path, 
                data_path=osp.join(round_path, "score_gpt4_mini"), 
                data_path_corrected=osp.join(round_path, "score_gpt4_mini_corrected"),
                round_n=round_n
            )
            patience += 1
            if len(scores) < 50 or patience >= 5:
                print("Round: {} Score Finished".format(round_n))
                break

        # filter
        if round_n == 1:
            filter_round1(seed_data_path, round_path)
        elif round_n == 2:
            filter_round2(seed_data_path, round_path)
        else:
            filter_round3(seed_data_path, round_path)
        
    # merge
    assert round_n == 3
    p = osp.join(round_path, "filtered_qa")
    data_file_list = os.listdir(p)

    merged_data = []
    for data_file in tqdm(data_file_list, desc="Merge-Final-Data"):
        data_file_path = osp.join(p, data_file)
        data = json.load(open(data_file_path, "r"))
        merged_data.append(data)
    
    json.dump(merged_data, open(final_save_path, "w"), indent=4)
    print("Saveing file to {}".format(final_save_path))