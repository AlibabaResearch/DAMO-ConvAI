import sys
sys.path.append("/mnt/workspace/workgroup/haonan/evolution-code")
import os
import json 
import numpy as np
import multiprocessing
import sys
import os.path as osp
import requests
import cv2
import time
import pprint
import base64
from tqdm import tqdm
from utils import data_process
from uuid import uuid4
from base import BaseAPI
from prompt_score import difficuty_score_prompt_latest_v1, difficuty_score_prompt_latest_v2, difficuty_score_prompt_latest_v3, difficuty_score_prompt_latest_v4


APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4o-mini': 128000,
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
        # self.key = "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjI1ODczMCIsInBhc3N3b3JkIjoiMjU4NzMwMTIzIiwiZXhwIjoyMDE5NTUwNzAxfQ.JuqnTa7yauGkSzWkBiEig1K_rxvfAYTXS9F9_m-h4q8"
        # self.key = "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjI3NDM2OCIsInBhc3N3b3JkIjoiMjc0MzY4MTIzIiwiZXhwIjoyMDEyNjEzNjA4fQ.7OUpHs-AFPaFHuUy_p7XxXyNYhca2_-7F5GBtaahfe4"
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJ1c2VybmFtZSI6IjQ0MzQ1NSIsInBhc3N3b3JkIjoiNDQzNDU1MTIzIiwiZXhwIjoyMDMxNzA1NTA3fQ.7g4a6t9dKcRXVRa7MwQb5m2oirFu1OxjXhWbNM0w50s"
        # self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJ1c2VybmFtZSI6IjQzOTg2OSIsInBhc3N3b3JkIjoiNDM5ODY5MTIzIiwiZXhwIjoyMDMxNzA3NjkzfQ.ly9XNzVW7pEeW_bTZxzaqB3jt2kRr14XQIpT0DbCTto"
        # self.model = "gpt-4o-2024-08-06"
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
        message, path = task
        with open(path, 'w') as file:
            file.write(super(GPT4V, self).generate(message))

def score_parallel(data_path, save_path, round_n, root_path):    
    # Read Data
    data_list = json.load(open(data_path, "r"))
    # run_idx = ['000000203269', '000000000540']
    
    messages, paths, raw_paths = [], [], []

    for idx, data in data_list.items():
        
        idx = idx + ".json"
        given_qa = []
        rewritten_qa = []

        tmp = data['conversation_v0']['conversations_v0']
        
        for index in tmp:
            given_qa.append({
                "question": index['question'],
                "answer": index['answer']
            })
        rewritten_qa = given_qa
        # try:
        #     ss_ = json.load(open(os.path.join(root_path, "round{}/score_gpt4_mini_corrected".format(round_n), idx), "r"))
        #     assert len(ss_) == len(gen_info)
        #     continue
        # except:
        values = {
            'given_qa': str(given_qa),
            'rewritten_qa': str(rewritten_qa)
        }

        prompt = difficuty_score_prompt_latest_v4.format(**values)

        message = [
            {"type": "text", "value": prompt}
        ]

        messages.append(message)
        paths.append(save_path.format(idx.replace(".json", "")))
    
    gpt4v = GPT4V()
    tasks = list(zip(messages, paths))
    with multiprocessing.Pool(processes=50) as pool:
            for _ in tqdm(pool.imap(gpt4v.generate, tasks), total=len(messages), desc="Score-Round-{}".format(round_n)):
                pass
    return messages

def socre(data_path, round_n, root_path):

    # 存储路径
    save_path = osp.join(root_path, "round{}".format(round_n), "score_gpt4_mini/{}.json")
    messages = score_parallel(data_path, save_path, round_n=round_n, root_path=root_path) # round_n
    return messages

if __name__=='__main__':
    # start_time = time.time()
    root_path = "/mnt/workspace/workgroup/haonan/evolution-code/evolution/single_imgs/multi_round_v1_single_imgs_ini_prompt_v0_score"
    round_n = 0

    data_path = "/mnt/workspace/workgroup/haonan/evolution-code/datasets/improv_score_reason_old.json"
    # 存储路径
    save_path = osp.join(root_path, "round{}".format(round_n), "score_gpt4_mini/{}.json")
    score_parallel(data_path, save_path, round_n=round_n, root_path=root_path) # round_n

    # 数据后处理
    tmp = osp.join(root_path, "round{}".format(round_n))
    data_process.func_4_score(
        path=tmp, 
        data_path=osp.join(tmp, "score_gpt4_mini"), 
        data_path_corrected=osp.join(tmp, "score_gpt4_mini_corrected")
    )
    # 过滤
