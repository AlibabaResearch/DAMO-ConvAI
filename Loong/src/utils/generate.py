import json
from tqdm import tqdm
import multiprocessing
import requests
import numpy as np
from functools import partial
from decimal import Decimal
import numpy as np
import time
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            try:
                return str(obj, encoding='utf-8')
            except:
                return str(obj, encoding='gbk')
        elif isinstance(obj, Decimal):
            return float(obj)
        # print(obj, type(obj))
        return json.JSONEncoder.default(self, obj)


def get_api_results(prompt_input, config):
    prompt = prompt_input['prompt']

    if config['type'] == 'openai' or config['type'] == 'vllm':
        client = OpenAI(api_key=config['args']['api_key'],
                        base_url=config['args']['api_url'] if config['args']['api_url']!='' else None)
        try: 
            response = client.chat.completions.create(
                messages=[{"role": "user","content": prompt}],
                model=config['args']['api_name'],
                temperature=config['run_args']['temperature']
            )
            return response.choices[0].message.content
        except Exception as e:
                print(e)
                return []
        
    elif config['type'] == 'gemini':
        genai.configure(api_key=config['args']['api_key'])

        model = genai.GenerativeModel(name=config['args']['api_name'])
        try:
            response = model.generate_content(prompt,
                        generation_config=genai.types.GenerationConfig(
                        temperature=config['run_args']['temperature']))
            return response.text
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'claude':
        client = Anthropic(api_key=config['args']['api_key'])
        try:
            message = client.messages.create(
                messages=[{"role": "user", "content": prompt,}],
                model=config['args']['api_name'],
            )
            return message.content
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'http':
        headers = {"Content-Type": "application/json",
                "Authorization": config['args']['api_key']}
        raw_info = {
            "model": config['args']['api_name'],
            "messages": [{"role": "user", "content": prompt}],
            "n": 1}
        raw_info.update(config['run_args'])
        try:
            callback = requests.post(config['args']['api_url'], data=json.dumps(raw_info, cls=MyEncoder), headers=headers,
                                    timeout=(600, 600))
            result = callback.json()
            # todo: customize the result
            return result['data']['response']['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            return []
        
    else:
        raise f"type of {config['type']} is not valid"

def fetch_api_result(prompt_input, config, max_retries=5):
    """Attempt to get a valid result from the API, with a maximum number of retries."""
    for _ in range(max_retries):
        result = get_api_results(prompt_input, config)
        if result: 
            return result
        # Sleep briefly to not hammer the API in case of errors or rate limits
        time.sleep(5) # Uncomment if needed
    return None


def api(prompt, output_path, config, tag):
    response_content = fetch_api_result(prompt, config)
    result = prompt.copy()
    result[tag] = response_content or ""
    with open(output_path, 'a', encoding='utf-8') as fw:
        fw.write(json.dumps(result, ensure_ascii=False) + '\n')


def generate(prompts, config, output_path, process_num, tag):
    func = partial(api, output_path=output_path, config=config, tag=tag)
    with multiprocessing.Pool(processes=process_num) as pool:
        for _ in tqdm(pool.imap(func, prompts), total=len(prompts)):
            pass