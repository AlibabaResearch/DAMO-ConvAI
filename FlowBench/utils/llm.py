import concurrent
import threading
from string import Template
import numpy as np
import pandas as pd
import json
import math
import pickle
import os, re
import sys
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import openai
import random
import logging
import time
import traceback
import regex
from utils.request_openai import request_openai_standard


model_type = "gpt-4-1106-preview"
class GPT:
    token_count = Counter()
    def __init__(self, model_name = "gpt-4-0125-preview", max_retry=20, name=None):
        keys_file = os.path.join(os.path.dirname(__file__), 'keys.json')
        self.keys = json.load(open(keys_file))
        self.max_retry = max_retry
        self.name = name
        self.model_name = model_name

    def record_token_usage(self, response):
        model = response['model']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        GPT.token_count[f'token_in_{model}'] += prompt_tokens
        GPT.token_count[f'token_out_{model}'] += completion_tokens
        if self.name is not None:
            GPT.token_count[f'{self.name}.token_in_{model}'] += prompt_tokens
            GPT.token_count[f'{self.name}.token_out_{model}'] += completion_tokens

    def record_token_usage_http(self, response):
        model = self.model_name
        if "data" in response and "promptTokens" in response["data"]:
            GPT.token_count[f'token_in_{model}'] += response["data"]["promptTokens"]
            GPT.token_count[f'token_out_{model}'] += response["data"]["completionTokens"]

    def infer_single_turn(self, user, system=None, model=model_type, temperature=None):
        msg_list = [{'role': 'user', 'content': user}]
        if system is not None:
            msg_list.insert(0, {'role': 'system', 'content': system})
        for i in range(self.max_retry):
            try:
                if temperature is None:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=msg_list
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=msg_list,
                        temperature=temperature
                    )
                response_message = response["choices"][0]["message"]
                self.record_token_usage(response)
                response_message = json.loads(json.dumps(response_message, ensure_ascii=False))
                text = response_message['content']
                time.sleep(0.1)
                return text
            except Exception as e:
                print('Error', e)
                traceback.print_exc()
                print('Retrying...')
                time.sleep(3)
        raise ValueError('LLM error')

    def infer_multi_turn_with_functions(self, messages, functions=None, model=model_type, temperature=0.):
        messages = [dict(m) for m in messages]
        for i in range(self.max_retry):
            try:
                openai.api_key = random.choices(self.keys)[0]
                if functions is not None:
                    response = request_openai_standard(self.model_name,messages, temperature, functions)
                else:
                    response = request_openai_standard(self.model_name,messages, temperature)
                message = response['choices'][0]
                if "Action:" in message['message']["content"] and "Action Input:" in message['message']["content"]:
                    response_message = {'role': 'assistant', 'content': message['message']["content"]}
                    action_match = re.search(r"Action: (?:functions\.)?(\w+)", message['message']["content"])
                    input_match = regex.search(r"Action Input: (\{(?:[^{}]*|(?R))*\})", message['message']["content"])
                    tmptext = re.sub(r'Response:.*?(\n|$)', '',   response_message["content"], flags=re.DOTALL)
                    first_thought_end = tmptext.find('Thought:') + tmptext[tmptext.find('Thought:'):].find('\n') + 1
                    response_message["content"] = tmptext[:first_thought_end] + re.sub(r'Thought:.*?(\n|$)', '', tmptext[first_thought_end:], 1, flags=re.DOTALL)                            
                    action = action_match.group(1) if action_match else None
                    action_input = input_match.group(1) if input_match else None
                    if action is None or action_input is None:
                        print("API error")
                    response_message["function_call"] = {"name": action.replace("functions.",""),"arguments": action_input}
                else:
                    response_message = {'role': 'assistant', 'content': message['message']["content"]}

                use_tool = bool(response_message.get('function_call'))
                if use_tool:
                    try:
                        response_message['function_call']
                    except Exception as err:
                        raise ValueError('function_call not parsed')
                time.sleep(0.2)
                return response_message
            except Exception as e:
                print('Error', e)
                traceback.print_exc()
                print('Retrying...')
                time.sleep(2)
        raise ValueError('LLM error')

