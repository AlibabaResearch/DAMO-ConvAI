import json
import time
import logging
from typing import List, Dict, Union, Any
import requests
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError

from .base import LMAgent

logger = logging.getLogger("agent_frame")


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


class FastChatAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.controller_address = config["controller_address"]
        self.thought_model_name = config["thought_model_name"]
        self.thought_model_temperature = config.get("thought_model_temperature", 0)
        self.thought_model_top_p = config.get("thought_model_top_p", 0)
        self.action_model_name = config["action_model_name"]
        self.action_model_temperature = config.get("action_model_temperature", 0)
        self.action_model_top_p = config.get("action_model_top_p", 0)
        self.max_new_tokens = config.get("max_new_tokens", 512)

    def __call_thought__(self, messages: List[dict]) -> str:
        controller_addr = self.controller_address
        worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.thought_model_name,
            "temperature": self.thought_model_temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.thought_model_top_p,
        }
        conv = get_conversation_template(self.thought_model_name)
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        new_stop = set()
        _add_to_set(self.stop_words, new_stop)
        _add_to_set(conv.stop_str, new_stop)
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(new_stop),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            assert False, data["text"]
                        text = data["text"]
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")
    
    
    def __call_action__(self, messages: List[dict]) -> str:
        controller_addr = self.controller_address
        worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.action_model_name,
            "temperature": self.action_model_temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.action_model_top_p,
        }
        conv = get_conversation_template(self.action_model_name)
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        new_stop = set()
        _add_to_set(self.stop_words, new_stop)
        _add_to_set(conv.stop_str, new_stop)
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(new_stop),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            assert False, data["text"]
                        text = data["text"]
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")
