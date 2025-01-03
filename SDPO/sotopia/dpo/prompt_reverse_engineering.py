import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, cast
import transformers
import pandas as pd
import rich
from rich.console import Console
from rich.terminal_theme import MONOKAI

from sotopia.database.logs import EpisodeLog
from sotopia.messages.message_classes import ActionType
import numpy as np
import json

# PROMPT_PREFIX = "Prompt after formatting:\n"
MAX_TOKEN = 2048  # 5000

PROMPT_TEMPLATE = """Prompt after formatting:\nImagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
Note that {agent}'s secret and goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
{history}.
You are at Turn #{turn_number}."""

# PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

FORMAT_TEMPLATE = """ Your available action types are
"none action speak non-verbal communication leave".
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}
the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.
\nHere is the output schema:\n```\n{\"description\": \"An interface for messages.\\nThere is only one required method: to_natural_language\", \"properties\": {\"action_type\": {\"title\": \"Action Type\", \"description\": \"whether to speak at this turn or choose to not do anything\", \"enum\": [\"none\", \"speak\", \"non-verbal communication\", \"action\", \"leave\"], \"type\": \"string\"}, \"argument\": {\"title\": \"Argument\", \"description\": \"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\", \"type\": \"string\"}}, \"required\": [\"action_type\", \"argument\"]}\n```\u001b[0m
"""
# static
ACTION_LIST = "none action speak non-verbal communication leave"  # " ".join(ActionType)

ACTION_REVERSE_MAP = {"left ": "leave", "did n": "none", "said:": "speak"}

MODEL_CHECKPOINT =  "/mnt/coai-nas/yibai/LLMs/Llama-2-7b-ms" #"/mnt/coai-nas/yibai/LLMs/Meta-Llama-3.1-8B"


TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    padding=False,
    truncation=False,
)


def to_natural_language(self) -> str:
    match self.action_type:
        case "none":
            return "did nothing"
        case "speak":
            return f'said: "{self.argument}"'
        case "non-verbal communication":
            return f"[{self.action_type}] {self.argument}"
        case "action":
            return f"[{self.action_type}] {self.argument}"
        case "leave":
            return "left the conversation"


SELECTED_TAG = ["gpt-4_gpt-4_v0.0.1_clean"]


def detect_action(msg):
    # first detect what action type is, default at none
    if msg.startswith("said:"):
        action = "speak"
    elif msg.startswith("left"):
        action = "leave"
    elif msg.startswith("[non-verbal communication]"):
        action = "non-verbal communication"
    elif msg.startswith("[action]"):
        action = "action"
    else:
        action = "none"

    return action


def generate_result(msg):
    action = detect_action(msg)
    result = {}
    result["action_type"] = action
    result["argument"] = ""
    # know formating argument based on action type
    match action:
        case "speak":
            # NOTE: this assume that the speech is in quotes, not ending without punctuation
            result["argument"] = msg.replace("said: ", "")[1:-1]
        case "action":
            result["argument"] = msg
        case "non-verbal communication":
            result["argument"] = msg

    str_result = json.dumps(result)

    return str_result


def surpass_max_token_check(string, max_token=MAX_TOKEN, tokenizer=TOKENIZER):
    prompt_tokens = len(tokenizer(string)["input_ids"])
    return max(prompt_tokens - max_token, 0)


def truncate_prompt_to_length(dia_his, surpass_num, tokenizer=TOKENIZER):
    # context_len = len(tokenizer(context)['input_ids'])
    dia_sen = dia_his.split("\n")
    remove_len = 0
    i = 0
    while remove_len < surpass_num:
        remove_len += len(tokenizer(dia_sen[i])["input_ids"])
        i += 1
    trunc_dia = "\n".join(p for p in dia_sen[i:])
    return trunc_dia


def reverse_episode_log(
    epilog, later_speak=False, include_format=True, max_token=MAX_TOKEN
):
    episode_msg = epilog.messages
    # per episode
    agent_model = epilog.models[1] if not later_speak else epilog.models[2]
    promt_template = PROMPT_TEMPLATE

    if len(episode_msg) > 0:
        init_loop = episode_msg[0]
        # figure out who speak later, as we must use the 2nd player's data, else turn 0 have nothing to predict the beginning
        if later_speak:
            speaker = init_loop[-1][0]  # this would be the agent as well
            turn_div = 1
        # figure out who speak the first
        else:
            speaker = init_loop[-2][0]
            turn_div = 0

    prompt_result_instances = []
    dial_history = ""

    for i in range(0, len(episode_msg)):
        msg = episode_msg[i]
        if (len(msg) != 4) and i < (len(episode_msg) - 1):
            continue
        turn_dic = {"model": agent_model}
        for tpl in msg:
            if tpl[0] == "Environment" and (tpl[1] == speaker):
                if i > 0:
                    dial_history += "\n" + tpl[2]
                else:
                    # for the first context, we don't need \n
                    context = tpl[2]
                    dial_history += context

            if tpl[0] == speaker:  # if speaker is the agent, use what he said as result
                str_result = generate_result(tpl[2])
                # check if this is the end
        if i % 2 == turn_div:
            # take alternative turns as we always want to predict one agent, not both
            next_turn = i
            prompt = promt_template.format(
                agent=speaker, history=dial_history, turn_number=next_turn
            )
            over_tokens = surpass_max_token_check(prompt, max_token)
            if over_tokens > 0:
                all_dial = dial_history[len(context) :]
                # print(all_dial)
                trun_dial = truncate_prompt_to_length(all_dial, over_tokens)
                prompt = promt_template.format(
                    agent=speaker,
                    history=context + "\n" + trun_dial,
                    turn_number=next_turn,
                )
            if include_format:
                prompt += FORMAT_TEMPLATE
            turn_dic["prompt"] = prompt
            turn_dic["result"] = str_result
            prompt_result_instances.append(turn_dic)

    return prompt_result_instances


def concat_episode_msg(epilog):
    episode_msg = epilog.messages
    # per episode

    if len(episode_msg) > 0:
        init_loop = episode_msg[0]
        speaker = init_loop[-2][0]
    dial_history = ""

    for i in range(0, len(episode_msg)):
        msg = episode_msg[i]
        if (len(msg) != 4) and i < (len(episode_msg) - 1):
            continue
        for tpl in msg:
            if tpl[0] == "Environment" and (tpl[1] == speaker):
                if i > 0:
                    dial_history += "\n" + tpl[2]
                else:
                    # for the first context, we don't need \n
                    context = tpl[2]
                    dial_history += context

    return dial_history


def parse_prompt_to_json(episode, dir, init_speak, include_format=False):
    prompt_result_instances = reverse_episode_log(episode, init_speak, include_format)

    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(len(prompt_result_instances)):
        instance = prompt_result_instances[i]
        todump = json.dumps(instance, indent=4)
        with open(dir + "/{}-{}-{}.json".format(episode.pk, init_speak, i), "w") as f:
            f.write(todump)


def run_reverse_by_pk_agent(episode_pk, agent_side, save_dir):
    """
    Entry function if you want to reverse engineer given a pk, not a episode
    """
    episode = EpisodeLog.find(EpisodeLog.pk == episode_pk).all()[0]
    parse_prompt_to_json(episode, save_dir, agent_side, False)


def run_all_tag_reverse(filter_env_dic, dir):
    # tag_episodes = get_clean_episodes(selected_tags=[tag])[tag]
    for k, v in filter_env_dic.items():
        cutoff = len(v) // 2
        for i in range(len(v)):
            episode = v[i]
            if i < cutoff:
                parse_prompt_to_json(episode, dir, False)
            else:
                parse_prompt_to_json(episode, dir, True)