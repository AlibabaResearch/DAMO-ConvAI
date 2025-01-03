import os
import re

import json
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from prompt_reverse_engineering import reverse_episode_log
from sotopia.database.logs import EpisodeLog
from data_filtering_negative import filter_negative


def get_episodes_by_env_dict(
    tags: list,
    scenarios: set,
) -> dict[str, list[EpisodeLog]]:
    # Load all episodes from tags
    eps_by_tag = {}
    for tag in tags:
        eps = EpisodeLog.find(EpisodeLog.tag == tag).all()
        if len(eps) > 0:
            eps_by_tag[tag] = eps
    eps_list = sum(eps_by_tag.values(), [])

    # Only select episodes under target scenarios
    eps_by_env = {}
    for ep in eps_list:
        if ep.environment in scenarios:
            if ep.environment in eps_by_env:
                eps_by_env[ep.environment].append(ep)
            else:
                eps_by_env[ep.environment] = [ep]

    return eps_by_env


def get_reward_for_agent_model(
    eps_by_env: dict, agent_model: str, reward_metric: str = "goal"
) -> dict:
    reward_by_env = {}
    if type(list(eps_by_env.values())[0][0]) is EpisodeLog:
    
        for env, eps in eps_by_env.items():
            score_list = []
            for ep in eps:
                rewards = ep.rewards
                if rewards[0] != 0:
                    if ep.models[1] == agent_model:
                        score = rewards[0][1][reward_metric]
                        score_list.append(score)
                    if ep.models[2] == agent_model:
                        score = rewards[1][1][reward_metric]
                        score_list.append(score)
                else:
                    score = 0
                    score_list.append(score)
            reward_by_env[env] = score_list
    else:
        for env, eps in eps_by_env.items():
            score_list = []
            for ep in eps:
                # print(ep[0])
                # print(ep[1])
                rewards = ep[0].rewards
                score = rewards[ep[1]-1][1][reward_metric]
                score_list.append(score)
            reward_by_env[env] = score_list

    return reward_by_env

def get_action_type_argument(text, another_speaker):
    action_type = "speak" if " said: \"" in text else "action"
    if action_type == "speak":
        argument = re.search(r'said: "(.+?)"\n', text, re.DOTALL).group(1)
    else:
        # print(another_speaker)
        # print(text)
        start_idx = text.index(another_speaker) + len(another_speaker)
        end_indx = text.index('\n')
        argument = text[start_idx:end_indx].strip()
        argument = argument.replace("[action]", "").strip()
        argument = argument.replace("[action]", "").strip()
        argument = argument.replace("[action]", "").strip()
    return action_type, argument

def reshape_prompt(prompt):
    speaker = re.search(r'Imagine you are (.+?),', prompt).group(1)
    is_first = re.search(r'\nParticipants: (.+?)\n', prompt).group(1).split(" and ")[0] == speaker
    if is_first == True:
        another_speaker = re.search(r'\nParticipants: (.+?)\n', prompt).group(1).split(" and ")[1]
    else:
        another_speaker = re.search(r'\nParticipants: (.+?)\n', prompt).group(1).split(" and ")[0]

    prompts = prompt.split("Turn #")
    messages = []

    background = prompt.split("Conversation Starts:")[0] + \
'''
Your available action types are speak, leave, and action.

Please only generate a JSON string including the action type and the argument. Your action should follow the given format: {"action_type": "", "argument": ""}

Here is the output schema:
```
{"properties": {"action_type": {"description": "speak, leave, or choose to take actions related to the communication", "enum": ["speak", "leave", "action"], "title": "Action Type", "type": "string"}, "argument": {"description": "the utterance if choose to speak, an empty string if choose to leave, or the actions to be taken if choose to act.", "title": "Argument", "type": "string"}}, "required": ["action_type", "argument"]}
```

** Note: You should "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave. This helps ensure the conversation ends smoothly. When you choose to leave, just output {"action_type": "leave", "argument": ""}**

Conversation Starts:'''
    try:
        if is_first == True:
            messages.append({"role": "user", "content": background + " (You speak first)"})
            for i in range(1, len(prompts) - 1):
                action_type, argument = get_action_type_argument(prompts[i], speaker if i % 2 != 0 else another_speaker)
                speech = '''{"action_type": "''' + action_type + '''", "argument": "''' + argument + '''"}'''
                if i % 2 != 0:
                    messages.append({"role": "assistant", "content": speech})
                else:
                    messages.append({"role": "user", "content": speech})
        else:
            first_ac, first_ag = get_action_type_argument(prompts[1], another_speaker)
            first_speech = '''{"action_type": "''' + first_ac + '''", "argument": "''' + first_ag + '''"}'''
            background = background + "\n\nThe other person's action: " + first_speech
            messages.append({"role": "user", "content": background})
            for i in range(2, len(prompts) - 1):
                action_type, argument = get_action_type_argument(prompts[i], another_speaker if i % 2 != 0 else speaker)
                speech = '''{"action_type": "''' + action_type + '''", "argument": "''' + argument + '''"}'''
                if i % 2 != 0:
                    messages.append({"role": "user", "content": speech})
                else:
                    messages.append({"role": "assistant", "content": speech})
    except Exception:
        print(prompt)
        exit()
    return messages    

def get_train_data_from_eps_list(eps_list: list, output_file: str):
    conv_list = []
    for ep in eps_list:
        if ep[1] == 1:  # target agent at position 1
            conv = reverse_episode_log(
                ep[0], include_format=True, later_speak=False
            )
        else:  # target agent at position 2
            conv = reverse_episode_log(
                ep[0], include_format=True, later_speak=True
            )
        idx = -1 if "leave" in conv[-1]["result"] else -2
        # for turn in conv:
        #     print("*"*10)
        #     print(turn["prompt"][25:])
        #     print()
        #     print(turn["result"])
        #     print("*"*10)
        # exit()
        # for turn in conv:
        #     conv_list.append(
        #         {
        #             "instruction": "",
        #             "input": turn["prompt"][25:],
        #             "output": turn["result"],
        #         }
        #     )
        # print(conv[-1]["result"])
        if "Gwen Pierce" in conv[idx]["prompt"][25:] and "Agent1 is interested in buying a used car from Agent2, who is a car salesperson at a local dealership" in conv[idx]["prompt"][25:]:
            continue
        messages = reshape_prompt(conv[idx]["prompt"][25:])
        messages.append({"role": "assistant", "content": conv[idx]["result"].replace("[action]", "").strip()})
        conv_list.append(convert_sharegpt(messages))
        
    random.shuffle(conv_list)

    with open(output_file, "w") as f:
        f.write(json.dumps(conv_list, indent=4))



def show_statistics(reward_by_env_target, metric):
    # scores = []
    # relations = []
    # for scores_list, relations_list in reward_by_env_target.values():
    #     scores.extend(scores_list)
    #     relations.extend(relations_list)

    scores = sum(reward_by_env_target.values(), [])
    print("Number of rewards: ", len(scores))
    print(f"{metric} Mean: ", np.mean(scores), "Median: ", np.median(scores), "Var: ", np.var(scores))

    score2num = {}
    for s in scores:
        if s in score2num.keys():
            score2num[s] += 1
        else:
            score2num[s] = 1
    for s in score2num.keys():
        score2num[s] /= len(scores)
    print(f"Distribuion of {metric}: ", score2num)


    print("Number of envs: ", len(reward_by_env_target.keys()))
    env_score2num = {}
    for env, rewards in reward_by_env_target.items():
        avg_score = np.mean(rewards)
        if int(avg_score) in env_score2num.keys():
            env_score2num[int(avg_score)] += 1
        else:
            env_score2num[int(avg_score)] = 1
    print(f"Distribution of env {metric} ", env_score2num)
    

def convert_sharegpt(messages):
    dialogue = {"conversations": []}
    for m in messages:
        role = "human" if m["role"] == "user" else "gpt"
        dialogue["conversations"].append({"from": role, "value": m["content"]})
    return dialogue

def filter_positive_self():
    pair_data = []
    eps_idx_negative = filter_negative()
    positive_tag = "benchmark_custom/sft@http://localhost:8000/v1/_custom/copy@http://localhost:8001/v1/_gpt-4o_hpos_trial"

    all_eps_positive = [EpisodeLog.find(EpisodeLog.tag == positive_tag + str(i)).all() for i in range(5)]
    
    
    print(len(eps_idx_negative))
    # print(len(eps_positive))
    
    rule2num = {0:0, 1:0, 2:0}
    goal_n, goal_p, relation_n, relation_p = [], [], [], []
    for ep_n, idx in eps_idx_negative:
        goal_n.append(ep_n.rewards[idx-1][1]["goal"])
        relation_n.append(ep_n.rewards[idx-1][1]["relationship"])
        best_ep = None
        for i in range(5):
            eps_positive = all_eps_positive[i]
            for ep_p in eps_positive:
                if ep_n.environment == ep_p.environment and ep_n.agents == ep_p.agents and ep_p.models[idx] == "custom/sft@http://localhost:8000/v1/":
                    if best_ep == None:
                        best_ep = ep_p
                    elif best_ep.rewards[idx-1][1]["goal"] < ep_p.rewards[idx-1][1]["goal"]:
                        best_ep = ep_p
                    elif best_ep.rewards[idx-1][1]["goal"] == ep_p.rewards[idx-1][1]["goal"] and best_ep.rewards[idx-1][1]["relationship"] < ep_p.rewards[idx-1][1]["relationship"]:
                        best_ep = ep_p
        # print(best_ep)
        goal_p.append(best_ep.rewards[idx-1][1]["goal"])
        relation_p.append(best_ep.rewards[idx-1][1]["relationship"])
        adoptable = False
        if goal_n[-1] < goal_p[-1]:
            adoptable = True
            rule2num[0] += 1
        elif goal_n[-1] == goal_p[-1] and relation_n[-1] < relation_p[-1]:
            adoptable = True
            rule2num[1] += 1
        # elif goal_n[-1] > goal_p[-1] and relation_n[-1] < relation_p[-1] and (ep_n.rewards[0][1]["goal"] + ep_n.rewards[1][1]["goal"] < best_ep.rewards[0][1]["goal"] + best_ep.rewards[1][1]["goal"] - 2):
        #     adoptable = True
        #     rule2num[2] += 1

        if adoptable == True:
            pair_data.append([ep_n, best_ep, idx, goal_p[-1] - goal_n[-1], relation_p[-1] - relation_n[-1]])

    assert len(goal_n) == len(goal_p), "len of goal_n and goal_p is not consistent!"
    print("Number of preference data pairs: ", len(pair_data))
    print("Negative Goal Mean: ", np.mean(goal_n), " Positive Goal Mean: ", np.mean(goal_p))
    print("Negative Relation Mean: ", np.mean(relation_n), " Positive Relation Mean: ", np.mean(relation_p))
    print("Rule Distribution: ", rule2num)

    envs = set()
    for i in range(len(pair_data)):
        envs.add(pair_data[i][0].environment)
    print("Number of envs: ", len(envs))

    eps_list = [pair[1:] for pair in pair_data]
    get_train_data_from_eps_list(
        eps_list=eps_list, output_file="./dpo/positive_data.json"
    )

    get_train_data_from_pair_data(
        pair_data
    )


def get_train_data_from_pair_data(pair_data, output_file="./dpo/preference_data.json"):
    preference_data = []
    for ep_n, ep_p, idx, goal, relation in pair_data:
        if idx == 1:  # target agent at position 1
            conv_n = reverse_episode_log(
                ep_n, include_format=True, later_speak=False
            )
            conv_p = reverse_episode_log(
                ep_p, include_format=True, later_speak=False
            )
        else:  # target agent at position 2
            conv_n = reverse_episode_log(
                ep_n, include_format=True, later_speak=True
            )
            conv_p = reverse_episode_log(
                ep_p, include_format=True, later_speak=True
            )

        for conv in [conv_n, conv_p]:
        
            idx = -1 if "leave" in conv[-1]["result"] else -2

            if "Gwen Pierce" in conv[idx]["prompt"][25:] and "Agent1 is interested in buying a used car from Agent2, who is a car salesperson at a local dealership" in conv[idx]["prompt"][25:]:
                continue
            messages = reshape_prompt(conv[idx]["prompt"][25:])
            messages.append({"role": "assistant", "content": conv[idx]["result"].replace("[action]", "").strip()})
            preference_data.append(convert_sharegpt(messages))
            preference_data[-1]["goal"] = goal
            preference_data[-1]["relationship"] = relation

#    random.shuffle(preference_data)
    with open(output_file, "w") as f:
        f.write(json.dumps(preference_data, indent=4))

if __name__ == "__main__":
    filter_positive_self()