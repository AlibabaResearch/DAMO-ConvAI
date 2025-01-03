import os
import re

import json
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from prompt_reverse_engineering import reverse_episode_log
from sotopia.database.logs import EpisodeLog

def convert_sharegpt(messages):
    dialogue = {"conversations": []}
    for m in messages:
        role = "human" if m["role"] == "user" else "gpt"
        dialogue["conversations"].append({"from": role, "value": m["content"]})
    return dialogue
    
def get_episodes_by_env_dict(
    tags: list,
    scenarios: set,
) -> dict[str, list[EpisodeLog]]:
    # Load all episodes from tags
    eps_by_tag = {}
    for tag in tags:
        eps = EpisodeLog.find(EpisodeLog.tag == tag).all()
        eps = sorted(eps)
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

        assert ep[0].models[ep[1]] == "custom/sft@http://localhost:8000/v1/", ep[0].models

        if "Gwen Pierce" in conv[idx]["prompt"][25:] and "Agent1 is interested in buying a used car from Agent2, who is a car salesperson at a local dealership" in conv[idx]["prompt"][25:]:
            continue
        messages = reshape_prompt(conv[idx]["prompt"][25:])
        messages.append({"role": "assistant", "content": conv[idx]["result"].replace("[action]", "").strip()})
        new_data = convert_sharegpt(messages)
        new_data["env"] = ep[0].environment
        new_data["agents"] = ep[0].agents
        new_data["idx"] = ep[1] - 1
        conv_list.append(new_data)

    with open(output_file, "w") as f:
        f.write(json.dumps(conv_list, indent=4))


def filter_episodes(eps_by_env: dict) -> dict:
    eps_by_env_filtered = {}
    for env, eps in eps_by_env.items():
        # first sort on goal score, second sort on overall score
        eps_by_env_filtered[env] = []
        ep_reward_list_1 = []
        ep_reward_list_2 = []
        for ep in eps:
            if ep.models[1] == "custom/sft@http://localhost:8000/v1/" and ep.rewards[0] != 0.0 and (ep.rewards[0][1]["goal"] <= 7 or ep.rewards[0][1]["relationship"] <= 2) and len(ep_reward_list_1) < 3:
                ep_reward_list_1.append(
                    (
                        ep.rewards[0][1]["goal"],
                        ep.rewards[0][1]["overall_score"],
                        ep,
                    )
                )
            if ep.models[2] == "custom/sft@http://localhost:8000/v1/" and ep.rewards[1] != 0.0 and (ep.rewards[1][1]["goal"] <= 7 or ep.rewards[1][1]["relationship"] <= 2) and len(ep_reward_list_2) < 3:
                ep_reward_list_2.append(
                    (
                        ep.rewards[1][1]["goal"],
                        ep.rewards[1][1]["overall_score"],
                        ep,
                    )
                )
        if len(ep_reward_list_1) == 0 and len(ep_reward_list_2) == 0:
            eps_by_env_filtered.pop(env)
        for idx, ep_reward_list in enumerate([ep_reward_list_1, ep_reward_list_2]):
            for ep_reward in ep_reward_list:    
                eps_by_env_filtered[env].append((ep_reward[2], idx + 1))

    return eps_by_env_filtered


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
    

def filter_negative():
    # get conversation episodes according to social task scenarios
    random.seed(11)  # This is important for reproducement
    eps_list = []
    # TODO: Fill in the tag. The tag should be the --TAG you used in `../data_generate/scripts/sotopia_conf/generation_utils_conf/generate_xxx.gin`
    TAGS = ["benchmark_custom/sft@http://localhost:8000/v1/_gpt-4o_gpt-4o_neg_trial9"] #["benchmark_custom/sft@http://localhost:8000/v1/_custom/sft@http://localhost:8000/v1/_gpt-4o_neg_trial0"]
    with open("./data/used_env.json", "r") as f:
        # TODO: Fill in the experiment name. The name should be the key value of social tasks in `../data_generate/env_files/used_env.json`
        ENVS = json.loads(f.read())
        SCENARIOS = ENVS["dev-round-1"] + ENVS["selftrain-round-2"] + ENVS["selftrain-round-3"] + ENVS["selftrain-round-4"]
    # TODO: Fill in the target model and gpt model (optional)
    TARGET_MODEL = "custom/sft@http://localhost:8000/v1/"
    # GPT_MODEL = "custom/sft@http://localhost:8000/v1/"

    eps_by_env = get_episodes_by_env_dict(tags=TAGS, scenarios=set(SCENARIOS))
    
    goal_reward_by_env_target = get_reward_for_agent_model(
        eps_by_env=eps_by_env, agent_model=TARGET_MODEL, reward_metric="goal"
    )
    relation_reward_by_env_target = get_reward_for_agent_model(
        eps_by_env=eps_by_env, agent_model=TARGET_MODEL, reward_metric="relationship"
    )
    print("Before filtering:")
    show_statistics(goal_reward_by_env_target, metric="goal")
    show_statistics(relation_reward_by_env_target, metric="relationship")


    # filter the conversation episodes
    eps_by_env_filtered = filter_episodes(eps_by_env)
    print("After filtering:")
    show_statistics(get_reward_for_agent_model(eps_by_env=eps_by_env_filtered, agent_model=TARGET_MODEL, reward_metric="goal"), metric="goal")
    show_statistics(get_reward_for_agent_model(eps_by_env=eps_by_env_filtered, agent_model=TARGET_MODEL, reward_metric="relationship"), metric="relationship")

    for env in eps_by_env_filtered:
        eps_list += eps_by_env_filtered[env]

    print(len(eps_list))
    random.shuffle(eps_list)

    # generate training data based on filtered conversations
    get_train_data_from_eps_list(
        eps_list=eps_list, output_file="./dpo/negative_data.json"
    )

    get_negative_combos(eps_by_env_filtered)

    return eps_list

def get_negative_combos(eps_by_env_filtered):
    
    combos = []
    for env, eps in eps_by_env_filtered.items():
        for ep in eps:
            combos.append({"env_id": ep[0].environment, "agent_ids": ep[0].agents, "idx": ep[1]-1})
    with open("./dpo/negative_combos.json", "w") as f:
        f.write(json.dumps(combos, indent=4))

if __name__ == "__main__":
    filter_negative()