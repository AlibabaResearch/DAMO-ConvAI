# -*- coding:utf-8 -*-
import argparse
import json
import os
import time
import multiprocessing
import datetime
import re
import random
import copy
from utils import experts_task

import openai
from tqdm import tqdm

MAX_API_RETRY = 200
REQ_TIME_GAP = 4

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--answer-model-name", nargs="+", default=[])
parser.add_argument("-af", "--all_file")
parser.add_argument('-o', '--output', help='Output file (defaults to stdout)')
parser.add_argument("-m", "--eval-model", default="gpt-3.5-turbo-0301")
parser.add_argument("-lm", "--limited", action="store_true", help="if limit")

args = parser.parse_args()

if args.eval_model == "gpt-4":
    cost_per_promtp_token = 0.03 / 1000
    cost_per_completion_token = 0.06 / 1000
elif args.eval_model == "gpt-3.5-turbo-0301":
    cost_per_promtp_token = 2/ 10**6
    cost_per_completion_token = 2/ 10**6
else:
    raise ValueError("Invalid evaluator name")

os.environ["OPENAI_API_KEY"] = "sk-*******"
openai.api_key = os.environ["OPENAI_API_KEY"]

task_funct = {
    "webgpt": [experts_task.gen_prompt_aspectu_QA, experts_task.gen_prompt_aspect_QA, experts_task.gen_prompt_init_QA, experts_task.gen_prompt_QA],
    "summeval": [experts_task.gen_prompt_aspectu_SUM, experts_task.gen_prompt_aspect_SUM, experts_task.gen_prompt_init_SUM, experts_task.gen_prompt_SUM],
    "openmeva": [experts_task.gen_prompt_aspectu_Story, experts_task.gen_prompt_aspect_Story, experts_task.gen_prompt_init_Story, experts_task.gen_prompt_Story],
    "bagel": [experts_task.gen_prompt_aspectu_DataText, experts_task.gen_prompt_aspect_DataText, experts_task.gen_prompt_init_DataText, experts_task.gen_prompt_DataText],
    "hellaswag": [experts_task.gen_prompt_aspectu_NLI, experts_task.gen_prompt_aspect_NLI, experts_task.gen_prompt_init_NLI, experts_task.gen_prompt_NLI],
    "shp": [experts_task.gen_prompt_aspectu_QA, experts_task.gen_prompt_aspect_QA, experts_task.gen_prompt_init_QA, experts_task.gen_prompt_QA],
    "eli5": [experts_task.gen_prompt_aspectu_QA, experts_task.gen_prompt_aspect_QA, experts_task.gen_prompt_init_QA, experts_task.gen_prompt_QA],
    "rlhf-reward-chinese": [experts_task.gen_prompt_aspectu_QA, experts_task.gen_prompt_aspect_QA, experts_task.gen_prompt_init_QA, experts_task.gen_prompt_QA],
    "reward-aira-portuguese": [experts_task.gen_prompt_aspectu_QA, experts_task.gen_prompt_aspect_QA, experts_task.gen_prompt_init_QA, experts_task.gen_prompt_QA],
    "summarizefeedback": [experts_task.gen_prompt_aspectu_SUM, experts_task.gen_prompt_aspect_SUM, experts_task.gen_prompt_init_SUM, experts_task.gen_prompt_SUM],
    "rlhf-reward-russian": [experts_task.gen_prompt_aspectu_SDia, experts_task.gen_prompt_aspect_SDia, experts_task.gen_prompt_init_SDia, experts_task.gen_prompt_SDia],
    "hh-rlhf": [experts_task.gen_prompt_aspectu_SDia, experts_task.gen_prompt_aspect_SDia, experts_task.gen_prompt_init_SDia, experts_task.gen_prompt_SDia],
    "pku-saferlhf": [experts_task.gen_prompt_aspectu_SaQA, experts_task.gen_prompt_aspect_SaQA, experts_task.gen_prompt_init_SaQA, experts_task.gen_prompt_SaQA],
    "mt-bench": [experts_task.gen_prompt_aspectu_MDia, experts_task.gen_prompt_aspect_MDia, experts_task.gen_prompt_init_MDia, experts_task.gen_prompt_MDia],
    "code-contests": [experts_task.gen_prompt_aspectu_Code, experts_task.gen_prompt_aspect_Code, experts_task.gen_prompt_init_Code, experts_task.gen_prompt_Code],
}

def query_gpt(messages, tp=1, k=1):
    for i in range(MAX_API_RETRY):
        try:
            if tp == 0:
                response = openai.ChatCompletion.create(
                    model=args.eval_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=512,
                    n=k
                )
            else:
                response = openai.ChatCompletion.create(
                    model=args.eval_model,
                    messages=messages,
                    max_tokens=512,
                    n=k
                )
            return response
            # time.sleep(10)
        except openai.error.RateLimitError as e1:
            print(e1)
            time.sleep(10)
        except Exception as e:
            print(e)
            # time.sleep(5)
            # print('error')
    response = {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Evaluation evidence: None Score of Assistant 1: 0 Score of Assistant 2: 0 Confidence: 0",
                "role": "assistant"
                }
            }
        ],
        "usage": {
            "completion_tokens": 457,
            "prompt_tokens": 562,
            "total_tokens": 1019
        }
    }
    return response

    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")

def get_aspects(messages, dataset_name):
    aspects = []
    cnt = 0
    contents = []
    cost = 0
    while cnt < 10 and len(aspects) == 0:
        response = query_gpt(messages, tp=0)
        content = response["choices"][0].message.get("content", "")
        contents.append(content)
        aspects = parse_aspect_from_review(content, dataset_name)
        cnt += 1
        cost += response['usage']['prompt_tokens'] * cost_per_promtp_token
        cost += response['usage']['completion_tokens'] * cost_per_completion_token

    return aspects, content, cost

def get_scores(messages):
    score1, score2 = -1, -1
    cnt = 0
    contents = []
    cost = 0
    while cnt < 10 and (score1 == -1 or score2 == -1):
        response = query_gpt(messages)
        content = response["choices"][0]["message"]["content"]
        contents.append(content)
        score1, score2 = parse_score_from_review(content)
        cnt += 1
        cost += response['usage']['prompt_tokens'] * cost_per_promtp_token
        cost += response['usage']['completion_tokens'] * cost_per_completion_token

    if score1 == -1 or score2 == -1:
        score1, score2 = 0.0, 0.0

    return score1, score2, content, cost

def aspect_layer(ques, ans1, ans2, asp_num, dataset_name):
    if args.limited:
        user_prompt = task_funct[dataset_name][1](ques, ans1, ans2, asp_num)
    else:
        user_prompt = task_funct[dataset_name][0](ques, ans1, ans2, asp_num)
    messages = [
            {"role": "user", "content": user_prompt},
    ]
    aspects, content, cost = get_aspects(messages, dataset_name)

    return aspects, content, cost


def init_layer(ques, ans1, ans2, neuro_num, aspects, dataset_name):
    scores = {}
    contents = {}
    user_prompts = {}
    cost = 0
    neuro_num = min(neuro_num, len(aspects))
    if neuro_num < 1:
        aspects = ["Accuracy", "Relevance"]
        neuro_num = 2
    for i in range(neuro_num):
        neuro_name = "m"+str(i+1)
        system_prompt, user_prompt = task_funct[dataset_name][2](ques, ans1, ans2, aspects[i])
        messages_12 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
        score1_12, score2_12, content_12, c1 = get_scores(messages_12)
        user_prompts[neuro_name] = [user_prompt]

        scores[neuro_name] = [[score1_12, score2_12]]
        contents[neuro_name] = [content_12]
        cost += c1

        system_prompt_, user_prompt_ = task_funct[dataset_name][2](ques, ans2, ans1, aspects[i])
        messages_21 = [
                        {"role": "system", "content": system_prompt_},
                        {"role": "user", "content": user_prompt_},
                    ]
        score2_21, score1_21, content_21, c2 = get_scores(messages_21)
        content_21 = content_21.replace("Assistant 1", "*****").replace("Assistant 2", "Assistant 1").replace("*****", "Assistant 2")
        user_prompts[neuro_name].append(user_prompt_)
        scores[neuro_name].append([score1_21, score2_21])
        contents[neuro_name].append(content_21)
        cost += c2
    
    return scores, contents, cost, user_prompts, aspects

def single_layer(ques, ans1, ans2, M, Ml, aspects, dataset_name):
    # M = {"m1":[], "m2":[], "m3":[]}
    # aspects = [["accuracy", ...], ...]
    neuro_num = len(M)
    
    scores = {}
    contents = {}
    cost = 0
    ret_asps = []
    user_prompts = {}
    win_size = 2
    
    for i in range(neuro_num):
        neuro_name = "m"+str(i+1)
        own = copy.deepcopy(M[neuro_name])
        if len(Ml) > 0:
            own = copy.deepcopy(Ml[neuro_name]) + own
        others = []
        asps = []
        start_ii = max(i-win_size+1, 0)
        end_ii = min(neuro_num, i+win_size)
        for ii in range(start_ii, end_ii):
            asps.append(aspects[ii])
            if ii != i:
                others = others+copy.deepcopy(M["m"+str(ii+1)])

        system_prompt, user_prompt, asp_r = task_funct[dataset_name][3](ques, ans1, ans2, own, others, asps)
        messages_12 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
        score1_12, score2_12, content_12, c1 = get_scores(messages_12)
        user_prompts[neuro_name] = [user_prompt]

        ret_asps.append(asp_r.split(", "))

        scores[neuro_name] = [[score1_12, score2_12]]
        contents[neuro_name] = [content_12]
        cost += c1

        for oi in range(len(own)):
            own[oi] = own[oi].replace("Assistant 1", "*****").replace("Assistant 2", "Assistant 1").replace("*****", "Assistant 2")
        for oj in range(len(others)):
            others[oj] = others[oj].replace("Assistant 1", "*****").replace("Assistant 2", "Assistant 1").replace("*****", "Assistant 2")

        system_prompt_, user_prompt_, _ = task_funct[dataset_name][3](ques, ans2, ans1, own, others, asps)
        messages_21 = [
                        {"role": "system", "content": system_prompt_},
                        {"role": "user", "content": user_prompt_},
                    ]
        score2_21, score1_21, content_21, c2 = get_scores(messages_21)
        content_21 = content_21.replace("Assistant 1", "*****").replace("Assistant 2", "Assistant 1").replace("*****", "Assistant 2")
        user_prompts[neuro_name].append(user_prompt_)
        scores[neuro_name].append([score1_21, score2_21])
        contents[neuro_name].append(content_21)
        cost += c2
    
    return scores, contents, cost, ret_asps, user_prompts

def widedeep_eval(input_):
    ques, ans1, ans2, hm, dataset_name = input_
    num_neuro = 2
    num_layer = 2
    cost = 0
    layer = 0
    hist_contents = {}
    hist_scores = {}
    hist_user_prompts = {}
    aspects, asp_content, cst = aspect_layer(ques, ans1, ans2, num_neuro, dataset_name)
    cost += cst
    if not args.limited:
        num_neuro = len(aspects)
    scores, contents, cst, init_user, aspects = init_layer(ques, ans1, ans2, num_neuro, aspects, dataset_name)
    cost += cst
    hist_scores["l"+str(layer+1)] = scores
    hist_contents["l"+str(layer+1)] = contents
    hist_user_prompts["l"+str(layer+1)] = init_user
    M = copy.deepcopy(contents)
    aspect_cum = [[asp] for asp in aspects]
    Ml = {}
    while layer < num_layer-1:
        layer += 1
        scores, contents, cost, aspect_cum, single_user = single_layer(ques, ans1, ans2, M, Ml, aspect_cum, dataset_name)
        cost += cst
        hist_scores["l"+str(layer+1)] = scores
        hist_contents["l"+str(layer+1)] = contents
        hist_user_prompts["l"+str(layer+1)] = single_user
        M = copy.deepcopy(contents)

    scores_list = []
    for ly in hist_scores:
        for neuro in hist_scores[ly]:
            scores_list.extend(hist_scores[ly][neuro])
    score1 = sum([score[0] for score in scores_list]) / (len(scores_list)+1e-5)
    score2 = sum([score[1] for score in scores_list]) / (len(scores_list)+1e-5)
    with open(f"{args.output}", "a+", encoding="utf-8") as ff:
        results = {
                    "query": ques,
                    "response1": ans1,
                    "response2": ans2,
                    "aspect": asp_content,
                    "review": hist_contents,
                    "user_prompt": hist_user_prompts,
                    "scores": hist_scores,
                    "scores_list": scores_list,
                    "score": [score1, score2],
                    "human": hm
        }
        ff.write(json.dumps(results, ensure_ascii=False) + "\n")

    return ques, ans1, ans2, hist_contents, cost, hist_scores, [score1, score2], hm

def parse_aspect_from_review(review, dataset_name):
    asp = []
    try:
        pattern = r"\$(.*?)&"
        if dataset_name == "code-contests":
            pattern = r"\$(.*?)@"
        # res = review.split("=======")[-1]
        res = review
        matches = re.findall(pattern, res, re.DOTALL)

        for match in matches:
            aspect = match.strip().replace("\n","")
            asp.append(aspect)
        return asp
    except:
        print(f'Failed to parse aspects from {review}')
        return []
    
def parse_score_from_review(review):
    try:
        score1 = re.search(r"Score of (?:the )?Assistant 1:\s*([\d\.]+)", review).group(1)
        score2 = re.search(r"Score of (?:the )?Assistant 2:\s*([\d\.]+)", review).group(1)
        return [float(score1), float(score2)]
    except:
        print(f'Failed to parse scores from {review}')
        return [-1, -1]

def get_json_all(file_path):
    file_path = os.path.expanduser(file_path)
    json_all = []
    with open(file_path, "r", encoding="utf-8") as f:
        # json_all = json.load(f)
        for line in f:
            json_all.append(json.loads(line.strip()))
    
    return json_all


if __name__ == "__main__":
    all_json = get_json_all(args.all_file)
    model2id = {args.answer_model_name[0]:"model1", args.answer_model_name[1]:"model2"}
    id2model = {"model1":args.answer_model_name[0], "model2":args.answer_model_name[1]}

    reviews = []
    histories = []
    
    # for task in all_json:
    for ds in all_json:
        ques = ds["query"]
        ans1 = ds["response1"]
        ans2 = ds["response2"]
        hm = ds["human"]
        dn = ds["dataset"]
        histories.append([ques, ans1, ans2, hm, dn])

    with multiprocessing.Pool(processes=50) as pool:
        for rs in tqdm(pool.imap(widedeep_eval, histories), total=len(histories)):
            reviews.append(rs)
