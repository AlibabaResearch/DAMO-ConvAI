import json
import re
from rouge import Rouge
from tqdm import tqdm
import spacy
import nltk
from nltk.corpus import wordnet as wn
import os

nltk.download('wordnet')

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    for s1 in synsets1:
        for s2 in synsets2:
            if s1 == s2:
                return True
    return False

def is_number(s):
    try:
        s = s.replace(',', '')
        float(s)
        return True
    except ValueError:
        return False


def str_to_num(s):
    s = s.replace(',', '')
    if is_number(s):
        return float(s)


def extract_number(s):
    pattern = r'[\d]+[\,\d]*[\.]{0,1}[\d]+'

    if re.search(pattern, s) is not None:
        result = []
        for catch in re.finditer(pattern, s):
            result.append(catch[0])
        return result
    else:
        return []


def relaxed_accuracy(pr, gt):
    return abs(float(pr) - float(gt)) <= 0.05 * abs(float(gt))


nlp = spacy.load('en_core_web_sm')

def remove_units(text):
    doc = nlp(text)
    new_text = []
    i = 0

    while i < len(doc):
        token = doc[i]
        if token.pos_ == 'NUM':
            j = i + 1

            possible_unit_parts = []
            while j < len(doc) and (doc[j].pos_ == 'NOUN' or doc[j].pos_ == 'ADP' or doc[j].tag_ in ['NN', 'IN']):
                possible_unit_parts.append(doc[j].text)
                j += 1
            if possible_unit_parts:
                new_text.append(token.text)  
                i = j 
                continue
        new_text.append(token.text)
        i += 1

    return ' '.join(new_text)

# For evalution except map
def evaluator(path,source):
    eval_file = []
    with open(path) as f:
        for line in f:
            eval_file.append(json.loads(line))
    eval_file=[i for i in eval_file if i['source']==source]
    ok_results = []
    bad_results = []
    structural_cnt = 0
    data_extraction_cnt = 0
    math_reasoning_cnt = 0
    color_cnt = 0
    caption_cnt = 0
    summary_cnt = 0

    rouge = Rouge()
    summary_score = 0.0

    for result in tqdm(eval_file):
        pr = result['text']
        gt = result['truth']

        pr = pr.strip().lower()
        gt = gt.strip().lower()

        pattern = r'the answer is (.*?)(?:\.\s|$)'
        match = re.search(pattern, pr)
        if match:
            pr = match.group(1)

        match = re.search(pattern, gt)
        if match:
            gt = match.group(1)

        if len(pr) > 0:
            if pr[-1] == '.':
                pr = pr[:-1]
                if len(pr) >= 1 and pr[-1] == '.':
                    pr = pr[:-1]
            if len(pr) >= 1 and pr[-1] == '%':
                pr = pr[:-1]
            if pr.endswith("\u00b0c"):
                pr = pr[:-2]

        if len(gt) > 0:
            if gt[-1] == '.':
                gt = gt[:-1]
            if gt[-1] == '%':
                gt = gt[:-1]
            if gt.endswith("\u00b0c"):
                gt = gt[:-2]

        pr = remove_units(pr)
        gt = remove_units(gt)

        numeric_values = extract_number(pr)

        if result['type'] == 'STRUCTURAL':
            structural_cnt += 1
        elif result['type'] == 'DATA_EXTRACTION':
            data_extraction_cnt += 1
        elif result['type'] == 'MATH_REASONING':
            math_reasoning_cnt += 1
        elif result['type'] == 'COLOR':
            color_cnt += 1
        elif result['type'] == 'CAPTION':
            caption_cnt += 1
        elif result['type'] == 'SUMMARY':
            summary_cnt += 1

        if result['type'] == 'SUMMARY':
            if pr != '':
                summary_score += rouge.get_scores(gt, pr, avg=True)['rouge-l']['f']
            continue

        if is_number(pr) and is_number(gt) and relaxed_accuracy(str_to_num(pr), str_to_num(gt)):
            ok_results.append(result)
        elif is_number(gt):
            flag = False
            for v in numeric_values:
                if relaxed_accuracy(str_to_num(v), str_to_num(gt)):
                    ok_results.append(result)
                    flag = True
                    break
            if not flag:
                bad_results.append(result)
        elif pr in ['a', 'b', 'c', 'd'] or gt in ['a', 'b', 'c', 'd']:
            if pr == gt:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif len(gt) >= 2 and gt[0] == '[' and gt[-1] == ']':
            if pr == gt:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif len(gt) >= 2 and gt[0] == '(' and gt[-1] == ')':
            first = gt[1]
            second = gt[-2]
            pr_values = extract_number(pr)
            if len(pr_values) == 2 and pr_values[0] == first and pr_values[1] == second:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif pr != "" and pr in gt or gt in pr:
            ok_results.append(result)
        elif pr != "" and are_synonyms(pr, gt):
            ok_results.append(result)
        else:
            bad_results.append(result)

    print(f'Overall Accuracy: {len(ok_results) / (len(eval_file) - summary_cnt) * 100:.2f}%')

    if summary_cnt > 0:
        print(f'Overall Summary Rouge-L Score: {summary_score / summary_cnt:.2f}')

    assert len(ok_results) + len(bad_results) == len(eval_file) - summary_cnt
    return ok_results, bad_results,len(ok_results) / (len(eval_file) - summary_cnt) * 100


def extract_marker(s):
    pattern = r'(?:[A-Za-z][0-9]|[0-9][A-Za-z])'
    marker_list = []
    for match in re.finditer(pattern, s):
        marker_list.append(match[0])

    return marker_list

# For map evaluation
def evaluator_map(path,source):
    eval_file = []
    with open(path) as f:
        for line in f:
            eval_file.append(json.loads(line))
    
    eval_file=[i for i in eval_file if i['source']==source]

    data_file = []

    with open('checkpoints/zwq2018/Multi-modal-Self-instruct/test/maze/test_3k.jsonl', 'r') as f:
        for line in f:
            data_file.append(json.loads(line))

    ok_res = []
    bad_res = []

    score = 0.0
    for result in eval_file:
        index = 0
        pr = result['text']
        gt = result['truth']

        pr_list = extract_marker(pr)
        while data_file[index]['question_id'] != result['question_id']:
            index += 1
        gt_list = data_file[index]['markers']
        # gt_list = result['markers']
        # gt_list = extract_marker(gt)

        if len(gt_list) == 0:
            continue

        pr_list = list(dict.fromkeys(pr_list))

        cnt = 0
        match_index = []
        for i in range(len(pr_list)):
            if pr_list[i] in gt_list:
                cnt += 1

        if cnt / len(gt_list) >= 0.9:
            ok_res.append(result)
        elif cnt / len(gt_list) <= 0.1:
            bad_res.append(result)

        score = score + cnt / len(gt_list)

    print(f'Accuracy: {score / len(eval_file) * 100:.2f}%')
    return ok_res, bad_res, score / len(eval_file) * 100


answer_files="answers/vila_1.5_13b_mminst_prediction.jsonl"

sources=["chart","dashboard","flowchart","iq","layout","maze","org","table"]

# Evaluate 7 scenario: charts, tables, dashboards, flowcharts, relation graphs, floor plans, and visual puzzles
a_sum=0
print("========> charts")
_,_,a=evaluator(answer_files,sources[0])
a_sum+=a



print("========> tables")
_,_,a=evaluator(answer_files,sources[7])
a_sum+=a


print("========> dashboards")
_,_,a=evaluator(answer_files,sources[1])
a_sum+=a


print("========> flowcharts")
_,_,a=evaluator(answer_files,sources[2])
a_sum+=a


print("========> relation graphs")
_,_,a=evaluator(answer_files,sources[6])
a_sum+=a

print("========> floor plans")
_,_,a=evaluator(answer_files,sources[4])
a_sum+=a

print("========> visual puzzles")
_,_,a=evaluator(answer_files,sources[3])
a_sum+=a

print("========> simulated maps")
_,_,a=evaluator_map(answer_files,sources[5])
a_sum+=a

print(a_sum/8)
