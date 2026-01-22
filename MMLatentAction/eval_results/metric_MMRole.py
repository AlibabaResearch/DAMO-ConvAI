import json
import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from api_utils import robust_API_response

eval_system_prompt = """
You are an objective and precise evaluator, specializing in rigorously assessing the role-playing and multimodal understanding abilities of various models.
"""

eval_prompt_template = """
## [Question Start]
{question}
## [Question End]

## [Model A's Response Start]
{evaluated_answer}
## [Model A's Response End]

## [Model B's Response Start]
{groundtruth_answer}
## [Model B's Response End]

## [Instruction]
The task instruction of the two models is to directly role-play as {role_name} and talk with a curious human about the given image using the distinctive tone, manner and vocabulary of {role_name}. 

Here is the detailed character information about {role_name}:
{role_info}

Please evaluate the following aspects of each model's response:
1. Instruction Adherence: Do the responses accurately adhere to the task instruction, directly role-playing as {role_name} and only including words that {role_name} should say, without any additional explanatory prefixes or suffixes?
2. Fluency: Are the responses grammatically correct and smoothly articulated?
3. Coherency: Do the responses maintain a coherent thread of dialogue without contradicting earlier parts of the conversation or previously established facts?
4. Image-Text Relevance: Are the responses closely related to the visual content of the image?
5. Response Accuracy: Do the responses accurately answer the curious human's words or appropriately initiate a conversation based on the image?
6. Personality Consistency: Do the responses accurately and sufficiently reflect the personality of {role_name}?
7. Knowledge Consistency: Are the responses consistent with the factual knowledge that {role_name} should possess, including experiences, abilities, and relationships?
8. Tone Consistency: Do the responses maintain a consistent tone that aligns with {role_name}'s typical manner of speaking and catchphrases, rather than resembling the style of AI assistants?

For each aspect, provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from 1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.

The output should be in the following format:
1. Instruction Adherence:  {{Qualitative Evaluation}}, [Scores]: ({{the score of Model A}}, {{the score of Model B}})
2. Fluency: {{Qualitative Evaluation}}, [Scores]: ({{the score of Model A}}, {{the score of Model B}})
etc.

Please ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment.
Format requirement: Please ensure that your evaluations only include 8 score pairs, which means that there can only be eight pairs of [Scores]: () in your output text.
"""

import re


def extract_score_pairs(evaluation_text):
    pattern = r'\[Scores\]:\s*\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, evaluation_text)
    scores = [(int(a), int(b)) for a, b in matches]
    return scores


num_dimensions = 8

def get_eval_score(eval_disc, evaluated_answer, model_engine, num_eval_iters=5):
    # 构建评估 prompt
    eval_prompt = eval_prompt_template.format(
        question=eval_disc["question"],
        evaluated_answer=evaluated_answer,
        groundtruth_answer=eval_disc["groundtruth_answer"],
        role_name=eval_disc["role_name"],
        role_info=eval_disc["role_info"]
    )

    all_evaluation_criteria = [[[],[]] for _ in range(num_dimensions)]

    for iter in range(num_eval_iters):

        try:
            res = robust_API_response(
                model_engine=model_engine,
                system_prompt=eval_system_prompt,
                user_prompt=eval_prompt,
                flag_web_search=False,
                require_json=False,
                temperature=0,
            )
            evaluation_text = res.strip()
        except Exception as e:
            evaluation_text = f"[ERROR during evaluation]: {str(e)}"

        iter_evaluation_criteria = None
        try:
            iter_evaluation_criteria = extract_score_pairs(evaluation_text)
        except Exception as e:
            print(e)

        try:
            assert len(iter_evaluation_criteria)==num_dimensions
            for dim_idx in range(num_dimensions):
                assert len(iter_evaluation_criteria[dim_idx])==2
        except Exception as e:
            print(e)
            continue

        for dim_idx in range(num_dimensions):
            all_evaluation_criteria[dim_idx][0].append(iter_evaluation_criteria[dim_idx][0])
            all_evaluation_criteria[dim_idx][1].append(iter_evaluation_criteria[dim_idx][1])
    print(all_evaluation_criteria)
    valid_iters_num = len(all_evaluation_criteria[dim_idx][0])
    if valid_iters_num < 1:
        return None,None

    evaluation_criteria = [[0,0] for _ in range(num_dimensions)]
    for dim_idx in range(num_dimensions):
        evaluation_criteria[dim_idx][0]=sum(all_evaluation_criteria[dim_idx][0])/valid_iters_num
        evaluation_criteria[dim_idx][1]=sum(all_evaluation_criteria[dim_idx][1])/valid_iters_num
    print(evaluation_criteria)

    return evaluation_criteria, evaluation_text


