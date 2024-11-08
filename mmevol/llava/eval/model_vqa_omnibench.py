import argparse
import json
import math
import os
import sys
sys.path.append("/mnt/workspace/lr/workspace/Open-LLaVA-NeXT/")

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

import random
import numpy as np
import re

def split_string_by_options(input_string):
    pattern = r'(A\..*?)(B\..*?)(C\..*?)(D\..*)'
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches[0]]


def parse_multi_choice_response(response, all_choices, index2ans, default_answer=None):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        if default_answer is None:
            pred_index = random.choice(all_choices)
        else:
            pred_index = default_answer
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name)

    if args.square_eval:
        model.config.image_grid_pinpoints = [
            [
                672,
                672
            ]
        ]
    if args.question_file.endswith("jsonl"):
        questions = [json.loads(q) for q in open(
            os.path.expanduser(args.question_file), "r")]
    else:
        questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["index"]
        image_file = line["image_path"]
        # qs = line["question"]
        qs =f"Speech:{line['audio content']} \nPlease answer the questions based on the user's input image and speech. Carefully read the following question and select the letter corresponding to the correct answer. Highlight the applicable choices without giving explanations. \n{line['question']}.\n."
        for i,j in enumerate(line['options']):
            qs+=f"{chr(ord('A') + i)}. {j}\n"
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(
            args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images(
            [image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True)

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        options=line['options']
        all_choices = ['A', 'B', 'C', 'D']
        index2ans = {chr(ord("A") + i): "" for i in range(4)}
        all_options = options
        for j in range(len(all_choices)):
            index2ans[all_choices[j]] = all_options[j].replace(f"{all_choices[j]}.", "").strip()
        parsed_response = parse_multi_choice_response(outputs, all_choices, index2ans)
        correct_answer = parse_multi_choice_response(line['answer'], all_choices, index2ans)

        ans_file.write(json.dumps({"index": line['index'],
                        "audio_type": line['audio type'], 
                        "question":qs,
                        "options": options,
                        "response": outputs,
                        "parsed_response": parsed_response,
                        "correct_answer": correct_answer,
                        "is_correct": parsed_response == correct_answer}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/Open-LLaVA-NeXT/checkpoints/llava-v1.6-7b_qwen-7b_clip-large-336_debug_ablation_sft_evol/checkpoint-14000")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/workspace/lr/workspace/OmniBench/mm_data/image")
    parser.add_argument("--question-file", type=str,
                        default="/mnt/workspace/lr/workspace/OmniBench/dataset/batch-5_1142_20240817.jsonl")
    parser.add_argument("--answers-file", type=str, default="/mnt/workspace/lr/workspace/OmniBench/dataset/prediction.jsonl")
   # parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--square_eval", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)
