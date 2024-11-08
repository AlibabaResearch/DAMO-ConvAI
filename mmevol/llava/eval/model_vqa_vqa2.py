import argparse
import json
import math
import os
import sys
sys.path.append("/mnt/workspace/lr/workspace/Open-LLaVA-NeXT/")

import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)

        image = Image.open(os.path.join(
            self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images(
            [image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder,
                            tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(mate):
    args=mate
    # print(mate)
    # os.environ['CUDA_VISIBLE_DEVICES']=str(idx%args.n_gpus)
    # args.chunk_idx=idx
    # args.answers_file=f"/mnt/workspace/lr/datasets/playground/playground/data/eval/gqa/answer_{idx}.jsonl"
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

    questions = [json.loads(q) for q in open(
        os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # data_loader = create_data_loader(
    #     questions, args.image_folder, tokenizer, image_processor, model.config)

    for line in tqdm(questions):
        idx=line['question_id']
        image_file = line["image"]
        qs = line["text"]

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
        # print(outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    return args.answers_file

import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/Open-LLaVA-NeXT/checkpoints/llava-v1.6-8b_llama3-8b_clip-large-336_debug_ablation_sft_seed/checkpoint-11213")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/workspace/lr/datasets/coco/test2015")
    parser.add_argument("--question-file", type=str,
                        default="/mnt/workspace/lr/datasets/playground/playground/data/eval/vqav2/share4v_vqav2_mscoco_test-dev2015.jsonl")
    parser.add_argument("--answers-file", type=str, default="/mnt/workspace/lr/workspace/MiniCPM-V/vqa_v2.jsonl")
    # parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--square_eval", type=bool, default=True)
    args = parser.parse_args()
    
    # data=[(args,i) for i in range(args.num_chunks)]

    # with Pool(processes=args.num_chunks) as pool:
    #     output = list(tqdm(pool.imap(eval_model, data), total=len(data)))
    # eval_model(args)
    merge_data=[]
    for idx in range(args.num_chunks):
        # merge_data+=[json.loads(i) for i in open(f"/mnt/workspace/lr/datasets/playground/playground/data/eval/gqa/answer_{idx}.jsonl",'r')]
        merge_data+=[json.loads(i) for i in open(args.answers_file,'r')]
    # all_answers = []
    # for line_idx, res in enumerate(merge_data):
    #     question_id = res['question_id']
    #     text = res['text'].rstrip('.').lower()
    #     all_answers.append({"questionId": question_id, "prediction": text})
    # json.dump(all_answers,open(args.answers_file,'w'))
    # eval_model(args)

    test_split = args.question_file
    # os.makedirs(os.path.dirname(dst), exist_ok=True)
    results=merge_data
    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    # with open(dst, 'w') as f:
    json.dump(all_answers, open(args.answers_file, 'w'))

