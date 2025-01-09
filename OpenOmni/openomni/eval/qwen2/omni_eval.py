import argparse
import torch
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import json
import sys
sys.path.append("/mnt/workspace/lr/workspace/OpenOmni")
from tqdm import tqdm
import shortuuid
import whisper
from PIL import Image
import os
import pandas as pd
import json
from tqdm import tqdm
import requests
import time
from io import BytesIO
import urllib
import numpy as np
from uuid import uuid4
import base64
import re
from tqdm import tqdm
from multiprocessing import Pool

from openomni.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from openomni.conversation import conv_templates, SeparatorStyle
from openomni.mm_utils import process_images
from openomni.model.builder import load_pretrained_qwen_model
from openomni.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader

import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

def eval_model(args):
    # Model
    disable_torch_init()
    # print(args.model_path)
    model_path = os.path.expanduser(args.model_path)
    # model_path=args.model_path
    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, args.model_base, is_lora=args.is_lora)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    tokenizer.chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")

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
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    use_speech=False
    for line in tqdm(questions):

        idx = line["index"]
        image_file = line["image_path"]
        qs =f"<speech>\n Please answer the questions in the user's input speech. Carefully read the following question and select the letter corresponding to the correct answer. Highlight the applicable choices without giving explanations. \n {line['question']}.\n."
        # qs =f"Speech:\n {line['audio content']} \n  Please answer the questions in the user's input speech. Carefully read the following question and select the letter corresponding to the correct answer. \n{line['question']}.\n."
        # qs=f"<speech>\n Please answer the following question based on the given image and audio:\n{line['question']}.\nPlease choose only one answer from the following options:\n"
        for i,j in enumerate(line['options']):
            qs+=f"{chr(ord('A') + i)}. {j}\n"
        question=qs+"Answer with the option's letter from the given choices directly."
        speech_file = line["audio_path"]
        answer=line["answer"]

        prompt=question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        prompt = prompt.strip()

        question_prompt=prompt
        print(question_prompt)

        input_id=[]
        system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message},
        {"role" : "user", "content" : question_prompt}],add_generation_prompt=True)
        for idx, encode_id in enumerate(input_id):
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX

        input_ids = torch.tensor([input_id], dtype=torch.long)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
    
        image = Image.open(os.path.join(
            args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images(
            [image], image_processor, model.config)[0]

        speech = whisper.load_audio(os.path.join(
            args.speech_folder, speech_file))
        if args.input_type == "raw":
            speech = torch.from_numpy(speech)
        elif args.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)

        speech_length=torch.LongTensor([speech_tensor.shape[0]])
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True).unsqueeze(0)
        speech_length = speech_length.to(device='cuda', non_blocking=True)
        # print(speech_tensor.shape)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                faster_infer=True,
                streaming_unit_gen=False,
            )
            output_ids, output_units = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"H-{idx}\t{outputs}")
        
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": question_prompt,
                                   "text": outputs,
                                   "image": os.path.join(args.image_folder, image_file),
                                   "answer_id": ans_id,
                                   "answer": line['answer'],
                                   "source": "sss",
                                   "model_id": "llava_her",
                                   'type': line['audio type'],
                                   "metadata": {}}) + "\n")
    ans_file.close()

def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret

def prepare_inputs(inputs):
    input_msgs = []
    has_images = np.sum([x['type'] == 'image' for x in inputs])
    if has_images:
        content_list = []
        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                from PIL import Image
                img = Image.open(msg['value'])
                b64 = encode_image_to_base64(img, target_size=512)
                img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail='low')
                content_list.append(dict(type='image_url', image_url=img_struct))
        input_msgs.append(dict(role='user', content=content_list))
    else:
        assert all([x['type'] == 'text' for x in inputs])
        text = '\n'.join([x['value'] for x in inputs])
        input_msgs.append(dict(role='user', content=text))
    return input_msgs

def make_request(meta):

    api_base=""
    key=""
    assert len(api_base)>0 and len(key)>0, "set api_base and key first"
    # gpt_model="gpt-4o-2024-05-13"
    gpt_model = "gpt-4o-mini"

    idx, question =  meta
    generated = False

    attempt = 5
    answer="error"
    
    while attempt > 0 and generated == False:
        try:
            input_msgs = prepare_inputs([
                                        {"type": "text", "value": question}
                                    ])
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
            payload = dict(
                model=gpt_model,
                messages=input_msgs,
                max_tokens=10,
                n=1,
                temperature=0.0,
                seed=3407)
            response = requests.post(api_base, headers=headers, data=json.dumps(payload), timeout=60 * 10)
            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            resp_struct = json.loads(response.text)
            print(resp_struct)
            answer = resp_struct['data']['response']['choices'][0]['message']['content'].strip()
            generated = True      
        except:
            attempt -= 1
            time.sleep(0.1)
    # # print(answer)
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    return answer.lower()
        
def compute_acc(answer_file, gt_file):

    data=[json.loads(i) for i in open(answer_file,'r')]

    all_types=[]
    for i,j in zip(data,[json.loads(i) for i in open(gt_file,'r')]):
        i['type']=j['task type']
        all_types.append(j['task type'])

    k_data=data

    for x in data:
        print(x['text'],x['prompt'][x['prompt'].find(x['answer'])-3:x['prompt'].find(x['answer'])])

    output=['yes' if x['text'] in x['prompt'][x['prompt'].find(x['answer'])-3:x['prompt'].find(x['answer'])] else 'no' for x in data]

    # data=[(idx,f"Given the following question {i['prompt']}, the correct answer is {i['answer']}. Does the following answer correctly answers the question, answer:{i['text']}? Please answer yes or no in a single word") for idx,i in enumerate(data)]

    # with Pool(processes=50) as pool:
    #     output = list(tqdm(pool.imap(make_request, data), total=len(data)))

    # print(output)
    for i in set(all_types):
        num_correct, num_total = 0, 0
        for gpt_grade,j in zip(output,k_data):
            if j['type']==i:
                if "yes" in gpt_grade:
                    num_correct+=1
                num_total += 1
        print(f"The {i} accuracy is {num_correct/num_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage3_qwen/checkpoint-17500")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/workspace/lr/workspace/OmniBench/mm_data/image")
    parser.add_argument("--speech-folder", type=str, default="/mnt/workspace/lr/workspace/OmniBench/mm_data/audio")
    parser.add_argument("--question-file", type=str,default="/mnt/workspace/lr/workspace/OmniBench/dataset/batch-5_1142_20240817.jsonl")
    parser.add_argument("--answer-file", type=str,default="/mnt/workspace/lr/workspace/OmniBench/dataset/prediction_17500.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_qwen_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--square_eval", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)

    compute_acc(args.answer_file, args.question_file)
