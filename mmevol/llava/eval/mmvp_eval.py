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
    api_base = "http://47.88.8.18:8088/api/ask"
    key = ""
    gpt_model = "gpt-4o-2024-05-13"

    idx, question =  meta
    generated = False
    # assert response != 'error'

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
            answer = resp_struct['data']['response']['choices'][0]['message']['content'].strip()
            # df_llava['score_raw'][i] = answer
            # df.loc[i, "score_raw"] = answer
            # print(answer)
            generated = True

            # print(response.choices[0].message.content.strip())
            
        except:
            attempt -= 1
            time.sleep(0.1)
    # print(answer)
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    # if yes_no_regex.match(answer):
    return answer.lower()
    # else:
    #     return "Could not determine yes or no."
        
# answer_file="/mnt/workspace/lr/datasets/checkpoints/MMVP/MMVP/baseline_answer.json"
# answer_file="/mnt/workspace/lr/answers/llava_qwen_mmvp_predition.json"

answer_file="/mnt/workspace/lr/workspace/OmniBench/dataset/llave_her.jsonl"

data=[json.loads(i) for i in open(answer_file,'r')]



data=[(idx,f"Given the following question {i['prompt']}, the correct answer is {i['answer']}. Does the following answer correctly answers the question, answer:{i['text']}? Please answer yes or no in a single word") for idx,i in enumerate(data)]

with Pool(processes=50) as pool:
    output = list(tqdm(pool.imap(make_request, data), total=len(data)))

print(output)
for i in set(all_types):
    
    for j in data:
        if j['type']==i
num_correct, num_total = 0, 0
# Continue with the processing of the JSONL file
index=0
round_correct=0

for gpt_grade in output:

    index += 1

    if "yes" in gpt_grade:
        round_correct += 1
    if index == 2:
        index = 0
        if round_correct == 2:
            num_correct += 1
        round_correct = 0

        num_total += 1

print(f"The accuracy is {num_correct/num_total}")