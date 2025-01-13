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

    api_base=""
    key=""

    assert len(api_base)>0 and len(key)>0, "make sure tha both api_base and key are configured correctly"

    # gpt_model="gpt-4o-2024-05-13"
    gpt_model = "gpt-4o-mini"

    source, question =  meta
    generated = False
    # assert response != 'error'

    # print(question)
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
                temperature=1.0,
                seed=3407)
            response = requests.post(api_base, headers=headers, data=json.dumps(payload), timeout=60 * 10)
            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            resp_struct = json.loads(response.text)
            # print(response)
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
    return (source, answer.lower())
        
answer_file="llava_qwen_cvbench_predition.json"

data=[json.loads(i) for i in open(answer_file,'r')]

data=[(i['source'],f"Given the following question {i['prompt']}, the correct answer is {i['answer']}. Does the following answer correctly answers the question, answer:{i['text']}? Please answer yes or no in a single word") for idx,i in enumerate(data)]

with Pool(processes=50) as pool:
    output = list(tqdm(pool.imap(make_request, data), total=len(data)))

# print(output)

num_correct_2d_ade, num_total_2d_ade = 0, 0
num_correct_2d_coco, num_total_2d_coco = 0, 0
num_correct_3d, num_total_3d = 0, 0
# Continue with the processing of the JSONL file

for (source, gpt_grade) in output:

    # index += 1
    if source =='ADE20K':
        num_total_2d_ade+=1
        if "yes" in gpt_grade:
            num_correct_2d_ade += 1
    if source =='COCO':
        num_total_2d_coco+=1
        if "yes" in gpt_grade:
            num_correct_2d_coco += 1
    if source =='Omni3D':
        num_total_3d+=1
        if "yes" in gpt_grade:
            num_correct_3d += 1

combined_accuracy=num_correct_2d_coco/num_total_2d_coco/4+num_correct_2d_ade/num_total_2d_ade/4+num_correct_3d/num_total_3d/2
# Print the results
print(f"CV-Bench Accuracy: {combined_accuracy:.4f}")
