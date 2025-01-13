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


def generate_prompt(d, response):
    instruction = d['instruction']
    weight = d['component_weight'] * 1
    d['num_of_component'] = len(d['components'])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d['num_of_component'] == 1:
        components = '''The first component is:' ''' + d['components'][0] + "'"  
        score = '''The first component is worth ''' + weight[0] + ' scores.'
    elif d['num_of_component'] == 2:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + "'" 
        score = '''The first and second component is each worth ''' + weight[0] + ' and ' + weight[1]+ ' scores.'
    elif d['num_of_component'] == 3:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] + "'" 
        score = '''The first second, and third component is each worth ''' + weight[0] + ', ' + weight[1]+ ' and ' + weight[2] + ' scores.'
    elif d['num_of_component'] == 4:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] + "'" 
        score = '''The first second, third, and fourth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ' and ' + weight[3] + ' scores.'
    elif d['num_of_component'] == 5:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] +  '''', and the fifth component is:' ''' + d['components'][4] + "'" 
        score = '''The first second, third, fourth and fifth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ', ' + weight[3] + ' and ' + weight[4] + ' scores.'      
    return '''Here is an instruction for a multimodal LLM: ' ''' + instruction + ''' You need to grade if the response from the model follows each component of the instruction. ''' + components + ''' The response is:' '''  + response +  '''' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. ''' + score + ' List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/10, total score: z/10. Then explain your reasons.'

def process_rawscore(component_type, raw_score):
    component_type=eval(component_type)
    first_sentence = raw_score.split('''.''')[0].split(''',''')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(''':''')[1][1:].split('''/''')
        score = int(score_[0])/int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i].split(''':''')[1][1:].split('''/''')
    total_score = int(score_[0])/int(score_[1])
    score_dict['total_score'] = total_score
    return score_dict  

def get_score_dict(df, column_name):
    cat_score_dict = {}
    for i in range(len(df)):
        try:
            # print(df[column_name][i])
            score_dict = process_rawscore(df['component_type'][i], df[column_name][i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val)/len(val)
    return cat_score_dict_average

df = pd.read_json('datasets/ml-mia-bench/instruction_benchmark_all_image.json')
print(df.head())

ans_file="answers/llava_llama_3d1_mia_predition.jsonl"

answers = [json.loads(q) for q in open(ans_file, 'r')]

df_llava = pd.DataFrame(answers)
print(df_llava.head())

df_llava['score_raw'] = [_ for _ in range(len(df_llava))]

def get_d(i):
# for i in tqdm(range(len(df_llava))):
    d = {}
    for col in df.columns:
        d[col] = df[col][i]
    return d

data=[(df_llava['text'][i], df_llava['image'][i], get_d(i)) for i in tqdm(range(len(df_llava)))]

print(len(data))
print(data[0])

def make_request(meta):

    api_base=""
    key=""

    assert len(api_base)>0 and len(key)>0, "make sure tha both api_base and key are configured correctly"

    gpt_model="gpt-4o-2024-05-13"
    # gpt_model = "gpt-4o-mini"


    response, image, d= meta
    question =  generate_prompt(d, response)
    generated = False
    assert response != 'error'

    attempt = 5
    answer="error"
    
    while attempt > 0 and generated == False:
        try:
            input_msgs = prepare_inputs([
                                        {"type": "text", "value": question},
                                        {"type": "image","value": image},
                                    ])
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
            payload = dict(
                model=gpt_model,
                messages=input_msgs,
                max_tokens=2048,
                n=1,
                temperature=1.0,
                seed=3407)
            response = requests.post(api_base, headers=headers, data=json.dumps(payload), timeout=60 * 3)
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

    return answer


with Pool(processes=50) as pool:
    output = list(tqdm(pool.imap(make_request, data), total=len(data)))

for i,answer in enumerate(output):
    df.loc[i, "score_raw"] = answer


print(df['score_raw'])
df.to_csv('./mia.csv', index=False)
print(get_score_dict(df,'score_raw'))

df = pd.read_csv('./mia.csv')
# print(df.head())
print(get_score_dict(df,'score_raw'))
