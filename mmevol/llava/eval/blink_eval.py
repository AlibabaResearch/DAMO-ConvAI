import os
import json
import requests
import time
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

    api_base="http://47.88.8.18:8088/api/ask"
    key = ""
    gpt_model="gpt-4o-2024-05-13"

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
    # else:
    #     return "Could not determine yes or no."
        
answer_file="workspace/MiniCPM-V/blink_prediction.jsonl"
# answer_file="workspace/Open-LLaVA-NeXT/eval_mm/vlmevalkit/LLaVA-Llama3-V-1_6_ablation_evol_14k_evol/LLaVA-Llama3-V-1_6_ablation_evol_14k_evol_BLINK.xlsx"
if not answer_file.endswith(".xlsx"):

    data=[json.loads(i) for i in open(answer_file,'r')]
    # data=json.load(open(answer_file,"r"))
else:
    import pandas as pd

    def get_prompt(data,idx):
        q=data.loc[idx,'question']
        prompt=q+'\n\n'
        for k in ['A','B','C','D']:
            if type(data.loc[idx,k])==str:
                prompt+=f"{k}. {data.loc[idx,k]} \n"
        return prompt

    data=pd.read_excel(answer_file)
    data=[{'source':data.loc[i,'category'],
    'prompt':get_prompt(data,i),
    'answer':data.loc[i,'answer'],
    'text':data.loc[i,'prediction'],} for i in range(len(data))]

data=[(i['source'],f"Given the following question {i['prompt']}, the correct answer is {i['answer']}. Does the following answer correctly answers the question, answer:{i['text']}? Please answer yes or no in a single word") for idx,i in enumerate(data)]

with Pool(processes=50) as pool:
    output = list(tqdm(pool.imap(make_request, data), total=len(data)))

# print(output)

# Continue with the processing of the JSONL file
all_sources=['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']
correct_num_dict, num_dict={k:0 for k in all_sources},{k:0 for k in all_sources}

for (source, gpt_grade) in output:

    # index += 1
    have=False
    for k_source in all_sources:
        if k_source in source:
            num_dict[k_source]+=1
            if "yes" in gpt_grade:
                correct_num_dict[k_source] += 1
            have=True
    if not have:
        print(source)
for k in all_sources:
    print(f"{k}:{correct_num_dict[k]/num_dict[k]}")
combined_accuracy=sum([correct_num_dict[k]/num_dict[k] for k in all_sources])/len(all_sources)
# Print the results
print(f"BLINK Accuracy: {combined_accuracy:.4f}")


    # if index == 2:
    #     index = 0
    #     if round_correct == 2:
    #         num_correct += 1
    #     round_correct = 0

    #     num_total += 1

# print(f"The accuracy is {num_correct/num_total}")