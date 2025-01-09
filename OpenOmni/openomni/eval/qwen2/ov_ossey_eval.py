import json
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np
import argparse
import pyarrow.parquet as pq
from decord import VideoReader
import decord
import tempfile
from PIL import Image
import numpy as np
import io
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

from openomni.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from openomni.conversation import conv_templates, SeparatorStyle
from openomni.mm_utils import process_images
from openomni.model.builder import load_pretrained_qwen_model
from openomni.utils import disable_torch_init
from openomni.utils.data import Dataset, DataLoader
import time
import math


decord.bridge.set_bridge("torch")

def load_video(video_path, n_frms=8, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_path)
            video_path = temp_file.name

    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if isinstance(temp_frms, np.ndarray) else temp_frms
    frms = tensor_frms.cpu().numpy()  # Convert to numpy (T, H, W, C)

    # print(frms.shape)

    images = []
    for frame in frms:
        image = Image.fromarray(frame)
        image_rgb = image.convert('RGB')
        images.append(image_rgb)
    
    # images=[concat_images_horizontally_with_margin(images)]
    # image_sizes=[images[0].size]

    return images

def concat_images_horizontally_with_margin(images, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - output_filename: String, the filename to save the concatenated image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - None. The result is saved to 'output_filename'.
    """
    # images = [Image.open(filename) for filename in image_filenames]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    return new_image

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'{choice}' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = 'A'
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
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
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def read_parquet_file(file_path):
        table = pq.read_table(file_path)
        return table

def preprocess(args):
    file_path = [
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part1.parquet',
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part2.parquet',
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part3.parquet',
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part4.parquet',
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part5.parquet',
                'AV-Odyssey/AV_Odyssey_Bench/av_odyssey_part6.parquet'
                ]
    file_path=[os.path.join(args.question_file,x) for x in file_path]
    question_type_dict = {}    
    for par_file in file_path:
        table = read_parquet_file(par_file)
        df = table.to_pandas()

        for index, row in df.iterrows():

            row_dict = row.to_dict()
            question_type_id = row_dict.get('question_type_id')
            row_dict['subpart'] = row_dict.pop('subfield')
            row_dict['image_path'] = [row_dict['image_1'], row_dict['image_2'],  row_dict['image_3'], row_dict['image_4']] if row_dict['image_2'] else [row_dict['image_1']]
            row_dict['audio_path'] = [row_dict['audio_1'], row_dict['audio_2'] , row_dict['audio_3'], row_dict['audio_4']] if row_dict['audio_2'] else [row_dict['audio_1']]
            row_dict['option_A'] = row_dict['options'][0]
            row_dict['option_B'] = row_dict['options'][1]
            row_dict['option_C'] = row_dict['options'][2]
            row_dict['option_D'] = row_dict['options'][3]
            row_dict['video_path'] = [row_dict.pop('video_1')]

            row_dict.pop('options')

            # print(row_dict['question'])

            if question_type_id not in question_type_dict:
                question_type_dict[question_type_id] = []
            question_type_dict[question_type_id].append(row_dict)
    return question_type_dict

def eval_model(args):
    model_name='openomni'
    question_type_dict=preprocess(args)
    question_id_list = [i for i in range(1, 27)]
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
    all_evaluation_results = []
    for current_question_id in question_id_list:
        current_json_data = question_type_dict[current_question_id]
        task_name = 'task' + str(current_question_id)
        evaluation_result=[]
        for current_data in current_json_data:
            if "image" in current_data['data_type']:
                images=[Image.open(io.BytesIO(image_path)).convert('RGB') for image_path in current_data['image_path']]
                image_sizes=[images[0].size]
                if len(images)>1:
                    # MMMU or BLINK
                    images=[concat_images_horizontally_with_margin(images)]
                    image_sizes=[images[0].size]
                question = "<image> \n"
            elif "video" in current_data['data_type']:
                images=load_video(current_data['video_path'][0])
                image_sizes=[images[0].size]
                if len(images)>1:
                    # MMMU or BLINK
                    images=[concat_images_horizontally_with_margin(images)]
                    image_sizes=[images[0].size]
                question = "<image> \n"
            else:
                images=[Image.new('RGB', (400, 400), (0, 0, 0))]
                image_sizes=[images[0].size]
                question = ""

            audios=[io.BytesIO(audio_path) for audio_path in current_data['audio_path']] 
            options=[current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']]
            
            question+='Here are some images and audios below: \n'
            temp_qs=current_data['question']
            for index in range(1, 1 + len(audios)):
                question+= f'[audio{index}]: <speech> \n'
                # temp_qs=temp_qs.replace(f"[audio{index}]",f"[speech{index}]")

            option_text = "A:" + options[0] + "\n" + "B:" + options[1] + "\n" + "C:" + options[2] + "\n" + "D:" + options[3] + "\n" 
            text = question +temp_qs + "\n" + option_text + "Carefully read the following question and select the letter corresponding to the correct answer. Highlight the applicable choices without giving explanations. Answer with the option's letter from the given choices directly."
            input_id=[]
            system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
            # print(text)
            input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message},
                                                    {"role" : "user", "content" : text}], add_generation_prompt=True)
            # input_id += tokenizer.apply_chat_template([])[1:]
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == speech_token_index:
                    input_id[idx] = SPEECH_TOKEN_INDEX

            input_ids = torch.tensor([input_id], dtype=torch.long)
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            image_tensors = process_images(
                images, image_processor, model.config)[0]
            speech_tensors=[]
            speech_lengths=[]
            for speech_file in audios:
                speech_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(speech_file.read())
                    temp_file_path = temp_file.name
                speech = whisper.load_audio(temp_file_path)
                speech = whisper.pad_or_trim(speech)
                speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)
                speech_tensors.append(speech_tensor)
                speech_lengths.append(speech_tensor.shape[0])
                os.remove(temp_file_path)
            speech_tensors = torch.stack(speech_tensors).to(dtype=torch.float16, device=torch.cuda.current_device(), non_blocking=True)
            speech_lengths = torch.LongTensor(speech_lengths).to(device=torch.cuda.current_device(), non_blocking=True)
            input_ids = input_ids.to(device=torch.cuda.current_device(), non_blocking=True)
            image_tensors=image_tensors.to(dtype=torch.float16, device=torch.cuda.current_device(), non_blocking=True)
            with torch.inference_mode():
                output_ids,output_units = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    speech=speech_tensors,
                    speech_lengths=speech_lengths,
                    do_sample=False,
                    num_beams=args.num_beams,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    streaming_unit_gen=False,
                    faster_infer=True)
            prediction = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True)[0].strip()
            print(prediction)
            evaluation_result.append({
                "question_id": current_data['question_id'],
                "answer": current_data['answer'],
                "prediction": prediction,
                "current_question_id": current_question_id,
                "subpart": current_data["subpart"]
            })

        # clean the answer, following MMMU (https://github.com/MMMU-Benchmark/MMMU)
        cleaned_evaluation_data = []
        for data, prediction in zip(current_json_data, evaluation_result):
            option_list = {'A': data['option_A'], 'B': data['option_B'], 'C': data['option_C'], 'D': data['option_D']}
            answer = parse_multi_choice_response(prediction['prediction'], ['A', 'B', 'C', 'D'], option_list)
            prediction['prediction'] = answer
            cleaned_evaluation_data.append(prediction)

        all_evaluation_results = all_evaluation_results + cleaned_evaluation_data

    with open(args.answer_file, 'w') as f:
        for item in all_evaluation_results:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage3_qwen_3/checkpoint-18400")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,default="/mnt/workspace/lr/datasets/checkpoints/")
    parser.add_argument("--answer-file", type=str,default="/mnt/workspace/lr/workspace/LLaVA_Her/ov_odyssey_answer2.jsonl")
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
    data=[json.loads(x) for x in open(args.answer_file,"r").readlines()]
    s=set([x['subpart'] for x in data])
    for xxx in s:
        a=0
        kk=[x for x in data if x['subpart']==xxx]
        for i in kk:
            if i['prediction'] in i['answer']:
                a+=1
        print(f"type: {xxx} acc: {a/len(kk)*100}")

   