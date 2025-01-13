'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry - Angry and Disgusted
    1: disgusted - Angry and Disgusted
    2: fearful - Fearful and Concerned
    3: happy - Positive
    4: neutral - Neutral
    5: other - Others
    6: sad - Confused and Negative
    7: surprised - Surprised and Curious
    8: unknown - 
'''

import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import re
import editdistance as ed
import torch
from tqdm import tqdm
import random

import argparse
import torch
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import json
from tqdm import tqdm
import shortuuid
import whisper
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
    
from openomni.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from openomni.conversation import conv_templates, SeparatorStyle
from openomni.mm_utils import process_images
from openomni.model.builder import load_pretrained_qwen_model
from openomni.flow_inference import AudioDecoder
from openomni.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader
import whisper
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torchaudio

import math
from evaluate import load

random.seed(8823)


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
    emotion_inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_base")
    # model_path=args.model_path
    audio_decoder = AudioDecoder(config_path=args.voice_config_path, 
                            flow_ckpt_path=args.flow_ckpt_path,
                            hift_ckpt_path=args.hift_ckpt_path,
                            device='cuda')
    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, args.model_base, is_lora=args.is_lora)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    tokenizer.chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # tokenizer.chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{''}}"
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
    support_emotion=["Angry and Disgusted","Angry and Disgusted","Fearful and Concerned","Positive","Neutral","Others","Confused and Negative","Surprised and Curious"]
    emotion_maps=["angry","disgusted","fearful","happy","neutral","other","sad","surprised"]
    emotion_maps={k:v for k,v in zip(emotion_maps,support_emotion)}
    questions=[x for x in questions if x['emotion'] in support_emotion]
    new_questions=[]
    for i in questions:
        for turn_idx in range(1,len(i['conversations'])//2):
            item={'emotion': i['emotion'], 'lang': i['lang']}
            item['prompt']=[]
            for j in i['conversations'][:2*turn_idx+1]:
                if j['from']=='human':
                    item['prompt'].append({"role":"user","content":j['value']})
                elif j['from']=='gpt':
                    item['prompt'].append({"role":"assistant","content":j['value']})
            new_questions.append(item)
    questions=new_questions
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    use_speech=False
    random.shuffle(questions)

    for line in tqdm(questions):

        answer=line["emotion"]
        prompt=line['prompt']


        image_file="./assets/example.png"
        speech_file="./assets/question.wav"
        
        # print(prompt)
        input_id=[]
        system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}
        ]+prompt,add_generation_prompt=True)

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
                streaming_unit_gen=False,
                faster_infer=False
            )
            output_ids, output_units = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.s2s:
            if args.speech_generator_type=="ar":
                output_units = output_units
            elif args.speech_generator_type=="ctc":
                output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
        
        print(f"H-{idx}\t{outputs}")
        print(f"T-{idx}\t{answer}")
        
        tts_speechs=[]

        audio_tokens=[int(x) for x in output_units.split(' ')]

        tts_token = torch.tensor(audio_tokens, device='cuda').unsqueeze(0)

        tts_speech = audio_decoder.offline_inference(tts_token)

        tts_speechs.append(tts_speech.squeeze())

        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        torchaudio.save(f"./assets/emotion_temp.wav", tts_speech.unsqueeze(0), 22050, format="wav")
        rec_result = emotion_inference_pipeline("./assets/emotion_temp.wav", output_dir="./assets", granularity="utterance", extract_embedding=False)
        ["Angry and Disgusted",
        "Angry and Disgusted",
        "Fearful and Concerned",
        "Positive",
        "Neutral",
        "Others",
        "Confused and Negative",
        "Surprised and Curious"]

        print(rec_result[0]['scores'])
        if answer=="Neutral" or answer=="Positive":
            fit_score=sorted(rec_result[0]['scores'])[-1] # skip neutral
            max_score_index = rec_result[0]['scores'].index(fit_score)
            selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
            choose_label=[emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown']
        elif answer=="Confused and Negative":
            fit_scores=sorted(rec_result[0]['scores'])[-2:] # skip neutral
            choose_label=[]
            for fit_score in fit_scores:
                max_score_index = rec_result[0]['scores'].index(fit_score)
                selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
                choose_label.append(emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown')
        elif answer=="Angry and Disgusted":
            fit_scores=sorted(rec_result[0]['scores'])[-5:] # skip neutral
            choose_label=[]
            for fit_score in fit_scores:
                max_score_index = rec_result[0]['scores'].index(fit_score)
                selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
                choose_label.append(emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown')
        elif answer=="Fearful and Concerned":
            fit_scores=sorted(rec_result[0]['scores'])[-8:] # skip neutral
            choose_label=[]
            for fit_score in fit_scores:
                max_score_index = rec_result[0]['scores'].index(fit_score)
                selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
                choose_label.append(emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown')
        elif answer=="Surprised and Curious":
            fit_scores=sorted(rec_result[0]['scores'])[-4:] # skip neutral
            choose_label=[]
            for fit_score in fit_scores:
                max_score_index = rec_result[0]['scores'].index(fit_score)
                selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
                choose_label.append(emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown')
        elif answer=="Others":
            fit_scores=sorted(rec_result[0]['scores'])[-8:] # skip neutral
            choose_label=[]
            for fit_score in fit_scores:
                max_score_index = rec_result[0]['scores'].index(fit_score)
                selected_label = rec_result[0]['labels'][max_score_index].split('/')[-1] # angry happy neutral sad <unk>
                choose_label.append(emotion_maps[selected_label] if selected_label!='<unk>' else 'unkown')



        print(f'T-{idx}\t{choose_label}')

        
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"prompt": prompt,
                                   "lang": line['lang'],
                                   "answer": answer,
                                   "units": output_units,
                                   'pre': choose_label,
                                   "metadata": {}}) + "\n",
                                   )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/openomni_stage3_qwen_ar/checkpoint-last")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--voice_config_path", type=str, default="ZhipuAI/glm-4-voice-decoder/config.yaml")
    parser.add_argument("--flow_ckpt_path", type=str, default="ZhipuAI/glm-4-voice-decoder/flow.pt")
    parser.add_argument("--hift_ckpt_path", type=str, default="ZhipuAI/glm-4-voice-decoder/hift.pt")
    parser.add_argument("--encoder_path", type=str, default="./checkpoints/openai-whisper/large-v3.pt")
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,default="./openomni/eval/qwen/openomni_emotion_val.json")
    parser.add_argument("--answer-file", type=str,default="./openomni/eval/qwen/openomni_emotion_val_pre.jsonl")
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
    parser.add_argument("--speech_generator_type", action="store_true", default="ar")
    args = parser.parse_args()

    eval_model(args) 

    data=[json.loads(x) for x in open(args.answer_file,"r").readlines()]

    acc={}

    for i in data:
        if i['lang']+i['answer'] not in acc:
            acc[i['lang']+i['answer']]=[1,0]
        else:
            acc[i['lang']+i['answer']][0]+=1
        
        if i['answer'] in i['pre']:
            acc[i['lang']+i['answer']][1]+=1
    for k,v in acc.items():
        print(f"{k}: {v[1]/v[0]*100}")

