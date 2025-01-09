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
from torch.utils.data import Dataset, DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import math
from evaluate import load


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

ds_collections={"wenetspeech":{"question_file":"./openomni/eval/wenetspeech_eval.json",
                                "answer_file":"./openomni/eval/wenetspeech_eval_asr_pre.jsonl",
                                "language":"zh",
                                "speech_folder": "/mnt/workspace/lr/datasets/asr/WeNetSpeech"},
                "librispeech":{"question_file":"./openomni/eval/librispeech_eval.jsonl",
                                "answer_file":"/./openomni/eval/librispeech_eval_asr_pre.jsonl",
                                "language":"en",
                                "speech_folder":"/mnt/workspace/lr/datasets/asr/LibriSpeech"},
                "aishell":{"question_file":"./openomni/eval/aishell2_eval.jsonl",
                                "answer_file":"./openomni/eval/aishell2_eval_asr_pre.jsonl",
                                "language":"zh",
                                "speech_folder":"/mnt/workspace/lr/datasets/asr"},
                "self_aishell":{"question_file":"./openomni/eval/self_aishell_eval.json",
                                "answer_file":"./openomni/eval/aishell2_eval_asr_self_pre.jsonl",
                                "language":"zh",
                                "speech_folder":""}}

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
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    use_speech=False
    for line in tqdm(questions[:1000]):
        if ("dev-clean" in line["audio"] or "dev-other" in line["audio"]) or ('Mic' in line["audio"] or 'Android' in line["audio"]):
            continue
        speech_file = line["audio"]
        answer=line["gt"]

        image_file="./assets/example.png"

        if ds_collections[args.dataset]['language']=="zh":
            prompt="<speech>\n 请将用户输入的语音一字一句地直接转译成对应的文本\n"

            
        elif ds_collections[args.dataset]['language']=="en":
            prompt="<speech>\n Please translate the user input's speech into the corresponding text \n"

        else:
            print("no surppot")

        input_id=[]
        system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message},
        {"role" : "user", "content" : prompt}],add_generation_prompt=True)

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
                faster_infer=True
            )
            output_ids, output_units = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"H-{idx}\t{outputs}")
        print(f"T-{idx}\t{answer}")
        # if args.s2s:
        #     print(f"U-{idx}\t{output_units}")
        
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"prompt": prompt,
                                   "text": outputs,
                                   "answer": answer,
                                   "source": line['source'],
                                   "units": output_units,
                                   "metadata": {}}) + "\n",
                                   )
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage3_qwen_asr/checkpoint-9000")
    # parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/llava-v1.6-8b_llama3-1-8b_clip-large-336_debug_ablation_sft_evol_3d1_en_zh_new/checkpoint-16404")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--dataset", type=str, default="self_aishell")
    parser.add_argument("--speech-folder", type=str, default="/mnt/workspace/lr/workspace/OmniBench/mm_data/audio")
    parser.add_argument("--question-file", type=str,default="")
    parser.add_argument("--answer-file", type=str,default="")
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

    if args.dataset=='all':
        for dataset in ['aishell','librispeech','wenetspeech']:
            args.question_file=ds_collections[args.dataset]['question_file']
            args.answer_file=ds_collections[args.dataset]['answer_file']
            args.speech_folder=ds_collections[args.dataset]['speech_folder']

            eval_model(args) 

            results_dict=dict()

            results = [json.loads(q) for q in open(
                    os.path.expanduser(args.answer_file), "r")]

            # results=json.load(open(args.answer_file,"r"))
            # results=[i for i in results if 'trans_text' in i]
            # warnings.filterwarnings("ignore")

            normalizer = BasicTextNormalizer()
            wer_metric = load("wer")
            cer_metric = load("cer")    

            for item in tqdm(results):
                source = item["source"]
                results_dict.setdefault(source, []).append(item)

            lan = ds_collections[args.dataset]['language']

            # print(results_dict.keys())

            for source in results_dict:
                prediction_list, reference_list = [], []
                results_list = results_dict[source]
                # print(len(results_list))
                for result in results_list:
                    # print(result)
                    gt = result["answer"]
                    response = result["text"]

                    normalized_prediction = normalizer(response)
                    normalizer_reference = normalizer(gt)

                    prediction_list.append(normalized_prediction)
                    reference_list.append(normalizer_reference)

                wer_ortho = 100 * wer_metric.compute(
                    references=reference_list, predictions=prediction_list
                )

                cer_ortho = 100 * cer_metric.compute(
                    references=reference_list, predictions=prediction_list
                )

                # print(reference_list, prediction_list)

                print(f"SOURCE: {source} WER:{wer_ortho}, CER:{cer_ortho}")
    else:
        args.question_file=ds_collections[args.dataset]['question_file']
        args.answer_file=ds_collections[args.dataset]['answer_file']
        args.speech_folder=ds_collections[args.dataset]['speech_folder']

        eval_model(args) 

        results_dict=dict()

        results = [json.loads(q) for q in open(
                os.path.expanduser(args.answer_file), "r")]

        # results=json.load(open(args.answer_file,"r"))
        # results=[i for i in results if 'trans_text' in i]
        # warnings.filterwarnings("ignore")

        normalizer = BasicTextNormalizer()
        wer_metric = load("wer")
        cer_metric = load("cer")    

        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)

        lan = ds_collections[args.dataset]['language']

        # print(results_dict.keys())

        for source in results_dict:
            prediction_list, reference_list = [], []
            results_list = results_dict[source]
            # print(len(results_list))
            for result in results_list:
                # print(result)
                gt = result["answer"]
                response = result["text"]

                normalized_prediction = normalizer(response)
                normalizer_reference = normalizer(gt)

                prediction_list.append(normalized_prediction)
                reference_list.append(normalizer_reference)

            wer_ortho = 100 * wer_metric.compute(
                references=reference_list, predictions=prediction_list
            )

            cer_ortho = 100 * cer_metric.compute(
                references=reference_list, predictions=prediction_list
            )

            # print(reference_list, prediction_list)

            print(f"SOURCE: {source} WER:{wer_ortho}, CER:{cer_ortho}")
