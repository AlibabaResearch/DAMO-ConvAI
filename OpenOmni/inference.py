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
import time
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
    print(_toks,len(_toks))
    # deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    deduplicated_toks = [v for i, v in enumerate(_toks)]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def eval_model(args):
    # Model
    disable_torch_init()
    # print(args.model_path)
    model_path = os.path.expanduser(args.model_path)
    # model_path=args.model_path
    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, args.model_base, is_lora=args.is_lora)
    # state_dict={k:v for k,v in model.named_parameters()}
    # state_dict=get_w(state_dict, "model.mm_projector")
    # torch.save(state_dict,"openomni/ckpt/pretrained/mm_projector/mm_projector.pt")
    use_speech=True

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

    speech_file = "./assets/question.wav"
    image_file="./assets/example.png"
    question="<image>\n Tell me something about the object in this image."
    prompt=question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    prompt = prompt.strip()

    if use_speech:
        prompt="<image>\n <speech>\n Please answer the questions in the user's input speech"
    question_prompt=prompt
    input_id=[]
    system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message},
    {"role" : "user", "content" : question_prompt}],
    add_generation_prompt=True)
    
    # input_id += encode_id
    for idx, encode_id in enumerate(input_id):
        if encode_id == image_token_index:
            input_id[idx] = IMAGE_TOKEN_INDEX
        if encode_id == speech_token_index:
            input_id[idx] = SPEECH_TOKEN_INDEX

    input_ids = torch.tensor([input_id], dtype=torch.long)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    image = Image.open(os.path.join(
        '', image_file)).convert('RGB')
    image_tensor = process_images(
        [image], image_processor, model.config)[0]

    speech = whisper.load_audio(os.path.join('',speech_file))
    if args.input_type == "raw":
        speech = torch.from_numpy(speech)
    elif args.input_type == "mel":
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)

    speech_length=torch.LongTensor([speech_tensor.shape[0]])
    speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True).unsqueeze(0)
    speech_length = speech_length.to(device='cuda', non_blocking=True)
    with torch.inference_mode():
        time1=time.time()
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
            faster_infer=False if args.s2s else True
        )
        time2=time.time()
        output_ids, output_units = outputs

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if args.s2s:
        if args.speech_generator_type=="ar":
            output_units=output_units
        elif args.speech_generator_type=="ctc":
            output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

    print(f"H-{time2-time1}-{idx}\t{outputs}")
    if args.s2s:
        print(f"U-{idx}\t{output_units}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage2_qwen_2/checkpoint-2180")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_qwen2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=True)
    parser.add_argument("--speech_generator_type", action="store_true", default="ar")
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--square_eval", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)
