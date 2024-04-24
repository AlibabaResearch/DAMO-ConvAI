"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial
import json

from datasets import Dataset
import datasets
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
import torch

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate, add_padding_token
from conversation import get_conv_template


def apply_conv_template(example, template_type):
    # preprocess instructions into prompted inputs
    conv = get_conv_template(template_type)
    conv.append_message(conv.roles[0], example['instruction'])
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    example.update({
        "prompt": prompt
    })

    return example

def generate_responses_batched(example, model, tokenizer, kwargs):
    prompt = example['prompt']
    print(prompt)
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.to(model.device)
    with torch.no_grad():
        model_output = model.generate(**encoding, **kwargs)
        input_len = encoding.input_ids.shape[-1]
        model_output = model_output[:, input_len:].cpu()
        decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    example.update({"output": decoded_output}) 
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama/7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="llama/7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/self-instruct-val(processed).jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/answers/self-instruct_llama7B.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1 # hardcode the vocab size for llama... 

    model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
        )
    add_padding_token(tokenizer)
    
    ## set the models to eval mode
    model = model.eval()
    
    
    if "dolly" in args.file_path or "vicuna" in args.file_path or "user_oriented_instructions" in args.file_path:
        tasks = load_jsonlines(args.file_path)
        raw_data = Dataset.from_list(tasks)
        if "dolly" in args.file_path:
            question_sources = "dolly"
        elif "vicuna" in args.file_path:
            question_sources = "vicuna"
            raw_data = raw_data.rename_column("text", "instruction")
        elif "user_oriented_instructions" in args.file_path:
            question_sources = "alpaca"
    elif "alpaca_eval" in args.file_path:
        raw_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        # set generator field to model name
        raw_data = raw_data.map(lambda x: {"generator": args.model_name if args.model_name else args.model})

    # reduce number of examples for debugging
    if args.debug:
        raw_data = raw_data.select(range(4))

    ## preprocess
    eval_preproc = partial(apply_conv_template, template_type=args.template_type)
    raw_data = raw_data.map(eval_preproc)

    ## run generation
    generate_kwargs = dict(max_length=2048, do_sample=False, top_p=0.1, 
                           num_return_sequences=1, temperature=0.1, 
                           repetition_penalty=1.2)
    generate = partial(generate_responses_batched, 
                       model=model,  
                       tokenizer=tokenizer,
                       kwargs=generate_kwargs)

    dataset_w_responses = raw_data.map(generate,
                                            batched=True,
                                            batch_size=args.batch_size)
    if "alpaca_eval" in args.file_path:
        # stringify the metadata, so that it becomes hashable
        dataset_w_responses = dataset_w_responses.map(lambda x: {"metadata": json.dumps(x["metadata"])})
        dataset_w_responses.to_json(args.save_file_name, orient="records", lines=False, indent=True)
        
    else:
        dataset_w_responses.to_json(args.save_file_name)



