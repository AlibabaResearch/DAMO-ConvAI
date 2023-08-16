#import some packages and reward funcs
import os
import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
import metrics2
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM
)
from infer_func_now import setup_seed, generate_pipeline
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--directory', default="best_checkpoint", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])# **accelerator_log_kwargs)
    rank = int(os.environ['RANK'])
    rank_sum = accelerator.num_processes
    model_name_or_path = os.path.join("..", "checkpoints", f"index_{args.index}", f"stage_{args.stage}", f"{args.directory}")
    model_device = "cuda:{}".format(rank)

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=model_config, torch_dtype=torch.bfloat16).to(model_device)
    if accelerator.is_main_process:
        print(type(model))
        print(model.config)
    if model.config.architectures[0].lower() == "llamaforcausallm":
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    tokenizer.pad_token=tokenizer.eos_token,
    tokenizer.pad_token_id=tokenizer.eos_token_id,
    tokenizer.sep_token = "<sep>"
    model.resize_token_embeddings(len(tokenizer))
      
    print(model.dtype)
    torch.cuda.empty_cache()
    model.eval()
    print(f"Rank {rank} is activated...")
    if accelerator.is_main_process:
        file_name = "test.json"
        save_path = os.path.join("inference_res/cache", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
        if os.path.exists(save_path):
            os.remove(save_path)
    accelerator.wait_for_everyone()
    
    file_name = "test.json"
    file_path = os.path.join("..", "data", "summarize_test", file_name)
    with open(file_path, "r", encoding='utf-8') as f:
        infer_data = {line_index: json.loads(l) for line_index, l in enumerate(f.readlines()) if (line_index-rank) % rank_sum == 0}

    for line_index in infer_data:
        infer_data[line_index]["line_index"] = line_index
    infer_data = [infer_data[line_index] for line_index in infer_data]

    prompts = [l['prefix'][0] for l in infer_data]
    
    setup_seed()
    generated_suffixes, truncated_prompts = generate_pipeline(model, tokenizer, prompts, add_special_tokens=True)
    setup_seed()        
    save_path = os.path.join("inference_res/cache", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
    
    for index in range(len(infer_data)):
        infer_data[index]['infer'] = {"t": generated_suffixes[index]}
    with open(save_path, 'a', encoding='utf-8') as f:
        for line in infer_data:
            content = json.dumps(line, ensure_ascii=False)
            f.write(content+'\n')
    
    accelerator.wait_for_everyone()
    
    print("")
    if accelerator.is_main_process:
        print("Eval on {}".format(file_name))
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()