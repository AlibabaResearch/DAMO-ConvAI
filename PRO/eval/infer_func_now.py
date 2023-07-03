import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def generate_pipeline(model, tokenizer, prompts, add_special_tokens=False, gen_kwarg={"max_new_tokens": 128, "num_beams": 1, "do_sample": False,}, batch_size = 28):
    def pipeline(prompts):
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        model_inputs = tokenizer(
            prompts,
            max_length=512-128, 
            truncation=True, 
            add_special_tokens=add_special_tokens, 
        )
        truncated_prompts = tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
        model_inputs = tokenizer(
            truncated_prompts, 
            max_length=512-128,
            truncation=True, 
            add_special_tokens=add_special_tokens,
            padding=True,
            return_tensors="pt"
        )
        truncated_prompts = tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
        prompts_size = [len(s) for s in truncated_prompts]
        return model_inputs, prompts_size, truncated_prompts
    
    model_inputs, prompts_size, truncated_prompts = pipeline(prompts)
    text_res = []
    for index in tqdm.tqdm(range(0, len(model_inputs["input_ids"]), batch_size)):
        if len(model_inputs["input_ids"]) - index < batch_size:
            batch_size = len(model_inputs["input_ids"]) - index
        
        batch = {key: model_inputs[key][index:index+batch_size].to(model.device) for key in model_inputs}
        with torch.no_grad():
            ts = model.generate(
                **batch,
                **gen_kwarg,
                pad_token_id=tokenizer.pad_token_id,
                
            ).cpu().detach()
        text_res.append(ts)
        
    for index in range(len(text_res)):
        text_res[index] = tokenizer.batch_decode(
            text_res[index], 
            skip_special_tokens=True
        )

    text_res = sum(text_res, [])
    for index in range(len(text_res)):
        text = text_res[index]
        assert truncated_prompts[index].rstrip() in text
        text = text.replace(truncated_prompts[index].rstrip(), "").strip()
        # text = text[prompts_size[index]:].strip()
        for stop in ["Human:", "human:", "Assistant:", "assistant:"]:
            stop_ix = text.find(stop)
            if stop_ix >= 0:
                text = text[:stop_ix].rstrip()
        text_res[index] = text
    
    return text_res, truncated_prompts