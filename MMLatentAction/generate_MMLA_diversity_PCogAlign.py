import argparse
import copy
import json
import os
import random

from datasets import load_dataset
from sympy.physics.units import temperature
from tqdm import tqdm
from PIL import Image

import torch
from transformers import Qwen2_5_VLProcessor

from prompt_templates import convert_messages_of_MMRP,convert_messages_of_PCogAlign

from trl.data_utils import (
    prepare_multimodal_messages,
)


def collect_generated_ids(args, batch_inputs, model, generation_mode):
    all_generated_ids = []

    N_d = args.N_d

    if N_d == 1:
        if generation_mode == "action_diversity":
            generated_ids = model.generate(
                **batch_inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                PolicyInference=True,
                deterministic=True,
            )
        else:
            generated_ids = model.generate(
                **batch_inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        all_generated_ids.append(generated_ids)
    else:
        for _ in range(N_d):
            if generation_mode == "action_diversity":
                generated_ids = model.generate(
                    **batch_inputs,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # To ensure the generation quality

                    PolicyInference=True,
                    deterministic=False,
                    action_top_k=64,  # the half of the
                    # action_tau=1,
                    action_tau=0.1,
                    action_top_p=1,
                )
            else:
                generated_ids = model.generate(
                    **batch_inputs,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    # temperature=1,
                    temperature=0.1,
                )
            all_generated_ids.append(generated_ids)
    return all_generated_ids


def load_model(args, device):
    if args.generation_mode == "action_diversity":
        from modeling_qwen2_5_vl_PolicyModel import Qwen2_5_VLForConditionalGeneration
        print(f"Loading model from {args.ckpt_path} with dtype {args.dtype}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.ckpt_path,
            torch_dtype=args.dtype,
            device_map={"": device},
            trust_remote_code=True,
            lm_mode="PolicyActionWorldVLM"
        ).to(device).eval()
    elif args.generation_mode == "llm_diversity":
        from transformers import Qwen2_5_VLForConditionalGeneration

        print(f"Loading model from {args.ckpt_path} with dtype {args.dtype}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.ckpt_path,
            torch_dtype=args.dtype,
            device_map={"": device},
            trust_remote_code=True
        ).to(device).eval()
    else:
        NotImplementedError

    return model


import os
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from functools import partial

from qwen_vl_utils import process_vision_info


def collate_batch_input(examples, processor):
    # Get the texts and images, and apply the chat template
    texts = []
    image_inputs = []
    for example in examples:
        conversation_for_image_inputs = [
            {"role": "user", "content": [{"type": "image", "image": example["images"][0],
                                          "resized_height": 280, "resized_width": 420}], }]
        example_image_inputs, _ = process_vision_info(conversation_for_image_inputs)

        image_inputs.append(example_image_inputs)

        # this prepare_multimodal_messages() prepares the <image pad token>
        prepare_multimodal_messages(example["messages"], len(example["images"]))
        text = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        texts.append(text)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    return batch


def worker(rank: int, world_size: int, args, eval_data):
    """
    Each worker runs on one GPU (rank = GPU ID)
    - Loads model on its device
    - Processes its shard of eval_data
    - Saves partial results to tmp file
    """
    print(f"[Rank {rank}] Starting worker on GPU {rank}")

    device = torch.device(f"cuda:{rank}")  # since only 1 visible device

    # 🔽 Load model on this GPU
    model = load_model(args, device=device)  # ← YOU implement this
    model.eval()

    # 🔽 Shard dataset
    local_examples = eval_data[rank::world_size]  # round-robin split

    results = []
    batch_size = args.batch_size // world_size  # e.g., global bs=32 → local bs=8
    if batch_size < 1: batch_size = 1

    num_batches = (len(local_examples) + batch_size - 1) // batch_size

    print(
        f"[Rank {rank}] Processing {len(local_examples)} examples. Local batch size {batch_size}. Nume batches {num_batches}")

    for i in tqdm(range(num_batches), total=num_batches):
        batch_start = i * batch_size
        batch_examples = local_examples[batch_start: batch_start + batch_size]

        try:
            batch_inputs = collate_batch_input(batch_examples, args.processor)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                all_generated_ids = collect_generated_ids(args, batch_inputs, model, args.generation_mode)

            # Decode
            input_ids = batch_inputs["input_ids"]
            for ex_idx, example in enumerate(batch_examples):
                outputs = {}
                for div_idx, gen_ids in enumerate(all_generated_ids):
                    out_ids = gen_ids[ex_idx][input_ids[ex_idx].size(0):]
                    text = args.processor.decode(out_ids, skip_special_tokens=True).strip()
                    outputs[div_idx] = text

                results.append({
                    "id": example["id"],
                    "raw_aux_info": example["raw_aux_info"],

                    "prompt_conversation": example["messages"],
                    "model_diverse_output": outputs,
                    "ground_truth": example["expected_response"]
                })

        except Exception as e:
            print(f"[Rank {rank}] Error in batch {i}: {e}")
            continue

    # Save partial result
    tmp_path = args.output_path.replace(".json", f".tmp_rank{rank}.json")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Rank {rank}] Saved {len(results)} results to {tmp_path}")



def main_eval_multigpu(args, eval_data):
    world_size = torch.cuda.device_count()

    print(f"🚀 Launching {world_size}-GPU evaluation")

    # 2️⃣ Spawn workers
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, args, eval_data))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 3️⃣ Gather & merge results
    all_results = []
    for rank in range(world_size):
        tmp_path = args.output_path.replace(".json", f".tmp_rank{rank}.json")
        if os.path.exists(tmp_path):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                all_results.extend(json.load(f))
            os.remove(tmp_path)  # cleanup

    # Sort by original id to ensure deterministic order (optional but recommended)
    try:
        all_results.sort(key=lambda x: int(x["id"]))
    except:
        all_results.sort(key=lambda x: x["id"])

    # 4️⃣ Save final output
    output_path = args.output_path.replace("model_predictions", "DIVERSE_predictions")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged {len(all_results)} predictions → {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save model predictions")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Currently only supports batch_size=1 due to variable-length prompts per sample")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 (requires compatible GPU)")

    parser.add_argument("--N_d", type=int, default=1, help="Number of samplings")

    args = parser.parse_args()

    args.dtype = torch.float32
    print(args.dtype)

    # Load processor
    args.processor = Qwen2_5_VLProcessor.from_pretrained(os.environ.get('LLM_PATH'))
    args.processor.tokenizer.padding_side = 'left'

    args.generation_mode = None
    if "PostTrainSFTVLM" in args.ckpt_path or "Action" in args.ckpt_path:
        args.generation_mode = "action_diversity"
    else:
        args.generation_mode = "llm_diversity"
    print(args.generation_mode)


    def process_mm_Eval_instance(row, image_root_path):
        image_path = row["image"]["file_path"]
        image_path = f"{image_root_path}/{image_path}"

        all_messages = convert_messages_of_PCogAlign(PCogAlign_instance=row)

        mm_language_modeling_instances = []
        for messages in all_messages:
            instance = {
                "id":image_path,
                "images": [image_path],
                "messages": messages[:-1],  # in test mode, only retain the system and user parts
                "expected_response": messages[-1]["content"],
                "raw_aux_info": copy.deepcopy(row),
            }
            mm_language_modeling_instances.append(instance)

        return mm_language_modeling_instances


    if args.eval_data == "IDtest":
        limited_size = [0, 1]
        MM_instances_json_path = "data/image_text_posttrain/YongqiLi/PCogAlignBench/version_v4/HCMAS-golden-test.json"
        with open(MM_instances_json_path, mode="r") as f:
            MM_instances = json.load(f)

        MM_root_path = "data/image_text_posttrain/images/YongqiLi/PCogAlignBench/version_v4"

        random.seed(42)
        random.shuffle(MM_instances)
        MM_instances = MM_instances[int(limited_size[0] * len(MM_instances)):int(limited_size[1] * len(MM_instances))]

        processed_rows = []
        for row in MM_instances:
            processed_rows += process_mm_Eval_instance(row, MM_root_path)

        dataset = processed_rows

    elif args.eval_data == "OODtest":
        limited_size = [0, 1]
        MM_instances_json_path = "data/image_text_posttrain/YongqiLi/PCogAlignBench/version_v4/HCSHR-golden-test.json"
        with open(MM_instances_json_path, mode="r") as f:
            MM_instances = json.load(f)

        MM_root_path = "data/image_text_posttrain/images/YongqiLi/PCogAlignBench/version_v4"

        random.seed(42)
        random.shuffle(MM_instances)
        MM_instances = MM_instances[int(limited_size[0] * len(MM_instances)):int(limited_size[1] * len(MM_instances))]

        processed_rows = []
        for row in MM_instances:
            processed_rows += process_mm_Eval_instance(row, MM_root_path)

        dataset = processed_rows

    else:
        raise NotImplementedError



    eval_data=[]
    for example in dataset:
        eval_data.append(example)


    print(f"Total Eval Data: {len(eval_data)}")

    output_path = main_eval_multigpu(args, eval_data)




