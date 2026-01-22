import json
import os
import time
import openai
from tqdm import tqdm



# TODO: first collect HCMAS related, then HCSHR related
tag="HCSHR"
SOURCE_FILE = f"{tag}-test.json"
target_train_filepath = f"{tag}-golden-train.json"
target_test_filepath = f"{tag}-golden-test.json"

def format_individual_role_set(role_set):
    """Convert role dict to natural-language list: 'a Father at Home; a Fireman at Community; ...'"""
    roles = [f"a {role} at {loc}" for loc, role in role_set.items()]
    return "; ".join(roles)


# 🌟 Updated Prompt Template — no primary/secondary separation
PROMPT_TEMPLATE = """You are an expert assistant generating ideal (golden) responses for a role-grounded multimodal benchmark.

Given the context below, generate a **single**, **natural**, **empathetic**, and **actionable** response to the user's query.

### User Background:
The user is: {all_roles_desc}.

### Visual Scene:
"{ImageDesc}"

### User's Expectations for the Assistant:
"{EvalHelp}"

### User's Query:
"{query}"

### Requirements:
- Respond in fluent, spoken-style English — like a helpful human.
- Be supportive, clear, and practical; address both emotional and procedural needs.
- Use the visual context and role background to ground your advice.
- Do NOT mention roles explicitly unless it feels natural.
- Output ONLY the response text — no headings, no markdown, no prefixes.

Now generate the golden_response:
"""

from api_utils import robust_API_response
model_engine = "qwen3-235b-a22b"
eval_system_prompt = "You are an expert assistant generating ideal (golden) responses for a role-grounded multimodal benchmark."

def generate_golden_response(individual_RoleSet, query, ImageDesc, EvalHelp):
    all_roles_desc = format_individual_role_set(individual_RoleSet)
    prompt = PROMPT_TEMPLATE.format(
        all_roles_desc=all_roles_desc.strip(),
        ImageDesc=ImageDesc.strip(),
        EvalHelp=EvalHelp.strip(),
        query=query.strip()
    )
    # print(prompt)
    # assert 1==0

    try:
        res = robust_API_response(
            model_engine=model_engine,
            system_prompt=eval_system_prompt,
            user_prompt=prompt,
            flag_web_search=False,
            require_json=False,
            temperature=0,
        )
        response = res.strip()
    except Exception as e:
        raise f"[ERROR during evaluation]: {str(e)}"

    return response


import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_instance(idx_instance):
    """Worker function: process one (index, instance) pair"""
    idx, instance = idx_instance
    try:
        individual_RoleSet = instance["individual_RoleSet"]
        query = instance["query"]
        ImageDesc = instance["eval_info"]["ImageDesc"]
        EvalHelp = instance["eval_info"]["EvalHelp"]

        golden = generate_golden_response(
            individual_RoleSet, query, ImageDesc, EvalHelp
        )
        instance["golden_response"] = golden
        return idx, instance  # return index to preserve order
    except Exception as e:
        print(f"\n⚠️  Instance {idx} failed: {e}")
        return idx, None  # mark as failed


def main():
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)[:]

    print(f"📊 Loaded {len(data)} instances from {SOURCE_FILE}")

    # ✅ STEP 1: Concurrent golden_response generation (10 threads)
    print("\n✨ Generating golden_response for all instances (10 workers)...")

    new_data = [None] * len(data)  # placeholder list to preserve order
    success_count = 0

    # Use ThreadPoolExecutor for I/O-bound API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_instance, (i, inst)): i
            for i, inst in enumerate(data)
        }

        # Collect results with tqdm
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Generating"):
            idx = future_to_idx[future]
            try:
                orig_idx, result_instance = future.result(timeout=120)  # 2-min timeout per task
                if result_instance is not None:
                    new_data[orig_idx] = result_instance
                    success_count += 1
                # else: skip (failed)
            except Exception as e:
                print(f"\n❌ Unexpected error for instance {idx}: {e}")
                continue

    # Filter out failed (None) entries and keep only successful ones
    new_data = [inst for inst in new_data if inst is not None]
    print(f"\n✅ Golden response generation done: {success_count}/{len(data)} succeeded.")

    if not new_data:
        raise RuntimeError("No instances succeeded. Check API key / network / prompt.")

    # ✅ STEP 2: Shuffle the FULL successful dataset
    random.seed(42)
    shuffled_data = new_data.copy()
    random.shuffle(shuffled_data)

    # ✅ STEP 3: 80% / 20% split
    n_total = len(shuffled_data)
    n_train = int(0.8 * n_total)
    train_data = shuffled_data[:n_train]
    test_data = shuffled_data[n_train:]

    print(f"\n🔀 Shuffled {n_total} successful instances and split:")
    print(f"   🟩 Train: {len(train_data)} ({len(train_data) / n_total:.1%})")
    print(f"   🟦 Test:  {len(test_data)} ({len(test_data) / n_total:.1%})")

    # ✅ STEP 4: Save
    with open(target_train_filepath, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(target_test_filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)


    print(f"\n🎉 Done! Results saved.")


if __name__ == '__main__':
    main()