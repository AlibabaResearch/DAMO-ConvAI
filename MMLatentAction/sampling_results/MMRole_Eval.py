import json
import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from api_utils import robust_API_response

eval_system_prompt = """
You are an objective and precise evaluator, specializing in rigorously assessing the role-playing and multimodal understanding abilities of various models.
"""

eval_prompt_template = """
## [Question Start]
{question}
## [Question End]

## [Model A's Response Start]
{evaluated_answer}
## [Model A's Response End]

## [Model B's Response Start]
{groundtruth_answer}
## [Model B's Response End]

## [Instruction]
The task instruction of the two models is to directly role-play as {role_name} and talk with a curious human about the given image using the distinctive tone, manner and vocabulary of {role_name}. 

Here is the detailed character information about {role_name}:
{role_info}

Please evaluate the following aspects of each model's response:
1. Instruction Adherence: Do the responses accurately adhere to the task instruction, directly role-playing as {role_name} and only including words that {role_name} should say, without any additional explanatory prefixes or suffixes?
2. Fluency: Are the responses grammatically correct and smoothly articulated?
3. Coherency: Do the responses maintain a coherent thread of dialogue without contradicting earlier parts of the conversation or previously established facts?
4. Image-Text Relevance: Are the responses closely related to the visual content of the image?
5. Response Accuracy: Do the responses accurately answer the curious human's words or appropriately initiate a conversation based on the image?
6. Personality Consistency: Do the responses accurately and sufficiently reflect the personality of {role_name}?
7. Knowledge Consistency: Are the responses consistent with the factual knowledge that {role_name} should possess, including experiences, abilities, and relationships?
8. Tone Consistency: Do the responses maintain a consistent tone that aligns with {role_name}'s typical manner of speaking and catchphrases, rather than resembling the style of AI assistants?

For each aspect, provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from 1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.

The output should be in the following format:
1. Instruction Adherence:  {{Qualitative Evaluation}}, [Scores]: ({{the score of Model A}}, {{the score of Model B}})
2. Fluency: {{Qualitative Evaluation}}, [Scores]: ({{the score of Model A}}, {{the score of Model B}})
etc.

Please ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment.
Format requirement: Please ensure that your evaluations only include 8 score pairs, which means that there can only be eight pairs of [Scores]: () in your output text.
"""

import re


def extract_score_pairs(evaluation_text):
    pattern = r'\[Scores\]:\s*\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, evaluation_text)
    scores = [(int(a), int(b)) for a, b in matches]
    return scores


num_dimensions = 8


def get_eval_score(eval_disc, evaluated_answer, model_engine, num_eval_iters=1):
    # 构建评估 prompt
    eval_prompt = eval_prompt_template.format(
        question=eval_disc["question"],
        evaluated_answer=evaluated_answer,
        groundtruth_answer=eval_disc["groundtruth_answer"],
        role_name=eval_disc["role_name"],
        role_info=eval_disc["role_info"]
    )

    all_evaluation_criteria = [[[], []] for _ in range(num_dimensions)]

    for iter in range(num_eval_iters):

        try:
            res = robust_API_response(
                model_engine=model_engine,
                system_prompt=eval_system_prompt,
                user_prompt=eval_prompt,
                flag_web_search=False,
                require_json=False,
                temperature=0,
            )
            evaluation_text = res.strip()
        except Exception as e:
            evaluation_text = f"[ERROR during evaluation]: {str(e)}"

        iter_evaluation_criteria = None
        try:
            iter_evaluation_criteria = extract_score_pairs(evaluation_text)
        except Exception as e:
            print(e)

        try:
            assert len(iter_evaluation_criteria) == num_dimensions
            for dim_idx in range(num_dimensions):
                assert len(iter_evaluation_criteria[dim_idx]) == 2
        except Exception as e:
            print(e)
            continue

        for dim_idx in range(num_dimensions):
            all_evaluation_criteria[dim_idx][0].append(iter_evaluation_criteria[dim_idx][0])
            all_evaluation_criteria[dim_idx][1].append(iter_evaluation_criteria[dim_idx][1])
    try:
        # print(all_evaluation_criteria)
        valid_iters_num = len(all_evaluation_criteria[dim_idx][0])
        if valid_iters_num < 1:
            return None, None, None

        evaluation_criteria = [[0, 0] for _ in range(num_dimensions)]
        for dim_idx in range(num_dimensions):
            evaluation_criteria[dim_idx][0] = sum(all_evaluation_criteria[dim_idx][0]) / valid_iters_num
            evaluation_criteria[dim_idx][1] = sum(all_evaluation_criteria[dim_idx][1]) / valid_iters_num
    except Exception as e:
        print(e)
        return None, None, None


    return evaluation_criteria, evaluation_text,all_evaluation_criteria




def calculate_mean_score_with_std(output_file):
    # =============== 新统计逻辑：每样本选最佳响应的平均分，再整体统计 ===============
    print("\n📊 Computing per-sample best average score (over 8 dimensions)...")
    print(f"Eval Engine: {model_engine}")

    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results or "eval_scores" not in results[0]:
        print("⚠️ No evaluation scores found. Skipping score computation.")
    else:
        num_generate_iters=len(results[0]["all_eval_scores"].keys())
        print(f"There are {num_generate_iters} generations.")
        # num_generate_iters=3
        all_eval_results_by_iters=[[] for _ in range(num_generate_iters)]


        dim_dict_scores={}
        for dim_idx in range(num_dimensions):
            dim_dict_scores[f"dim_{dim_idx}"]=[[] for _ in range(num_generate_iters)]

        for item in results:
            all_eval_scores_dict = item["all_eval_scores"]

            for generate_key_index,(generate_key, score_pairs_by_iters) in enumerate(all_eval_scores_dict.items()):
                valida_iters = 1 # the api call times for each response
                valida_iter_index = 0 # the api call times for each response

                try:
                    model_scores = [score_pairs_by_iters[dim_idx][0][valida_iter_index] for dim_idx in range(num_dimensions)]
                    gt_scores = [score_pairs_by_iters[dim_idx][1][valida_iter_index] for dim_idx in range(num_dimensions)]

                    if use_ratio:
                        model_a_scores = [model_scores[dim_idx] / (gt_scores[dim_idx] + 1e-6) for dim_idx in range(num_dimensions)]
                    else:
                        model_a_scores = [model_scores[dim_idx] for dim_idx in range(num_dimensions)]

                    item_mean_score = sum(model_a_scores) / len(model_a_scores)
                    all_eval_results_by_iters[generate_key_index].append(item_mean_score)

                    for dim_idx in range(num_dimensions):
                        dim_dict_scores[f"dim_{dim_idx}"][generate_key_index].append(model_a_scores[dim_idx])
                except:
                    continue


        print(f"\n✅ Saved per-sample best average score summary to {output_file}")

        # ——————————————————————————————————————————————————————————————
        # 🔹 Overall Evaluation Results Across Iterations
        # ——————————————————————————————————————————————————————————————
        print("\n" + "=" * 80)
        print("📊 Overall Evaluation Results Across Iterations")
        print("=" * 80)

        print("Raw results by iteration (each list = one eval iteration):")
        for i, res in enumerate(all_eval_results_by_iters):
            print(f"  Iter {i + 1:2d}: {res}")
            print(f"There are {len(res)} generations")

        eval_means = [mean(iter_scores) for iter_scores in all_eval_results_by_iters]
        overall_mean = mean(eval_means)
        overall_std = stdev(eval_means) if len(eval_means) > 1 else 0.0

        print(f"\n✅ Mean per iteration: {eval_means}")
        print(f"📈 Overall Mean (across iterations): {overall_mean:.4f}")
        print(f"📉 Std Dev (across iterations):    {overall_std:.4f}")
        print("=" * 80)

        # ——————————————————————————————————————————————————————————————
        # 🔹 Per-Dimension Evaluation Summary
        # ——————————————————————————————————————————————————————————————
        print("\n" + "=" * 80)
        print("🧩 Per-Dimension Evaluation Summary")
        print("=" * 80)

        for dim_idx in range(num_dimensions):
            dim_key = f"dim_{dim_idx}"
            dim_name = dimension_names[dim_idx] if dim_idx < len(dimension_names) else f"Dim {dim_idx}"

            dim_all_iter_scores = dim_dict_scores[dim_key]

            # Compute per-iteration means for this dimension
            iter_means = [mean(scores) for scores in dim_all_iter_scores]
            dim_mean = mean(iter_means)
            dim_std = stdev(iter_means) if len(iter_means) > 1 else 0.0

            print(f"\n🔹 Dimension {dim_idx} — '{dim_name}'")
            print(f"   • Scores per iteration: {iter_means}")
            print(f"   • Mean across iterations: {dim_mean:.4f}")
            print(f"   • Std Dev across iterations:  {dim_std:.4f}")
            print("   " + "─" * 50)

        print("=" * 80)

        # 保存新统计
        new_summary = {
            "all_eval_results_by_iters": all_eval_results_by_iters,
            "eval_means": eval_means,
            "overall_mean": overall_mean,
            "overall_std": overall_std,

            "total_samples": len(results),
            "eval_engine": model_engine,
        }

        new_summary_path = output_file.replace(".json", "_mean_with_std.json")
        with open(new_summary_path, 'w', encoding='utf-8') as f:
            json.dump(new_summary, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved per-sample best average score summary to {new_summary_path}")


import math
from statistics import mean, stdev

def std(lst):
    n = len(lst)
    if n < 2:
        return 0.0
    mean = sum(lst) / n
    variance = sum((x - mean) ** 2 for x in lst) / n  # population std
    # For sample std: use / (n - 1) instead
    return math.sqrt(variance)



# model_engine = "claude3.7-sonnet"
model_engine = "qwen3-235b-a22b"
# model_engine = "gpt-4o-mini"


load_old_results = False
# load_old_results=True


use_ratio = True
# use_ratio=False

TIMEOUT_SECONDS = 20

dimension_names = [
    "Instruction Adherence",
    "Fluency",
    "Coherency",
    "Image-Text Relevance",
    "Response Accuracy",
    "Personality Consistency",
    "Knowledge Consistency",
    "Tone Consistency"
]

if __name__ == '__main__':
    def extract_last_user_question(prompt_conversation):
        for turn in reversed(prompt_conversation):
            if turn["role"] == "user":
                content = turn["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            return item["text"]
                elif isinstance(content, str):
                    return content
        return "No user question found."


    tags = [
        # "DIVERSE_predictions-ckpt_pretrain_01051052-MMRole-DAPO-VLMActionRL-IDtest",
        "DIVERSE_predictions-ckpt_pretrain_01051052-MMRole-DAPO-VLMActionRL-OODtest",
    ]

    eval_dir = "diversity_evaluation_results"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    MAX_WORKERS = 10

    for tag in tags:
        input_file = f"{tag}.json"
        output_file = f"{eval_dir}/ENGINE[{model_engine}]-{input_file}"

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)[:]  # 限制样本数用于测试

        if not load_old_results:
            # Step 1: 构建所有评估任务
            all_tasks = []  # each: (sample_id, key, evaluated_answer, eval_disc)
            sample_lookup = {}  # sample_id -> original sample info for reconstruction

            for sample in data:
                sample_id = sample["id"]
                # question = extract_last_user_question(sample["prompt_conversation"])
                question = str(sample["prompt_conversation"])

                groundtruth_answer = sample["ground_truth"]
                role_name = sample["character_role"]
                role_info = str(sample["character_profile"])

                eval_disc = {
                    "question": question,
                    "groundtruth_answer": groundtruth_answer,
                    "role_name": role_name,
                    "role_info": role_info,
                }

                sample_lookup[sample_id] = {
                    "sample": sample,
                    "eval_disc": eval_disc,
                    "model_keys": list(sample["model_diverse_output"].keys())
                }

                for key, evaluated_answer in sample["model_diverse_output"].items():
                    all_tasks.append((sample_id, key, evaluated_answer, eval_disc))

            print(f"Total evaluation tasks for {tag}: {len(all_tasks)}")

            # Step 2: 全局并发执行所有任务
            results_by_sample = {sid: {"all_evaluation_criteria": {}, "scores": {}, "texts": {}} for sid in sample_lookup}

            with tqdm(total=len(all_tasks), desc=f"Evaluating {tag}") as pbar:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit all tasks
                    future_to_task = {
                        executor.submit(get_eval_score, task[3], task[2], model_engine): (task[0], task[1])
                        for task in all_tasks
                    }

                    # Collect results with timeout
                    for future in as_completed(future_to_task):
                        sample_id, key = future_to_task[future]
                        evaluation_criteria, evaluation_text,all_evaluation_criteria = future.result()

                        if evaluation_criteria is None:
                            results_by_sample[sample_id]["all_evaluation_criteria"][key] = []
                            results_by_sample[sample_id]["scores"][key] = []
                            results_by_sample[sample_id]["texts"][key] = "[ERROR]"
                        else:
                            results_by_sample[sample_id]["all_evaluation_criteria"][key] = all_evaluation_criteria
                            # 设置超时时间（例如 30 秒）
                            results_by_sample[sample_id]["scores"][key] = evaluation_criteria
                            results_by_sample[sample_id]["texts"][key] = evaluation_text

                        pbar.update(1)
            # Step 3: Reconstruct final results list in original sample order
            final_results = []
            for sample in data:
                sid = sample["id"]
                res = {
                    "id": sid,
                    "question": extract_last_user_question(sample["prompt_conversation"]),
                    "model_diverse_output": sample["model_diverse_output"],
                    "ground_truth": sample["ground_truth"],
                    "all_eval_texts": results_by_sample[sid]["texts"],
                    "eval_scores": results_by_sample[sid]["scores"],
                    "all_eval_scores": results_by_sample[sid]["all_evaluation_criteria"],

                }
                final_results.append(res)

            # Save results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            print(f"Evaluation completed. Results saved to {output_file}")

        calculate_mean_score_with_std(output_file=output_file)
