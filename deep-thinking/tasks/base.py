import json
import logging
import random
import re
from collections import defaultdict

import numpy as np
import datasets

from anchor import hf_datasets_root
from tasks.loader import TokenizedForMCRightPad
from utils.rng_ctx import RandomContext, EmptyContext

logger = logging.getLogger("task")


class BaseProbInference:
    def __init__(self, prompt_version):
        if prompt_version == "default":
            self.prompt_version = self.default_prompt_version()
        else:
            self.prompt_version = prompt_version

        self.raw_data_result = None
        self.raw_data_sample = None
        self.raw_data_dev = None

        self.can_be_stratified = False
        self.CHOICES = None
        self.num_base_shot = 1

        self._rng_context = EmptyContext()

        self._cached_prefix = None
        self._cached_ex_list = None
        self.shuffled_mapping = None

    def default_prompt_version(self):
        raise NotImplementedError

    def set_seed(self, seed):
        self._rng_context = RandomContext(seed=seed)

    def dataset_signature(self):
        # {
        #      "result":  (dataset_name, subset, split),  # which produce the final result
        #      "sample": (dataset_name, subset, split),  # which we sample ICL few-shot examples
        # }
        raise NotImplementedError

    def dataset_part(self, part):
        return self.dataset_signature()[part]

    def dataset_preprocess(self, raw_data):
        raise NotImplementedError

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        raise NotImplementedError

    def multiple_choice_promptify(self, query, choice):
        raise NotImplementedError

    def shuffle_exemplars(self):
        prefix = self._cached_prefix
        ex_list = self._cached_ex_list

        ex_list_with_idx = list(enumerate(ex_list))
        with self._rng_context:
            random.shuffle(ex_list_with_idx)

        indices, ex_list = zip(*ex_list_with_idx)
        self.shuffled_mapping = indices

        return self.build_exemplar_from_examples(prefix, ex_list)

    def random_selected_exemplars(self, num_shots):
        prefix = ""

        with self._rng_context:
            sampled = random.sample(self.raw_data_sample, num_shots)

        ex_list = [[e["query"], e["choices"][e["answer_idx"]]] for e in sampled]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def stratified_sampling(self, num_k_shots):
        num_shots = self.num_base_shot * num_k_shots

        if not self.can_be_stratified:
            logger.info("Cannot be stratified, fallback to random selection.")
            return self.random_selected_exemplars(num_shots)

        prefix = ""

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = num_shots // len(ans_set)
        residual = num_shots - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        selected_exemplar = [self.raw_data_sample[i] for i in selected_ids]
        ex_list = [[e["query"], e["choices"][e["answer_idx"]]] for e in selected_exemplar]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def build_exemplar_from_examples(self, prefix, ex_list):
        s = prefix
        if len(s):
            s += self.exemplar_seperator()

        for query, choice in ex_list:
            _, line = self.multiple_choice_promptify(query, choice)  # query, <query_with_answer>
            s += line + self.exemplar_seperator()
        return s

    def dataset_file_path(self, part):
        dataset_name, subset, split = self.dataset_part(part)
        dumped_folder = hf_datasets_root.joinpath("dumped")
        if not dumped_folder.exists():
            dumped_folder.mkdir(parents=True)

        file_name = f"{dataset_name}-{subset}-{split}.jsonl"
        file_name = re.sub(r"[^\w_. -]", "_", file_name)
        return dumped_folder.joinpath(file_name)

    def do_load_part(self, part):
        f_path = self.dataset_file_path(part)
        if not f_path.exists():
            self.not_exist_download(part)
            return self.do_load_part(part)  # call once more
        else:
            with f_path.open("r") as f:
                raw_data = [json.loads(line) for line in f]
            data = self.dataset_preprocess(raw_data)
            logger.info(f"Data loaded: {part}.")
            return data

    def do_load(self):
        self.raw_data_result = self.do_load_part("result")
        self.raw_data_sample = self.do_load_part("sample")

    def not_exist_download(self, part):
        f_path = self.dataset_file_path(part)
        logger.info(f"{f_path} not exist, download from huggingface datasets hub...")

        dataset_name, subset, split = self.dataset_part(part)
        data = self.do_download(dataset_name, subset, split=split, cache_dir=str(hf_datasets_root))
        data.to_json(f_path)
        logger.info(f"... success, saved at: {f_path}")

    @staticmethod
    def do_download(dataset_name, subset, split, cache_dir):
        raw_data = datasets.load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        logger.info("Download success.")
        return raw_data

    def mk_result_dataset(self, tokenizer):
        return TokenizedForMCRightPad(self.raw_data_result, tokenizer, self.multiple_choice_promptify)

    def mk_test_dataset(self, tokenzier):
        return self.mk_result_dataset(tokenzier)

    def mk_dev_dataset(self, tokenizer):
        sample_size = len(self.raw_data_result)

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = sample_size // len(ans_set)
        residual = sample_size - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        self.raw_data_dev = [self.raw_data_sample[i] for i in selected_ids]
        return TokenizedForMCRightPad(self.raw_data_dev, tokenizer, self.multiple_choice_promptify)

    def mk_result_dataset_with_demostration(self, tokenizer, exemplar_str):
        def add_demostration(query, choice):
            with_query, with_query_and_choice = self.multiple_choice_promptify(query, choice)
            return f"{exemplar_str}{with_query}", f"{exemplar_str}{with_query_and_choice}"

        return TokenizedForMCRightPad(self.raw_data_result, tokenizer, add_demostration)

    @staticmethod
    def merge_choice_info(choice_info):
        merged = {}
        for k in ["lm_log_p", "norm_lm_log_p"]:
            one_metric_merged = []
            for info in choice_info:
                one_metric_merged.append(info[k])
            merged[k] = one_metric_merged
        return merged

    @staticmethod
    def choice_info_to_predictions(info):
        lm_log_p_idx = int(np.argmax(info["lm_log_p"]))
        norm_lm_log_p_idx = int(np.argmax(info["norm_lm_log_p"]))
        return {"lm_log_p": lm_log_p_idx, "norm_lm_log_p": norm_lm_log_p_idx}

    def post_process(self, generated_info, metric_output=True):
        full_info = []
        num_tested = 0
        num_correct = {"lm_log_p": 0, "norm_lm_log_p": 0}
        for idx, (data, choice_info) in enumerate(zip(self.raw_data_result, generated_info)):
            merged_choice_info = self.merge_choice_info(choice_info)
            merged_predictions_idx = self.choice_info_to_predictions(merged_choice_info)
            combined = {
                "_id": idx,
                "choice_logprob": merged_choice_info,
                "predicted": merged_predictions_idx,
                **data,  # query & answer_idx
            }
            num_tested += 1
            ground_idx = combined["answer_idx"]
            for k in num_correct:
                num_correct[k] += 1 if merged_predictions_idx[k] == ground_idx else 0
            full_info.append(combined)

        if metric_output:
            logger.info("v" * 30)
            for k in num_correct:
                t = num_correct[k] * 100 / num_tested
                logger.info(f"Acc @ {k} : {num_correct[k]} / {num_tested} = {t:.4f}")
            logger.info("^" * 30)

        acc_info = {k: f"{(v * 100 / num_tested):.4f}" for k, v in num_correct.items()}
        return full_info, acc_info
