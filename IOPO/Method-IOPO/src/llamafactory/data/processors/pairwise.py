# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)

def _encode_inputpairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]
    
    p_nums = len(prompt) # prompt: [{"role": "user", "content": "..."}] * (2T - 1)*2 the first half is
    chosen_messages = prompt[:p_nums//2] + [response[0]]  ##### Modify
    rejected_messages = prompt[p_nums//2:] + [response[1]]
    # rejected_messages = prompt[:p_nums//2] + [response[0]] 
    # chosen_messages = prompt[p_nums//2:] + [response[1]]
    # chosen_messages = prompt[:p_nums//2] + [response[0]]  ##### Modify
    # rejected_messages = prompt[:p_nums//2] + [response[1]]
    chosen_prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    rejected_prompt_ids, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        chosen_prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + chosen_prompt_ids
        rejected_prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + rejected_prompt_ids


    # consider the response is more important
    source_len, target_len = infer_seqlen(max(len(chosen_prompt_ids),len(rejected_prompt_ids)), max(len(chosen_ids),len(rejected_ids)), cutoff_len)
    chosen_prompt_ids = chosen_prompt_ids[:source_len]
    rejected_prompt_ids = rejected_prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids_v1 = chosen_prompt_ids + chosen_ids
    chosen_labels_v1 = [IGNORE_INDEX] * min(source_len, len(chosen_prompt_ids)) + chosen_ids ###
    rejected_input_ids_v1 = chosen_prompt_ids + rejected_ids
    rejected_labels_v1 = [IGNORE_INDEX] * min(source_len, len(chosen_prompt_ids)) + rejected_ids ###
    chosen_input_ids_v2 = rejected_prompt_ids + rejected_ids
    chosen_labels_v2 = [IGNORE_INDEX] * min(source_len, len(rejected_prompt_ids)) + rejected_ids ###
    rejected_input_ids_v2 = rejected_prompt_ids + chosen_ids
    rejected_labels_v2 = [IGNORE_INDEX] * min(source_len, len(rejected_prompt_ids)) + chosen_ids ###
    
    # rejected_input_ids = rejected_prompt_ids + rejected_ids
    # rejected_labels = [IGNORE_INDEX] * min(source_len, len(rejected_prompt_ids)) + rejected_ids ###
    # # rejected_labels = chosen_labels

    return chosen_input_ids_v1, chosen_labels_v1, rejected_input_ids_v1, rejected_labels_v1, chosen_input_ids_v2, chosen_labels_v2, rejected_input_ids_v2, rejected_labels_v2


def preprocess_inputpairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {
        "chosenv1_input_ids": [],
        "chosenv1_attention_mask": [],
        "chosenv1_labels": [],
        "rejectedv1_input_ids": [],
        "rejectedv1_attention_mask": [],
        "rejectedv1_labels": [],
        "chosenv2_input_ids": [],
        "chosenv2_attention_mask": [],
        "chosenv2_labels": [],
        "rejectedv2_input_ids": [],
        "rejectedv2_attention_mask": [],
        "rejectedv2_labels": [],
    }
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["chosen_token_type_ids"] = []
            model_inputs["rejected_token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        # if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
        #     logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
        #     continue
        if len(examples["prompt"][i]) % 2 == 1 or len(examples["response"][i]) > 2:  ### pairwise input is even, response is one
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        chosen_input_ids_v1, chosen_labels_v1, rejected_input_ids_v1, rejected_labels_v1, chosen_input_ids_v2, chosen_labels_v2, rejected_input_ids_v2, rejected_labels_v2 = _encode_inputpairwise_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["chosenv1_input_ids"].append(chosen_input_ids_v1)
        model_inputs["chosenv1_attention_mask"].append([1] * len(chosen_input_ids_v1))
        model_inputs["chosenv1_labels"].append(chosen_labels_v1)
        model_inputs["rejectedv1_input_ids"].append(rejected_input_ids_v1)
        model_inputs["rejectedv1_attention_mask"].append([1] * len(rejected_input_ids_v1))
        model_inputs["rejectedv1_labels"].append(rejected_labels_v1)
        model_inputs["chosenv2_input_ids"].append(chosen_input_ids_v2)
        model_inputs["chosenv2_attention_mask"].append([1] * len(chosen_input_ids_v2))
        model_inputs["chosenv2_labels"].append(chosen_labels_v2)
        model_inputs["rejectedv2_input_ids"].append(rejected_input_ids_v2)
        model_inputs["rejectedv2_attention_mask"].append([1] * len(rejected_input_ids_v2))
        model_inputs["rejectedv2_labels"].append(rejected_labels_v2)
        ### Need further revision
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["chosen_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(chosen_input_ids), processor)
                )
                model_inputs["rejected_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(rejected_input_ids), processor)
                )

    return model_inputs

def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    chosen_messages = prompt + [response[0]]
    rejected_messages = prompt + [response[1]]
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + prompt_ids

    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_labels": [],
    }
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["chosen_token_type_ids"] = []
            model_inputs["rejected_token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["chosen_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(chosen_input_ids), processor)
                )
                model_inputs["rejected_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(rejected_input_ids), processor)
                )

    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels_v1 = list(filter(lambda x: x != IGNORE_INDEX, example["chosenv1_labels"]))
    valid_rejected_labels_v1 = list(filter(lambda x: x != IGNORE_INDEX, example["rejectedv1_labels"]))
    valid_chosen_labels_v2 = list(filter(lambda x: x != IGNORE_INDEX, example["chosenv2_labels"]))
    valid_rejected_labels_v2 = list(filter(lambda x: x != IGNORE_INDEX, example["rejectedv2_labels"]))
    print("chosen_input_ids_v1:\n{}".format(example["chosenv1_input_ids"]))
    print("chosen_inputs_v1:\n{}".format(tokenizer.decode(example["chosenv1_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids_v1:\n{}".format(example["chosenv1_labels"]))
    print("chosen_labels_v1:\n{}".format(tokenizer.decode(valid_chosen_labels_v1, skip_special_tokens=False)))
    print("rejected_input_ids_v1:\n{}".format(example["rejectedv1_input_ids"]))
    print("rejected_inputs_v1:\n{}".format(tokenizer.decode(example["rejectedv1_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids_v1:\n{}".format(example["rejectedv1_labels"]))
    print("rejected_labels_v1:\n{}".format(tokenizer.decode(valid_rejected_labels_v1, skip_special_tokens=False)))
    print("chosen_input_ids_v2:\n{}".format(example["chosenv2_input_ids"]))
    print("chosen_inputs_v2:\n{}".format(tokenizer.decode(example["chosenv2_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids_v2:\n{}".format(example["chosenv2_labels"]))
    print("chosen_labels_v2:\n{}".format(tokenizer.decode(valid_chosen_labels_v2, skip_special_tokens=False)))
    print("rejected_input_ids_v2:\n{}".format(example["rejectedv2_input_ids"]))
    print("rejected_inputs_v2:\n{}".format(tokenizer.decode(example["rejectedv2_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids_v2:\n{}".format(example["rejectedv2_labels"]))
    print("rejected_labels_v2:\n{}".format(tokenizer.decode(valid_rejected_labels_v2, skip_special_tokens=False)))
    
    # print("rejected_input_ids_v1:\n{}".format(example["rejectedv1_input_ids"]))
    # print("rejected_inputs_v1:\n{}".format(tokenizer.decode(example["rejectedv1_input_ids"], skip_special_tokens=False)))
    # print("rejected_label_ids_v1:\n{}".format(example["rejectedv1_labels"]))
    # print("rejected_labels_v1:\n{}".format(tokenizer.decode(valid_rejected_labels_v1, skip_special_tokens=False)))
