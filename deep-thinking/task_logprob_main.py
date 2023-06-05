import gc
import json
import logging
import os
import textwrap


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchor import logger_root
from common import setup_env, mk_parser, AdvantageLogger
from models import build_model_signature, build_tokenizer, build_model
from models.meta_optimizer import AttnOptimWrapper
from tasks import load_task
from utils.logger import setup_logger, tabular_pretty_print
from utils.tools import ensure_folder

logger = logging.getLogger("task")


def the_shape(pack):
    if isinstance(pack, (list, tuple)):
        return f"{len(pack)} * {the_shape(pack[0])}"
    if isinstance(pack, torch.Tensor):
        return pack.size()


@torch.no_grad()
def do_infer_probs(exemplar_attn_kv, exemplar_attn_mask, batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        bs = len(batch_input_ids)

        merged_attn_mask = torch.cat((exemplar_attn_mask.expand(bs, -1), batch_attention_mask), dim=1)
        if args.model_type == "bloom":
            # [B*#Heads, Length, Hidden]
            def _expand(t, target_size):
                _bs, _head, _len, _hidden = 1, *t.size()
                return t.reshape(_bs, _head, _len, _hidden).expand(target_size * _bs, -1, -1, -1).reshape(target_size * _bs * _head, _len, _hidden)

            expand_exemplar_attn_kv = [[_expand(layer_k, bs), _expand(layer_v, bs)] for layer_k, layer_v in exemplar_attn_kv]
        else:
            # [B, #Heads, Length, Hidden]
            expand_exemplar_attn_kv = [[layer_k.expand((bs, -1, -1, -1)), layer_v.expand((bs, -1, -1, -1))] for layer_k, layer_v in exemplar_attn_kv]

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=merged_attn_mask,  # [B, L + L']
            past_key_values=expand_exemplar_attn_kv,  # num_layers * 2 * [B, num_heads, L, H]
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(batch_input_ids, batch_choice_start, batch_choice_end, batched_output):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1 : choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {"lm_log_p": lm_log_p, "norm_lm_log_p": norm_lm_log_p}
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs


if __name__ == "__main__":
    DEBUG = False

    parser = mk_parser()
    if DEBUG:
        fake_cmd = (
            "--prompt default "
            # "--dataset sst2 "
            # "--dataset sst5 "
            # "--dataset mr "
            # "--dataset agnews "
            "--dataset trec "
            # "--dataset qasc "
            # "--dataset obqa"
            # "--dataset hellaswag "
            # "--dataset copa "
            # "--dataset winogrande "
            # "--exemplar_method written "
            "--exemplar_method stratified --num_k_shots 1 "
            "--model_type opt --model_size 125m "
            # "--model_type gpt2 --model_size sm "
            # "--model_type e-gpt --model_size neo-125M "
            # "--model_type bloom --model_size 560m "
            "--kv_iter 15 "
            "--step_size 0.01 --momentum 0.9 "
            "--batch_size 16 "
            "--seed 0 "
            "--gpus 0 "
            "--in_8bit true"
        )
        args = parser.parse_args(fake_cmd.strip().split())
    else:
        args = parser.parse_args()

    if DEBUG:
        logger_root = logger_root.joinpath("DEBUG")

    logger_root = logger_root.joinpath("main")
    dataset_name = args.dataset
    logger_folder = logger_root.joinpath(dataset_name)

    task_name = f"seed{args.seed}_main{args.kv_iter}"
    task_name += f"_{args.prompt_version}"
    task_name += f"_{args.model_type}_{args.model_size}"
    task_name += f"_{args.exemplar_method}{'' if args.exemplar_method == 'written' else args.num_k_shots}"
    task_name += f"_eps{args.step_size}_beta{args.momentum}"

    setup_env(gpu_s=args.gpus, seed=args.seed)
    ensure_folder(logger_folder, parents=True)
    setup_logger(
        logger_folder,
        log_file_name=f"{task_name}.log",
        console_output=not args.no_console,
    )

    logger.info(f"Task Prepared: {task_name}")
    logger.info(f"\tDataset: {dataset_name}")
    logger.info(f"\tLogger save at {logger_folder}")

    # 1. load model, tokenizer
    model_signature = build_model_signature(args.model_type, args.model_size)
    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side="right")
    model = build_model(args.model_type, args.model_size, args.in_8bit)
    torch.autograd.set_grad_enabled(False)
    logger.info(f"Model loaded: {model_signature}")

    # 2. load dataset (with demonstrations)
    TaskHandler = load_task(dataset_name)
    task_agent = TaskHandler(args.prompt_version)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    dataset = task_agent.mk_result_dataset(tokenizer)

    if args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    elif args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots)
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    text_width = os.get_terminal_size().columns - 30
    exemplar_showcase = [["Line", "Text"]]
    for line_idx, line in enumerate(exemplar_str.split("\n")):
        if len(line) > text_width:
            splitted_lines = textwrap.wrap(line, text_width)
            exemplar_showcase.append([str(line_idx + 1), splitted_lines[0]])
            for remained in splitted_lines[1:]:
                exemplar_showcase.append(["", remained])
        else:
            exemplar_showcase.append([str(line_idx + 1), line])

    exemplar_showcase[-1][-1] += "<query starts from here>"
    for line in tabular_pretty_print(exemplar_showcase):
        logger.info(line)

    exemplar_input_ids, exemplar_attn_mask = [e.cuda() for e in dataset.tokenize_demonstration(exemplar_str)]
    meta_optim = AttnOptimWrapper(model, args.model_type, step_size=args.step_size, momentum=args.momentum)

    logger.info(f"Selected batch_size: {args.batch_size}")

    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=2)

    logger.info("Running ...")
    meta_optim.init()

    trace_logger = AdvantageLogger()
    for idx in range(args.kv_iter):
        exemplar_kv = meta_optim.step(exemplar_input_ids)

        generated_info = []  # question * [choice0_prob, choice1_prob]
        for batch_input in tqdm(loader, desc=f"idx={idx}"):
            batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
            batch_output = do_infer_probs(
                exemplar_kv,
                exemplar_attn_mask.unsqueeze(0),
                batch_input,
            )  # [batch_of_choice0, batch_of_choice1, ...]
            zipped_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
            generated_info.extend(zipped_logprobs)

        full_info, metric = task_agent.post_process(generated_info, metric_output=False)
        metric_s = json.dumps(metric, indent=None)
        logger.info(f"Iter={idx+1: <3} | {metric_s}")
        trace_logger.submit(idx + 1, metric["lm_log_p"])
        # gc.collect()

    for line in trace_logger.pretty_print():
        logger.info(line)
