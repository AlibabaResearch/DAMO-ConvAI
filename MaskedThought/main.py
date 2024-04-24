
from models import modeling_llama, mask_policy_utils
import torch
import transformers
from transformers import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from config.parse_args import *
from data.data_reader import *
import torch.distributed as dist
import trainer.Trainer as Trainer
logging.set_verbosity_info()
import torch.nn as nn
import numpy as np
import random
import math
logging.enable_explicit_format()
import logging as local_logging
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
local_logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",level=logging.INFO)
from transformers import (

    BitsAndBytesConfig,


)
try:
    from peft import get_peft_config, get_peft_model, PeftModel, LoraConfig, TaskType, prepare_model_for_int8_training, prepare_model_for_kbit_training
except:
    pass
from data.tokenizer_utils import prepare_tokenizer
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed, ALL_COMPLETED
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# from vllm import LLM
# import time
import os
local_rank = int(os.environ['LOCAL_RANK'])

def setup_seed(seed=42):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


    os.environ['PYTHONHASHSEED'] = str(seed)



    # set_seed(args.seed)
def main():
    base_args,train_args,model_args,task_args = parse_args()

    setup_seed(train_args.seed)
    model_name = train_args.model_name
    auto_tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)

    print(auto_tokenizer.unk_token)
    print(auto_tokenizer.unk_token_id)
    # < unk >
    # 0
    print(auto_tokenizer.eos_token)
    print(auto_tokenizer.eos_token_id)
    # < / s >
    # 2
    print(auto_tokenizer.bos_token)
    print(auto_tokenizer.bos_token_id)
    # < s >
    # 1
    print(auto_tokenizer.pad_token)
    print(auto_tokenizer.pad_token_id)
    # [PAD]
    # 32000

    special_tokens_dict = dict()
    if auto_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if auto_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if auto_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if auto_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    auto_tokenizer.unk_token = "<unk>"
    auto_tokenizer.bos_token = "<s>"
    auto_tokenizer.eos_token = "</s>"

    auto_tokenizer.add_special_tokens(special_tokens_dict)
    auto_tokenizer.add_tokens(["<mask>"])

    def resize(model):
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        old_num_tokens = input_embeddings.shape[0]
        num_new_tokens = len(auto_tokenizer) - old_num_tokens
        print('num_new_tokens', num_new_tokens)
        if num_new_tokens != 0:
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            model.resize_token_embeddings(len(auto_tokenizer))
            model.get_input_embeddings().weight.data[-num_new_tokens:] = input_embeddings_avg
            model.get_output_embeddings().weight.data[-num_new_tokens:] = output_embeddings_avg

    train_input, eval_input, predict_input = input_builder(model_args._name_or_path, train_args, task_args,
                                                           auto_tokenizer)
    auto_model = task_args.auto_model if hasattr(task_args,'auto_model') else getattr(models,train_args.task)[identifier(model_args)]
    kwargs = {}
    if hasattr(auto_model,'set_cfg'):
        kwargs["customize_cfg"] = task_args
        kwargs["train_cfg"] = train_args
    if train_args.load_in_8bit:
        compute_dtype = (torch.float16 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32))
        max_memory = {i: '30000MB' for i in range(torch.cuda.device_count())}

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_args,
            load_in_8bit=True,

            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                # bnb_4bit_use_double_quant=args.double_quant,
                # bnb_4bit_quant_type=args.quant_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            torch_dtype=(torch.float32 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32)),

        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif train_args.load_in_4bit:
        compute_dtype = (torch.float16 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32))
        max_memory = {i: '30000MB' for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_args,
            load_in_4bit=True,

            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                # bnb_4bit_use_double_quant=args.double_quant,
                # bnb_4bit_quant_type=args.quant_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            torch_dtype=(torch.float32 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32)),

        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_args, low_cpu_mem_usage=False)

    if train_args.resize_at_begin:
        resize(model)

    def find_all_linear_names( model):
        print(model.config)
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        print('lora_module_names', lora_module_names)
        return list(lora_module_names)
    if train_args.use_peft:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head', 'embed_tokens']
        if train_args.lora_no_emb:
            target_modules.remove('lm_head')
            target_modules.remove('embed_tokens')
        modules_to_save = []
        if train_args.modules_to_save:
            modules_to_save = ['lm_head', 'embed_tokens']
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            r=train_args.lora_rank, lora_alpha=32, lora_dropout=0.05, bias="none",
            # modules_to_save=['lm_head']
        )
        if train_args.adapter_path != "":
            model = PeftModel.from_pretrained(model, train_args.adapter_path, is_trainable=True)
        else:
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer = getattr(Trainer, train_args.trainer)(
        model=model,
        args = train_args,
        model_args = model_args,
        train_dataset = train_input,
        eval_dataset = eval_input if not train_args.do_predict else predict_input,
        task_args = task_args,
        auto_tokenizer=auto_tokenizer,
    )

    if train_args.do_train:
        trainer.train()
    if train_args.do_predict:
        trainer.predict()


if __name__ == "__main__":
    logger.info("checking GPU")
    if not torch.cuda.is_available():
        logger.warning("torch.cuda.is_available() Fail")
    else:
        logger.info("torch.cuda.is_available() Succeed")
    main()
