
# import models
import torch
import transformers
from transformers import logging
from config.parse_args import *
from data.data_reader import *
from transformers import Trainer
logging.set_verbosity_info()
logging.enable_explicit_format()
import logging as local_logging
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
local_logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",level=logging.INFO)

from peft import get_peft_config, get_peft_model, PeftModel, LoraConfig, TaskType, prepare_model_for_int8_training, prepare_model_for_kbit_training
from data.tokenizer_utils import prepare_tokenizer
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed, ALL_COMPLETED
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch.distributed as dist
def ddp_setup(rank: int, world_size: int):
   dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)

rank = os.environ["LOCAL_RANK"]
rank = int(rank)
print('rank', rank)
num_gpus = torch.cuda.device_count()
ddp_setup(rank, num_gpus)

n = 1
sampling_params = SamplingParams(temperature=0, max_tokens=512, n=n)
# import time
base_args,train_args,model_args,task_args = parse_args()

model_name = train_args.model_name

auto_tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
print(auto_tokenizer.pad_token)
print(auto_tokenizer.bos_token)
print(auto_tokenizer.eos_token)
print(auto_tokenizer.unk_token)
print(auto_tokenizer.pad_token_id)
print(auto_tokenizer.eos_token_id)
special_tokens_dict = dict()
if auto_tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if auto_tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if auto_tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if auto_tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

auto_tokenizer.add_special_tokens(special_tokens_dict)
auto_tokenizer.add_tokens(["<mask>"])

train_input, eval_input, predict_input = input_builder(model_args._name_or_path, train_args, task_args,
                                                       auto_tokenizer)
kwargs = {}

model = LLM(model_name, tokenizer_mode='slow')
print('finish build LLM')

json_data = open('data/gsm8k_test.json', 'r').readlines()

json_data = [json.loads(e.strip()) for e in json_data]

all_res = []
steps = 0
all_input = []
all_ori_input = []

for i, k in enumerate(json_data):
    input_text = k['source'][0]
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    input_text = PROMPT_DICT['prompt_no_input'].format(instruction=input_text)
    all_input.append(input_text)
    all_ori_input.append(k)


if train_args.use_vllm:
    gen_res = model.generate(all_input, sampling_params)
    assert len(gen_res) == len(all_ori_input)
    for i in range(len(gen_res)):
        gen_res_list = [gen_res[i].outputs[j].text for j in range(n)]
        all_ori_input[i]['kd_data'] = gen_res_list
        all_res.append(json.dumps(all_ori_input[i]))
else:
    gen_res = []
    batch_size = 25
    batches = [all_input[i:i + batch_size] for i in range(0, len(all_input), batch_size)]
    model.eval()
    for i in tqdm(range(len(batches))):
        batch = batches[i]
        auto_tokenizer.padding_side = "left"
        input_ids = auto_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids

        print(input_ids)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=256, num_return_sequences=1, num_beams=1, bos_token_id=auto_tokenizer.bos_token_id, eos_token_id=auto_tokenizer.eos_token_id, pad_token_id=auto_tokenizer.eos_token_id)

        for j, output_id in enumerate(output_ids):
            gen_res.append(auto_tokenizer.decode(output_id, skip_special_tokens=True))
            print(gen_res[-1])

    assert len(gen_res) == len(all_ori_input)
    for i in range(len(gen_res)):
        gen_res_list = [gen_res[i]]
        all_ori_input[i]['kd_data'] = gen_res_list
        all_res.append(json.dumps(all_ori_input[i]))

train_args.model_name = train_args.model_name.replace('/','')
output_file = open(train_args.output_dir + '/' + 'predict_greedy_{}.json'.format(train_args.model_name), 'w')
output_file.write('\n'.join(all_res))

