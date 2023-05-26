from unifymodel.utils import get_logger, seed_everything
from unifymodel.trainer import Trainer
from unifymodel.dataset import get_datasets
import torch
from settings import parse_args
import os
from transformers import AdamW, get_linear_schedule_with_warmup, Conv1D
from transformers import T5Tokenizer, T5Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_args()
logger = get_logger(args.log_file)

if args.local_rank in [0, -1]:
    logger.info('Pytorch Version: {}'.format(torch.__version__))
    for k, v in vars(args).items():
        logger.info("{}= {}".format(k, v))

# seed everything
seed_everything(args.seed)

cache_dir = os.path.join(args.output_dir, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)


# * Add special tokens. 
special_tokens_dict = {'additional_special_tokens': ['<ANS>','<QUES>']}
out = os.path.join(cache_dir,'out')
tokz = T5Tokenizer.from_pretrained('t5-base',cache_dir=out)

num_spe_token = tokz.add_special_tokens(special_tokens_dict)

if args.use_task_pmt:  
    pre_token_list = []
    for task in args.tasks:
        pre_token_list += [str(task)+':']
else:
    pre_token_list = ['TASK:' for i in range(len(args.tasks))]

num_add_tokens = tokz.add_tokens(pre_token_list)
print('We have added', num_spe_token+num_add_tokens, 'tokens to T5', flush=True)
tokz.save_pretrained(out)

if args.local_rank in [0, -1]:
    logger.info('Loading datasets...'+'.'*10)
datasets = get_datasets(args.data_dir, args.tasks, tokz, max_input_len=args.max_input_len)
logger.info('Finish loading datasets!')

if args.use_memory:
    memory = Memory()
else:
    memory = None

trainer = Trainer(args, tokz, datasets, logger, cache_dir, memory=memory)
trainer.train(args.tasks)

