import torch
from mycvae.trainer import Trainer
from settings import parse_args
from mycvae.utils import get_logger, seed_everything
from dataset import get_datasets
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D, BertTokenizer
from mycvae.memory import Memory

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

dec_out = os.path.join(cache_dir,'dec_out')
# enc_out = os.path.join(cache_dir,'enc_out')
tokz = GPT2Tokenizer.from_pretrained('gpt2')
tokz.save_pretrained(dec_out)

if args.local_rank in [0, -1]:
    logger.info('Loading datasets...'+'.'*10)
datasets = get_datasets(args.data_dir, args.tasks, tokz, num_workers=args.num_workers, ctx_max_len=args.ctx_max_len)
logger.info('Finish loading datasets!')

if args.use_memory:
    memory = Memory()
else:
    memory = None

if __name__ == '__main__':
    trainer = Trainer(args, tokz, datasets, logger, cache_dir, memory=memory)
    trainer.train(args.tasks)

