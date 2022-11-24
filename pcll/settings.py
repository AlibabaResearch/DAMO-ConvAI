import os
import argparse
import torch
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_during_train',type=bool,default=False)
    parser.add_argument('--test_all',type=bool,default=True)
    parser.add_argument('--z_dim',type=int, default=768, help='Dimension of the latent variable z.')
    parser.add_argument('--save_z',type=bool, default=False, help='Directly save the latent variable z into the memory.')
    parser.add_argument('--share_params', type=bool, default=False)
    parser.add_argument('--general_prompt', type=bool, default=False)
    parser.add_argument('--add_kd', type=bool, default=True)
    parser.add_argument('--data_type',type=str,default='intent')
    parser.add_argument('--KD_term', type=float, default=0.5, help="Control how many teacher model signal is passed to student.")
    parser.add_argument('--KD_temperature', type=float, default=1.0, help="Temperature used to calculate KD loss.")    
    parser.add_argument('--exact_replay', default=False, type=bool, help='Whether to do exact replay by storing some real samples of old tasks into the memory.')
    parser.add_argument('--memory_size', default=10, type=int, help='Number of posterior information of old tasks stored in the memory.')
    parser.add_argument("--num_cycle", default=1, type=int, help="Number of cycles for annealing")
    parser.add_argument('--cycle_ratio',default=0.9, type=float, help="Ratio for cycle annearling.")
    parser.add_argument("--classIL", default=False, type=bool, help="Whether use class incremental learning during testing.")
    parser.add_argument('--use_memory', default=False, type=bool, help="Whether store the learned latent variables z and other info into the memory.")
    parser.add_argument('--memory_path', default=None, type=str) # Whether store the learned latent variables z and other info into the memory.
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--tasks',nargs='+', default=['banking'])
    parser.add_argument("--data_dir", default="./PLL_DATA/", type=str, help="The path to train/dev/test data files.")   # Default parameters are set based on single GPU training
    parser.add_argument("--output_dir", default="./output/dstc", type=str,help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--tb_log_dir", default="./tb_logs/dstc", type=str,help="The tensorboard output directory.")
    parser.add_argument("--res_dir",default="./res", type=str,help="The path to save scores of experiments.")
    parser.add_argument('--nsamples', type=int, default=64) # For generation
    parser.add_argument("--gene_batch_size", default=64, type=int) # For generation.
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--do_train',type=bool, default=True)
    parser.add_argument('--gen_replay',type=bool, default=False, help='Whether use generative replay to avoid forgetting.')
    parser.add_argument('--model_path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument('--generate_dur_train', type=bool, help='Generate reconstructed input utterances during training with CVAE.')
    parser.add_argument('--temperature',type=float, default=0.95)
    parser.add_argument('--pseudo_data_ratio', type=float, default=0.05, help="How many pseudo data to generate for each learned task")
    parser.add_argument("--only_decoder", type=bool, help="Not use latent code z, only use prompt to generate pseudo data.")
    parser.add_argument("--only_vae", type=bool, help="Not use lm ce loss to update the model.")
    parser.add_argument('--latent_size',type=int, default=32, help='dimension of the latent variable z in CVAE.')
    parser.add_argument('--alpha_z',type=float, default=0.1, help='Multiply alpha when adding the latent z embedding onto the original embeddings.')

    parser.add_argument('--add_input', type=bool, default=False)
    parser.add_argument('--add_attn', type=bool, default=False)
    parser.add_argument('--add_softmax', type=bool, default=False)
    parser.add_argument('--attn_proj_vary', type=bool, default=False)

    parser.add_argument('--learn_prior', default=True, type=bool)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    # parser.add_argument('--iterations', type=int, default=200)  # wp 850001  wi 300001 ax 300001 yp 800001
    # parser.add_argument('--dataset', type=str, default='wi', choices=['ax', 'yp', 'wp', 'wi'], help="Dataset to use for training")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument('--seq-lens', nargs='+', type=int, default=[1024],
    #                     help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

   # * KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--beta_warmup', type=int, default=50000)
    # cyc_vae parameters
    parser.add_argument('--cycle', type=int, default=1000)

    parser.add_argument("--filename",type=str,help="Data original file to be preprocessed.")
    parser.add_argument("--init_model_name_or_path", default="./dir_model/gpt2", type=str, help="Path to init pre-trained model")

    parser.add_argument("--num_workers", default=1, type=int, help="workers used to process data")
    parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

    # Other parameters
    parser.add_argument("--ctx_max_len", default=128, type=int, help="Maximum input length for the sequence")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs for each task.")
    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup step. Will overwrite warmup_proportion.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--nouse_scheduler", action='store_true', help="dont use get_linear_schedule_with_warmup, use unchanged lr")

    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_epochs', type=float, default=2, help="Save checkpoint every X epochs.")
    parser.add_argument('--eval_steps', type=int, default=20, help="Eval current model every X steps.")
    parser.add_argument('--eval_times_per_task', type=float, default=10, help="How many times to eval in each task, will overwrite eval_steps.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for everything")
    parser.add_argument("--log_eval_res", default=False, help="Whether to log out results in evaluation process")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through pytorch implementation) instead of 32-bit")
    parser.add_argument("--debug", action="store_true", help="Use debug mode")
    args = parser.parse_args()

    # making dirs
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    # if os.path.exists(args.tb_log_dir) and os.path.isdir(args.tb_log_dir):
        # os.remove(args.tb_log_dir)
        # shutil.rmtree(args.tb_log_dir)
        # os.remove(args.log_file)
    if args.local_rank in [0, -1]:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.tb_log_dir, exist_ok=True)
    else:
        while (not os.path.isdir(args.output_dir)) or (not os.path.isdir(args.tb_log_dir)):
            pass

    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True
    
    # setup distributed training
    distributed = (args.local_rank != -1)
    if distributed:
        print(args.local_rank)
        # torch.cuda.set_device(0)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            args.device = torch.device("cpu")

    args.num_train_epochs = {task: args.num_train_epochs for task in args.tasks}

    return args
