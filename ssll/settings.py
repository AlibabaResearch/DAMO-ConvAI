import os
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # * New arguments for semi-supervised continual learning.
    parser.add_argument('--newmm_size', default=0.2, type=float, help='Different memory size for storing new task unlabeled data.')
    parser.add_argument('--random_initialization',default=False, type=bool)
    parser.add_argument('--evaluate_zero_shot',default=False, type=bool)
    parser.add_argument('--forward_augment',default=False, type=bool, help='Whether apply forward augmentation from old constructed memory.')
    parser.add_argument('--backward_augment', type=bool, default=False, help='Whether use the current task unlabeled data to augment the old tasks.')
    parser.add_argument('--select_adapter',default=False, type=bool, help='Whether select the previous adapter for the new task initialization.')
    parser.add_argument('--similarity_tau',default=0.6, type=float, help='Similarity threshold of cosine for KNN.')
    parser.add_argument('--back_kneighbors',default=5, type=int, help='Retrieve K nearest neighbors from the current unlabeled memory.')
    parser.add_argument('--diff_question', type=bool, default=False, help='If true, the dataset has different questions for each sample.')
    parser.add_argument('--debug_use_unlabel', default=False, type=bool, help='Always use unlabeled data for debugging.')
    parser.add_argument('--add_confidence_selection', type=bool, default=False, help='Whether add confidence selection for pseudo label.')
    parser.add_argument('--kneighbors', default=5, type=int, help='Retrieve K nearest neighbors from the memory.')
    parser.add_argument('--construct_memory',default=False, help='Construct memory for generated pseudo data from old tasks.')
    parser.add_argument('--aug_neighbors_from_memory', default=False, help='Whether augment retrieved neighbors inputs from memory.')
    parser.add_argument('--lm_lambda',type=float, default=0.5, help='The coefficient of language modeling loss.')
    parser.add_argument('--accum_grad_iter', type=bool, default=1, help='Accumulation steps for gradients.')
    parser.add_argument('--add_label_lm_loss', type=bool, default=False, help='Whether compute lm loss on the labeled inputs.')
    parser.add_argument('--add_unlabel_lm_loss', type=bool, default=False, help='Whether compute lm loss on the unlabeled inputs.')
    parser.add_argument('--pretrain_epoch',type=int, default=20, help='Pretrain first for e.g.20 epochs before finetuning.')
    parser.add_argument('--pretrain_first', type=bool, default=False, help='Whether to perform pretraining before each task finetuning.')
    parser.add_argument('--test_all', type=bool, default=False, help='Whether evaluate all test data after training the last epoch.')
    parser.add_argument('--add_pretrain_with_ft', type=bool, default=False, help='Whether add pretraining MLM loss during finetuning.')
    parser.add_argument('--rdrop', default=False, type=bool, help='Whether use R-drop as the consistency regularization, where dropout is used as model augmentation.')
    parser.add_argument('--stu_feedback', default=False,type=bool, help='Whether consisder feedback from student to teacher on labeled datasets.')
    parser.add_argument('--feedback_threshold', default=1e-1, type=float, help='Threshold of student feedback to determine whether to use teacher to teach the student.')
    parser.add_argument('--dropout_rate',default=0.3, type=float, help='Modify dropout rate for model augmentation with dropout.')
    parser.add_argument('--freeze_plm', default=False, type=bool, help='Whether to freeze the pretrained LM and only finetuning the added adapters.')
    parser.add_argument('--max_input_len',default=512, type=int, help='Max number of input tokens.')
    parser.add_argument('--max_ans_len', default=100, type=int, help='Max number of answer tokens.')
    parser.add_argument('--use_task_pmt',default=False, type=bool, help='Whether use task specific token as the prefix.')
    parser.add_argument('--use_ema', default=False, type=bool, help='Use EMA for fixmatch.')
    parser.add_argument('--fixmatch', default=False, type=bool, help='Use fixmatch to solve semi-supervised learning.')
    parser.add_argument('--num_label', default=500, type=int, help='Number of labeled data we use.')
    parser.add_argument('--unlabel_ratio', default=500, type=int, help='Ratio of unlabeled data over labeled data we use.')
    parser.add_argument('--logit_distance_cost', default=-1, type=float, help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--test_overfit', default=False, type=bool, help='Test whether the model can overfit on labeled train dataset.')
    parser.add_argument('--num_aug', type=int, default=3, help='Number of augmented sentences per sample with EDA.')
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency_rampup', type=int, default=30, metavar='EPOCHS', help='length of the consistency loss ramp-up')
    parser.add_argument('--consistency', type=float, default=100.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--meantc', type=bool, default=False, help='Whether to use mean teacher to deal with SSL.')
    parser.add_argument('--only_update_prompts', type=bool, default=False, help='Only update the prompt tokens params.')
    parser.add_argument('--pl_sample_num', type=int, default=1, help='Number of sampling for pseudo labeling.')
    parser.add_argument('--pl_sampling', type=bool, default=False, help='Perform pseudo labeling with sampling for unlabeled data. That means, sample several times for each example, and choose the one with the lowest ppl.')
    parser.add_argument('--unlabel_amount', type=str, default='1', help='How much unlabeled data used per labeled batch.')
    parser.add_argument('--ungamma', type=float, default=0.2, help='Weight added to the unlabeled loss.')
    parser.add_argument('--KD_term', type=float, default=1, help="Control how many teacher model signal is passed to student.")
    parser.add_argument('--KD_temperature', type=float, default=2.0, help="Temperature used to calculate KD loss.")        
    parser.add_argument('--consist_regularize', type=bool, default=False, help='Whether to use consistency regularization for SSL.')
    parser.add_argument('--stu_epochs', type=int, default=1, help='Number of epochs for training student model.')
    parser.add_argument('--noisy_stu', type=bool, default=False, help='Whether to use noisy student self-training framework for SSL, generate pseudo labels for all unlabeled data once.')
    parser.add_argument('--online_noisy_stu', type=bool, default=False, help='Noisy student self-training framework with PL and CR. Perform pseudo labeling online.')
    parser.add_argument('--naive_pseudo_labeling', type=bool, default=False, help='Whether to use pseudo-labeling for semi-supervised learning.')
    parser.add_argument('--only_pseudo_selection', type=bool, default=False, help='Whether to select pseudo labels with high confidence.')
    parser.add_argument('--pretrain_ul_input', type=bool, default=False, help='Only pretrain the inputs of unlabeled data.')
    parser.add_argument('--use_unlabel', type=bool, default=False, help='Whether to use unlabeled data during training.')
    parser.add_argument('--use_infix', type=bool, default=False, help='Whether to use infix prompts.')
    parser.add_argument('--use_prefix', type=bool, default=False, help='Whether to use prefix prompts.')
    parser.add_argument('--preseqlen', type=int, default=5, help='Number of prepended prefix tokens.')
    parser.add_argument('--train_dir',type=str, help='Divide part of labeled data to unlabeled.')
    parser.add_argument('--data_outdir',type=str, help='Directory of divided labeled and unlabeled data.')
    parser.add_argument('--pseudo_tau', type=float, default=0.6, help='Threshold to choose pseudo labels with high confidence.')
    parser.add_argument('--input_aug', type=bool, default=False, help='Whether to use EDA on inputs.')
    parser.add_argument('--model_aug', type=bool, default=False, help='Whether to use dropout on the model as the data augmentation.')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='Not use unlabeled data before training certain epochs.')

    # * Old arguments.
    parser.add_argument('--data_type',type=str,default='intent')
    parser.add_argument('--use_memory', default=False, type=bool, help="Whether store the learned latent variables z and other info into the memory.")    
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--tasks',nargs='+', default=['banking'])
    parser.add_argument("--data_dir", default="./PLL_DATA/", type=str, help="The path to train/dev/test data files.")   # Default parameters are set based on single GPU training
    parser.add_argument("--output_dir", default="./output/dstc", type=str,help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--tb_log_dir", default="./tb_logs/dstc", type=str,help="The tensorboard output directory.")
    parser.add_argument("--res_dir",default="./res", type=str,help="The path to save scores of experiments.")
    parser.add_argument("--gene_batch_size", default=16, type=int) # For generation.
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--do_train',type=bool, default=True)
    parser.add_argument('--gen_replay',type=bool, default=False, help='Whether use generative replay to avoid forgetting.')
    parser.add_argument('--model_path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument('--generate_dur_train', type=bool, help='Generate reconstructed input utterances during training with CVAE.')
    parser.add_argument('--temperature',type=float, default=0.95)
    parser.add_argument('--pseudo_data_ratio', type=float, default=0.05, help="How many pseudo data to generate for each learned task")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--warmup', type=int, default=100, help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--switch-time', type=float, default=0, help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    # * KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--beta_warmup', type=int, default=50000)
    parser.add_argument('--cycle', type=int, default=1000)
    parser.add_argument("--filename",type=str,help="Data original file to be preprocessed.")
    parser.add_argument("--init_model_name_or_path", default="./dir_model/gpt2", type=str, help="Path to init pre-trained model")
    parser.add_argument("--num_workers", default=1, type=int, help="workers used to process data")
    parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

    # Other parameters
    parser.add_argument("--ctx_max_len", default=128, type=int, help="Maximum input length for the sequence")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="Weight deay if we apply some.")
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
    parser.add_argument("--log_eval_res", default=True, help="Whether to log out results in evaluation process")
    parser.add_argument("--debug", action="store_true", help="Use debug mode")
   # 
    args = parser. parse_args() 
    args.log_file = os.path.join(args.output_dir, 'log.txt')

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
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    else:
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            args.device = torch.device("cpu")
    args.num_train_epochs = {task: args.num_train_epochs for task in args.tasks}

    return args

