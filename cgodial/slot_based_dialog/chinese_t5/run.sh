# training script
python train.py --mode train --context_window 2  \
                --cfg t5_path=./t5_chinese_small seed=557 batch_size=32  cuda=False cuda_device=0 exp_no=fewshot_10p

## testing script
#python train.py --mode test --context_window 2 \
#      --cfg eval_load_path=./experiments/cwfewshot_10p_sd557_lr0.0006_bs32/epoch28_trloss1.30_gpt2   \
#      seed=557 batch_size=32  cuda=True  cuda_device=1 exp_no=fewshot_10p \
#      use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=True \
#      use_true_curr_bspn=False use_true_curr_aspn=False