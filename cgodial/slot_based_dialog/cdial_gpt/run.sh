# training script
# python train.py -mode train -cfg cuda=True  cuda_device=1 exp_no=fewshot_10p

# testing script
python train.py -mode test -cfg eval_load_path=./experiments/all_fewshot_10p_sd1234_lr0.0001_bs2_ga16/epoch29_trloss0.31_gpt2 use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True cuda=True