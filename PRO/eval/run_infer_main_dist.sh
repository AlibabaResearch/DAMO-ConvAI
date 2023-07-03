export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

id=$1
ranking_len=$2
# 30 min
accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
    --index $id \
    --stage $ranking_len > logs/generate_infer_main_${id}_${ranking_len}.log 2>&1

#10 min
accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
    --index $id \
    --stage $ranking_len > logs/reward_infer_main_${id}_${ranking_len}.log 2>&1

#1 second
python -u infer_and_eval_main_score.py \
    --index $id \
    --stage $ranking_len > logs/score_infer_main_${id}_${ranking_len}.log 2>&1

# total 40 min