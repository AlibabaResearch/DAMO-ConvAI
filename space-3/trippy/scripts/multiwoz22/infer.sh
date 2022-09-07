#!/bin/bash

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Parameters ------------------------------------------------------
TASK="multiwoz22"
DATA_DIR="data/MULTIWOZ2.2"
MAX_SEQ_LENGTH=180
MODEL_NAME=SPACE-DST
BATCH_SIZE=48

# Project paths etc. ----------------------------------------------
OUT_DIR=outputs/${TASK}/${MODEL_NAME}-${BATCH_SIZE}-${MAX_SEQ_LENGTH}
mkdir -p ${OUT_DIR}

# Main ------------------------------------------------------------
for step in test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path=model/${MODEL_NAME} \
	    --do_lower_case \
	    --learning_rate=1e-4 \
	    --num_train_epochs=50 \
	    --max_seq_length=${MAX_SEQ_LENGTH} \
	    --per_gpu_train_batch_size=${BATCH_SIZE} \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
	    --save_epochs=5 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --eval_all_checkpoints \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
      --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
	    --seed 42 \
	    --mlm_pre \
	    --mlm_during \
	    ${args_add} \
        2>&1 | tee ${OUT_DIR}/${step}.log

    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi
done
