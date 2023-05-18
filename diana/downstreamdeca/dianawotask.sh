function fulldata_hfdata() {
SAVING_PATH=$1
PRETRAIN_MODEL_PATH=$2
TRAIN_BATCH_SIZE=$3
EVAL_BATCH_SIZE=$4





mkdir -p ${SAVING_PATH}

python -m torch.distributed.launch --nproc_per_node=4 dianawotask.py \
  --model_name_or_path ${PRETRAIN_MODEL_PATH} \
  --output_dir ${SAVING_PATH} \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --logging_steps 5 \
  --learning_rate 1e-4 \
  --predict_with_generate \
  --num_beams 4 \
  --save_strategy no \
  --evaluation_strategy no \
  --weight_decay 1e-2 \
  --max_source_length 512 \
  --label_smoothing_factor 0.1 \
  --do_lowercase True \
  --load_best_model_at_end True \
  --greater_is_better True \
  --save_total_limit 10 \
  --ddp_find_unused_parameters True 2>&1 | tee ${SAVING_PATH}/log
}

SAVING_PATH=$1
TRAIN_BATCH_SIZE=$2
EVAL_BATCH_SIZE=$3





MODE=try


SAVING_PATH=${SAVING_PATH}/lifelong/${MODE}
fulldata_hfdata  ${SAVING_PATH}  t5-base ${TRAIN_BATCH_SIZE} ${EVAL_BATCH_SIZE} 
