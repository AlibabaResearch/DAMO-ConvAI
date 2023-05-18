function fulldata_hfdata() {
SAVING_PATH=$1
PRETRAIN_MODEL_PATH=$2
TRAIN_BATCH_SIZE=$3
EVAL_BATCH_SIZE=$4
TRAIN_EPOCH=$5
GRAD_ACCU_STEPS=$6
LOGGING_STEPS=${7}
LR=${8}


mkdir -p ${SAVING_PATH}

python -m torch.distributed.launch --nproc_per_node=4 ./ablationknowledge.py \
  --model_name_or_path t5-base \
  --output_dir ${SAVING_PATH} \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
  --num_train_epochs ${TRAIN_EPOCH} \
  --warmup_ratio 0.1 \
  --logging_steps ${LOGGING_STEPS} \
  --learning_rate ${LR} \
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
  --ddp_find_unused_parameters False 2>&1 | tee ${SAVING_PATH}/log
}

SAVING_PATH=./ll
TRAIN_BATCH_SIZE=$1
EVAL_BATCH_SIZE=$2
TRAIN_EPOCH=$3
GRAD_ACCU_STEPS=2
LOGGING_STEPS=5
LR=1e-4

MODE=try


SAVING_PATH=${SAVING_PATH}/lifelong/${MODE}
fulldata_hfdata  ./ll  t5-base ${TRAIN_BATCH_SIZE} ${EVAL_BATCH_SIZE} ${TRAIN_EPOCH} 2 5 1e-4

