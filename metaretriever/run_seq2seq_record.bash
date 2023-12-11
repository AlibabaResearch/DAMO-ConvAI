#!/usr/bin/env bash
# -*- coding:utf-8 -*-
export batch_size="16"
export model_name=uie-base-en
export data_name=absa/14lap
export task_name="meta"
export decoding_format='spotasoc'

source scripts/function_code.bash

for index in $(seq 1 ${run_time}); do

  if [[ ! ${output_dir} ]]
  then
    output_dir=${model_folder}_run${index}
    echo "output_dir is not provided so create it automatically: ${output_dir}"
  else
    echo "output_dir is provided: ${output_dir}"
  fi

  # output_dir=${model_folder}_run${index}

  if [[ ${verbose} == true ]]
  then
    stdout_file=/dev/stdout
    stderr_file=/dev/stderr
    disable_tqdm=False
  else
    stdout_file=${output_dir}.log
    stderr_file=${output_dir}.err
    disable_tqdm=True
  fi

  # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} gdb --args ${run_command} run_seq2seq.py \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} run_seq2seq.py \
    --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
    --use_prompt_tuning_model=${use_prompt_tuning_model} \
    --trainer_type=${trainer_type} \
    --load_config_only=False \
    --use_fast_tokenizer=True \
    --ddp_find_unused_parameters=False \
    --predict_with_generate \
    --evaluation_strategy=${evaluation_strategy} \
    --save_strategy=${evaluation_strategy} \
    --metric_for_best_model eval_overall-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=${max_source_length:-"256"} \
    --max_prefix_length=${max_prefix_length:-"-1"} \
    --max_target_length=${max_target_length:-"192"} \
    --num_train_epochs=${epoch} \
    --task=${task_name} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --test_file=${data_folder}/test.json \
    --record_schema=${data_folder}/record.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --logging_dir=${output_dir}_log \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=100 \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --source_prefix="${task_name}: " \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format ${decoding_format} \
    --warmup_ratio ${warmup_ratio} \
    --preprocessing_num_workers=32 \
    --dataloader_num_workers=32 \
    --meta_negative=${negative} \
    --meta_positive_rate=${positive} \
    --skip_memory_metrics \
    --no_remove_unused_columns \
    --ordered_prompt=${ordered_prompt} \
    --save_better_checkpoint=False \
    --start_eval_step=${start_eval_step:-"0"} \
    --spot_noise=${spot_noise} \
    --asoc_noise=${asoc_noise} \
    --seed=${seed}${index} --disable_tqdm=${disable_tqdm} >${stdout_file} 2>${stderr_file}
  echo "exit code:" $?

    # --save_strategy=${evaluation_strategy} \
    # --save_total_limit 1 \
    # --load_best_model_at_end \

    # --save_strategy="steps" \
    # --save_steps=5000 \
    # --save_total_limit 9999999 \
    # --load_best_model_at_end=True \

  if [[ ${verbose} != true ]]
  then
    tail -n 200 ${stderr_file}
  fi

  echo "Map Config" ${map_config}
  python3 scripts/sel2record.py -p ${output_dir} -g ${data_folder} -v -d ${decoding_format} -c ${map_config}
  python3 scripts/eval_extraction.py -p ${output_dir} -g ${data_folder} -w -m ${eval_match_mode:-"normal"}

    # delete all optimizer.pt for saving disk
  # find ${output_dir}/ | grep -P "optimizer.pt" | xargs rm -rf

done
