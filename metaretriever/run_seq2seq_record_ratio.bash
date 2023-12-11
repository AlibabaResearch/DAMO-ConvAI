#!/usr/bin/env bash
# -*- coding:utf-8 -*-
export batch_size="16"
export model_name=uie-base-en
export data_name=absa/14lap
export task_name="meta"
export decoding_format='spotasoc'

source scripts/function_code.bash

for index in $(seq 1 ${run_time}); do
  output_dir=${model_folder}_run${index}

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

  ratio_data_folder=${data_folder}_ratio/seed${index}

  for ratio in $(ls ${ratio_data_folder})
  do
    run_data_folder=${ratio_data_folder}/${ratio}
    run_output_folder=${output_dir}_${ratio}

    if [[ ${max_prefix_length} == 0 ]]
    then
      run_output_folder=${run_output_folder}_noprefix
    fi

    eval_steps=$(python scripts/get_eval_batch_num.py ${run_data_folder}/train.json ${batch_size} 20)
    echo Eval each ${eval_steps} batch

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} run_seq2seq.py \
      --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
      --trainer_type=${trainer_type} \
      --load_config_only=False \
      --use_fast_tokenizer=True \
      --ddp_find_unused_parameters=False \
      --predict_with_generate \
      --evaluation_strategy=steps \
      --save_strategy=steps \
      --load_best_model_at_end \
      --metric_for_best_model eval_overall-F1 \
      --save_total_limit 1 \
      --max_source_length=${max_source_length:-"256"} \
      --max_prefix_length=${max_prefix_length:-"-1"} \
      --max_target_length=${max_target_length:-"192"} \
      --num_train_epochs=${epoch} \
      --task=${task_name} \
      --train_file=${run_data_folder}/train.json \
      --validation_file=${run_data_folder}/val.json \
      --test_file=${run_data_folder}/test.json \
      --record_schema=${run_data_folder}/record.schema \
      --per_device_train_batch_size=${batch_size} \
      --per_device_eval_batch_size=$((batch_size * 4)) \
      --output_dir=${run_output_folder} \
      --logging_dir=${run_output_folder}_log \
      --model_name_or_path=${model_name} \
      --learning_rate=${lr} \
      --source_prefix="${task_name}: " \
      --lr_scheduler_type=${lr_scheduler} \
      --label_smoothing_factor=${label_smoothing} \
      --eval_steps ${eval_steps} \
      --decoding_format ${decoding_format} \
      --warmup_ratio ${warmup_ratio} \
      --preprocessing_num_workers=4 \
      --dataloader_num_workers=0 \
      --meta_negative=${negative} \
      --meta_positive_rate=${positive} \
      --skip_memory_metrics \
      --no_remove_unused_columns \
      --ordered_prompt=${ordered_prompt} \
      --save_better_checkpoint=True \
      --spot_noise=${spot_noise} \
      --asoc_noise=${asoc_noise} \
      --seed=${seed} --disable_tqdm=${disable_tqdm} >${stdout_file} 2>${stderr_file}

    echo "Map Config" ${map_config}
    python3 scripts/sel2record.py -p ${run_output_folder} -g ${run_data_folder} -v -d ${decoding_format} -c ${map_config}
    python3 scripts/eval_extraction.py -p ${run_output_folder} -g ${run_data_folder} -w -m ${eval_match_mode:-"normal"}

    # delete all pytorch_model.bin of checkpoints in low-resource exps for saving disk
    # find ${run_output_folder}/ | grep -P "checkpoint-\d+/pytorch_model.bin" | xargs rm -rf
    # delete all optimizer.pt in low-resource exps for saving disk
    # find ${run_output_folder}/ | grep -P "optimizer.pt" | xargs rm -rf

  done

done
