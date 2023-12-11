#!/bin/bash

source scripts_exp/meta_run.bash
# selected_gpus=${GPU:-"`get_gpu_id $gpu_node`"}
export CUDA_VISIBLE_DEVICES=$selected_gpus

# Load Hyper-parameters
IFS=' '
read -ra BATCH_SIZE <<<"${BATCH_SIZE}"
read -ra LR_RATE <<<"${LR_RATE}"
read -ra WARMUP_PROP <<<"${WARMUP_PROP}"
read -ra LABEL_SMOOTHING <<<"${LABEL_SMOOTHING}"
read -ra NEGATIVE <<<"${NEGATIVE}"
read -ra NOISE <<<"${NOISE}"

for batch_size in "${BATCH_SIZE[@]}"; do
  echo "batch " ${batch_size}
  
  for noise in "${NOISE[@]}"; do
    echo "noise " ${noise}
    for learning_rate in "${LR_RATE[@]}"; do
      echo "learning rate " ${learning_rate}
      for warmup_ratio in "${WARMUP_PROP[@]}"; do
        echo "warmup ratio " ${warmup_ratio}
        for label_smoothing in "${LABEL_SMOOTHING[@]}"; do
          echo "label smoothing " ${label_smoothing}
            for negative in "${NEGATIVE[@]}"; do
            echo "negative " ${negative}

            bash run_seq2seq_record_ratio.bash -k ${run_time} \
              -m uie_models/${model_name} \
              -d ${selected_gpus} \
              -i ${dataset_name} \
              -f ${decoding_format} \
              --trainer_type ${trainer_type} \
              --use_prompt_tuning_model ${use_prompt_tuning_model} \
              --lr_scheduler constant \
              --epoch ${epoch} \
              --eval_steps ${eval_steps} \
              --batch ${batch_size} \
              --label_smoothing ${label_smoothing} \
              --lr ${learning_rate} \
              --warmup_ratio ${warmup_ratio} \
              --max_source_length ${max_source_length} \
              --spot_noise ${noise} --asoc_noise ${noise} \
              --negative ${negative} --random_prompt --map_config ${map_config}

            bash scripts/summary_performance.bash > output/best.performance.now

          done
        done
      done
    done
  done
done

bash scripts/summary_performance.bash

exit 0
