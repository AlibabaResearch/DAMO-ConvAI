gpuno="0"

backbone='unsup_simcse'
# backbone = 'sup_simcse'
dataset=$1
feature_checkpoint="${backbone}.${dataset}.features.pkl"


CUDA_VISIBLE_DEVICES=${gpuno} \
python3 run.py \
  --stage "embedding" \
  --backbone ${backbone} \
  --feature_checkpoint ${feature_checkpoint} \
  --test_batch_size 40 \
  --use_turn_embedding False \
  --use_role_embedding False \
  --use_sep_token False \
  --start_token "[CLS]" \
  --sep_token "[SEP]" \
  --dataset ${dataset}


CUDA_VISIBLE_DEVICES=${gpuno} \
python3 run.py \
  --stage "eval_from_embedding" \
  --backbone ${backbone} \
  --feature_checkpoint ${feature_checkpoint} \
  --test_batch_size 40 \
  --use_turn_embedding False \
  --use_role_embedding False \
  --use_sep_token False \
  --start_token "[CLS]" \
  --sep_token "[SEP]" \
  --dataset ${dataset}

  # > ./logs/dial2vec_${backbone}_${dataset}_${stage}_-1Epochs_${n_gpu}GPU.log 2>&1
