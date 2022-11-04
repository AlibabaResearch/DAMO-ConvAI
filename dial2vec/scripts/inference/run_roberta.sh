gpuno="0"
n_gpu=(${gpuno//,/ })
n_gpu=${#n_gpu[@]}

echo "Using ${n_gpu} GPU with DDP Training."

backbone='roberta'
dataset=$1
stage='test'
temperature=0.2
max_turn_view_range=100

CUDA_VISIBLE_DEVICES=${gpuno} \
python3 -m torch.distributed.launch \
           --nproc_per_node=${n_gpu} \
           --master_port 23333 \
           run.py \
  --stage ${stage} \
  --backbone ${backbone} \
  --temperature ${temperature} \
  --max_turn_view_range ${max_turn_view_range} \
  --use_turn_embedding False \
  --use_role_embedding False \
  --use_sep_token False \
  --max_seq_length 512 \
  --dataset ${dataset} 

# > ./logs/dial2vec_${backbone}_${dataset}_${stage}_-1Epochs_${n_gpu}GPU.log 2>&1
