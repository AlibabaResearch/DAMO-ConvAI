gpuid=$1
exp=$2
ord=$3
data_type=$4

if [ "$data_type" = "intent" ];
then
    # echo ${5}
    intent_list=${5}
    tasks=${intent_list}
    epochs=$6 
else
    # echo $data_type
    slot_list=$5
    tasks=${slot_list}
    epochs=$6 
fi
# echo $epochs
gene_batch_size=64
topk=100
topp=0.90
do_train=true
checkpoint=
generate_dur_train=
train_bs=64
evalstep=80
pseudo_ratio=$7 #*
logeval=true # Print out prediction on dev dataset.
gen_replay=true #* Whether do generative replay
# gen_replay=
# only_decoder=true #* No latent code z, not vaes.
only_decoder=
# only_vae=true #* Only consider vae loss, no lm loss. 
only_vae=
# alpha_z=0
alpha_z=0.1 #* latent z representation ratio
# alpha_z=0.5 #* latent z representation ratio

seed=42
classIL=  # * Class incremental learning: no task id during testing
num_cycle=4 # Number of cycles for annealing
ctx_max_len=50
add_kd=true
KD_term=0.5
# KD_term=0.8
KD_temperature=1
# share_params=true
share_params=
save_z=
# save_z=true
use_memory= # * Whether to use memory.
# use_memory=true
memory_size=50
z_dim=128

datadir=./DATA/
output=cvae_outputs/$exp/$ord
memory_path=$output
tb_log_dir=tb_logs/cvae_outs/$exp/$ord
mkdir -p tb_logs tb_logs/cvae_outs cvae_outputs cave_outputs/$exp $tb_log_dir $output  cvae_logs/  

py=python
# TODO: Running ===========================================================
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=$gpuid \
python lltrain.py \
  --gpu=$gpuid \
  --data_dir=$datadir \
  --fp16 \
  --output_dir=$output \
  --tb_log_dir=$tb_log_dir \
  --tasks $tasks \
  --experiment=$exp \
  --top_k=$topk \
  --top_p=$topp \
  --generate_dur_train=$generate_dur_train \
  --gene_batch_size=$gene_batch_size \
  --do_train=$do_train \
  --model_path=$checkpoint \
  --num_train_epochs=$epochs \
  --train_batch_size=$train_bs \
  --eval_steps=$evalstep \
  --gen_replay=$gen_replay \
  --pseudo_data_ratio=$pseudo_ratio \
  --log_eval_res=$logeval \
  --only_decoder=$only_decoder \
  --only_vae=$only_vae \
  --use_memory=$use_memory \
  --memory_path=$memory_path \
  --alpha_z=$alpha_z \
  --seed=$seed \
  --classIL=$classIL \
  --num_cycle=$num_cycle \
  --ctx_max_len=$ctx_max_len \
  --data_type=$data_type \
  --add_kd=$add_kd \
  --KD_term=$KD_term \
  --KD_temperature=$KD_temperature \
  --share_params=$share_params \
  --save_z=$save_z \
  --memory_size=$memory_size \
  --z_dim=$z_dim \
  > cvae_logs/${exp}_$ord.log 2>&1

  
