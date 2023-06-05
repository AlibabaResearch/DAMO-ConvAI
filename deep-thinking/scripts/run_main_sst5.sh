export BITSANDBYTES_NOWELCOME=1
export TOKENIZERS_PARALLELISM=false

DATA_NAME=sst5
prompt_version=default

# 0 1 2 3 4
seed=$1
# 1 2 3 4
num_k_shots=$2
# true false
in_8bit=$3

gpus=$4

echo "Start: data=$DATA_NAME, prompt=$prompt_version, seed=$seed, num_k_shot=$num_k_shots, in_8bit=$in_8bit, gpu=$gpus"

exemplar_method=stratified
batch_size=32 # -1 for forced re-find
step_size=0.01
momentum=0.9

active_models=("opt" "gpt2" "e-gpt" "bloom")
#active_models=("opt")
#active_models=("gpt2")
#active_models=("e-gpt")
#active_models=("bloom")

kv_iter=15

# 1A. run OPT
echo "------------------------------------------------------"
if [[ " ${active_models[*]} " =~ "opt" ]]; then
  echo "OPT enabled"
  declare -a size=("125m" "350m" "1.3b" "2.7b" "6.7b" "13b")
  len_size=${#size[@]}

  for (( i=0; i<$len_size; i++ ))
  do
    model_size=${size[$i]}
    echo "Running OPT @ $model_size"
    python task_logprob_main.py \
      --dataset $DATA_NAME --prompt_version $prompt_version \
      --exemplar_method $exemplar_method --num_k_shots $num_k_shots \
      --model_type opt --model_size $model_size \
      --kv_iter $kv_iter \
      --step_size $step_size --momentum $momentum \
      --batch_size $batch_size \
      --gpus $gpus \
      --in_8bit $in_8bit --seed $seed
    sleep 5s
  done
else
  echo "OPT disabled"
fi

# 2. run GPT-2
echo "------------------------------------------------------"
if [[ " ${active_models[*]} " =~ "gpt2" ]]; then
  echo "GPT-2 enabled"
  declare -a size=("sm" "medium" "large" "xl")
  len_size=${#size[@]}

  for (( i=0; i<$len_size; i++ ))
  do
    model_size=${size[$i]}
    echo "Running GPT-2 @ $model_size"
    python task_logprob_main.py \
      --dataset $DATA_NAME --prompt_version $prompt_version \
      --exemplar_method $exemplar_method --num_k_shots $num_k_shots \
      --model_type gpt2 --model_size $model_size \
      --kv_iter $kv_iter \
      --step_size $step_size --momentum $momentum \
      --batch_size $batch_size \
      --gpus $gpus \
      --in_8bit $in_8bit --seed $seed
    sleep 5s
  done
else
  echo "GPT-2 disabled"
fi

# 4. run bigscience/BLOOM
echo "------------------------------------------------------"
if [[ " ${active_models[*]} " =~ "bloom" ]]; then
  echo "bigscience/BLOOM enabled"
  declare -a size=("560m" "1b1" "1b7" "3b" "7b1")
  len_size=${#size[@]}

  for (( i=0; i<$len_size; i++ ))
  do
    model_size=${size[$i]}
    echo "Running bigscience/BLOOM @ $model_size"
    python task_logprob_main.py \
      --dataset $DATA_NAME --prompt_version $prompt_version \
      --exemplar_method $exemplar_method --num_k_shots $num_k_shots \
      --model_type bloom --model_size $model_size \
      --kv_iter $kv_iter \
      --step_size $step_size --momentum $momentum \
      --batch_size $batch_size \
      --gpus $gpus \
      --in_8bit $in_8bit --seed $seed
    sleep 5s
  done
else
  echo "bigscience/BLOOM disabled"
fi


# 3. run EleutherAI/gpt
echo "------------------------------------------------------"
if [[ " ${active_models[*]} " =~ "e-gpt" ]]; then
  echo "EleutherAI/gpt enabled"
  declare -a size=("neo-125M" "neo-1.3B" "neo-2.7B" "j-6B" "neox-20b")
  len_size=${#size[@]}

  for (( i=0; i<$len_size; i++ ))
  do
    model_size=${size[$i]}
    echo "Running EleutherAI/gpt @ $model_size"
    python task_logprob_main.py \
      --dataset $DATA_NAME --prompt_version $prompt_version \
      --exemplar_method $exemplar_method --num_k_shots $num_k_shots \
      --model_type e-gpt --model_size $model_size \
      --kv_iter $kv_iter \
      --step_size $step_size --momentum $momentum \
      --batch_size $batch_size \
      --gpus $gpus \
      --in_8bit $in_8bit --seed $seed
    sleep 5s
  done
else
  echo "EleutherAI/gpt disabled"
fi

 