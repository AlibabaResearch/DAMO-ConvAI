export MASTER_PORT=1092

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

para_result=./$1
para=${para_result}/task4_para.pt

data=../../dataset/simmc2.1/teststd_public_withlast.tsv
selected_cols=0,1,2,3
split='test'

generation_pred_json=${para_result}/test_predict.json
split_path=../../../data_dstc11/simmc2.1_dials_dstc11_teststd_public.json
save_path=${para_result}/dstc11-simmc-teststd-pred-subtask-4


CUDA_VISIBLE_DEVICES=$2 python3 -m torch.distributed.launch --nproc_per_node=$3 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${para} \
    --user-dir=${user_dir} \
    --task=simmc2 \
    --batch-size=12 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${para_result} \
    --beam=50 \
    --max-len-b=100 \
    --no-repeat-ngram-size=8 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"


python3 format_task4_generation.py \
    --generation-pred-json=${generation_pred_json} \
    --split-path=${split_path} \
    --save-path=${save_path}

:<<!
python3 response_evaluation_forall.py \
    --data_json_path=${split_path} \
    --model_response_path=${save_path} \
    --para_name=${para} \
    --bleu_path=${para_result} \
    --single_round_evaluation
!

