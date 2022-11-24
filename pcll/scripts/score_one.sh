exp=$1
ord=$2
data_type=$3

if [ "$data_type" = "intent" ];
then
    intent_list=$4
    echo ${5}
    tasks=${intent_list}
else
    echo $data_type
    slot_list=$4
    tasks=${slot_list}
fi

mkdir -p res res/cvae_outs/

res=res/cvae_outs/${exp}_
output=cvae_outputs/${exp}/$ord
python final_score.py \
  --res_dir=$res \
  --output_dir=$output \
  --tasks $tasks \
  --data_type=$data_type \
  > cvae_logs/score_${exp}.log 2> cvae_logs/score_${exp}.err

# seed_list=(2 12 32 52)
# for seed in ${seed_list[@]};
# do
#   exp=sd_${seed}_all_intent8
#   res=res/${exp}_
#   output=outputs/${exp}/${exp}/
#   python final_score.py \
#     --res_dir=$res \
#     --output_dir=$output \
#     --tasks ${intent_list[*]}
#     # > logs/score_$exp.log 2> logs/score_$exp.err 
# done
# echo "Finish calculating scores of the experiments."
