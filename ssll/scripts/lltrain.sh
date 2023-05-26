gpuid=$1
# K=$2
K=100
R=20 #* Ratio of unlabeled data over labeled data. 
# K=5000
run=K${K}_U${R}_run
ord=0
# data_type=tc #! 
# data_type=decanlp
data_type=mix
exp=semi_${data_type}_$run
task_type=semi #!
use_unlabel=true
meantc=true #* Use mean teacher for semi-supervised learning.
echo exp: $exp
echo data_type: $data_type
echo K: $K U: $R
echo order: $ord

# * tc ===========================================
if [[ $data_type == tc ]]
then
  if [[ $ord == 0 ]]
  then
    task_list=(ag yelp dbpedia amazon yahoo) #ord0
  elif [[ $ord == 1 ]]
  then
    task_list=(yahoo amazon ag dbpedia yelp) #ord1
  elif [[ $ord == 2 ]]
  then
    task_list=(ag dbpedia yahoo yelp amazon) #ord2
  elif [[ $ord == 3 ]]
  then
    task_list=(yahoo yelp amazon dbpedia ag) #ord3
  elif [[ $ord == 4 ]]
  then
    task_list=(dbpedia yelp amazon ag yahoo) #ord4
  fi
  max_input_len=150
  max_ans_len=20
  epochs=120
  # epochs=75
  train_bs=16
  gradient_accumulation_steps=1
  eval_bs=16
  feedback_threshold=0.05
  simi_tau=0.8
  # simi_tau=0.65
  # add_confidence_selection=
  add_confidence_selection=true
fi

# * decanlp ======================================
if [[ $data_type == decanlp ]]
then
  if [[ $ord == 0 ]]
  then
    task_list=(wikisql sst woz.en squad srl) #ord0
    # task_list=(woz.en sst)
  elif [[ $ord == 1 ]]
  then
    task_list=(srl squad wikisql woz.en sst) #ord1
  elif [[ $ord == 2 ]]
  then
    task_list=(wikisql woz.en srl sst squad) #ord2
  elif [[ $ord == 3 ]]
  then
    task_list=(srl sst squad woz.en wikisql) #ord3
  elif [[ $ord == 4 ]]
  then
    task_list=(woz.en sst squad wikisql srl) #ord4
  fi
  max_input_len=512
  max_ans_len=100
  epochs=200
  # epochs=100 #!test
  train_bs=4
  gradient_accumulation_steps=4
  eval_bs=16
  feedback_threshold=0.1
  # feedback_threshold=0.000001 #!test
  simi_tau=0.6
  add_confidence_selection=true
fi
# * MIX ================================================
if [[ $data_type == mix ]]
then
  if [[ $ord == 0 ]]
  then
    task_list=(yahoo amazon woz.en squad yelp ag wikisql dbpedia sst srl) #ord0
  elif [[ $ord == 1 ]]
  then
    task_list=(yelp wikisql yahoo sst srl ag dbpedia woz.en squad amazon) #ord1
  elif [[ $ord == 2 ]]
  then
    task_list=(woz.en sst yahoo squad ag dbpedia amazon srl yelp wikisql) #ord2
  fi
  max_input_len=512
  max_ans_len=100
  epochs=200
  train_bs=4
  gradient_accumulation_steps=4
  eval_bs=16
  feedback_threshold=0.1
  simi_tau=0.6
  add_confidence_selection=true
fi

# *---------------------------------------------- 
echo task list: ${task_list[@]}

# random_initialization=true
random_initialization=
num_label=$K
unlabel_ratio=$R
use_task_pmt=true
test_all=true
# test_all= #!test
# debug_use_unlabel=true
debug_use_unlabel=
# evaluate_zero_shot=true
evaluate_zero_shot=

lr=2e-4
# warmup_epoch=1 
warmup_epoch=2 # * warm up epoch
freeze_plm=true
# freeze_plm=
evalstep=500000
# test_overfit=true #! Use label_train for evaluation
test_overfit=

# * Unlabel hyper-params.
gen_replay=true #* Perform generative replay
# gen_replay=
pseudo_data_ratio=0.1
# pseudo_data_ratio=0.01 #!test
construct_memory=true #* Construct memory (Forward Aug)
# construct_memory=
backward_augment=true #* Backward Aug 
# backward_augment= 
back_kneighbors=3 # for backward_augment to retrieve the current unlabel memory
# forward_augment=true #* Forward Aug
forward_augment=
kneighbors=3 # 
# kneighbors=5 # for forward_augment to retrieve the old memory
select_adapter=
# select_adapter=true

unlabel_amount='1'
# unlabel_amount=$unlabel_ratio
# unlabel_amount='5'
# ungamma=0.1
ungamma=0.01
add_unlabel_lm_loss=true
# add_unlabel_lm_loss=
add_label_lm_loss=true
# add_label_lm_loss=
accum_grad_iter=$unlabel_amount
# feedback_threshold=0.5
# lm_lambda=0.25
lm_lambda=0.5
KD_temperature=2
KD_term=1
diff_question=true
pseudo_tau=1.5
num_aug=3 # for EDA input augmentation
consistency=10 
consistency_rampup=30
# consistency_rampup=10
# ema_decay=0.90
ema_decay=0.95
stu_feedback=true
input_aug=true
rdrop=true
model_aug=true
# model_aug=
# rdrop=

# *--------------------------------------------------
# datadir=./data_lamol/TC
if [[ $data_type == tc ]]
then
  datadir=../../DATA/MYDATA_DIVIDED/TC/label_$K
elif [[ $data_type == decanlp ]]
then
  datadir=../../DATA/MYDATA_DIVIDED/decaNLP/label_$K
elif [[  $data_type == mix ]]
then 
  datadir=../../DATA/MYDATA_DIVIDED/MIX/label_$K
fi

output=outputs/${data_type}/$exp/ord$ord
tb_log_dir=tb_logs/${data_type}/$exp/ord$ord
log=logs/${data_type}/${exp}_ord${ord}.log
err=logs/${data_type}/${exp}_ord${ord}.err
mkdir -p logs/${data_type} outputs $output $tb_log_dir
# *--------------------------------------------------
# python_file=tc_train.py
python_file=unitrain.py

# TODO: Running ================================================
CUDA_VISIBLE_DEVICES=$gpuid \
python $python_file \
  --gpu=$gpuid \
  --data_dir=$datadir \
  --output_dir=$output \
  --tb_log_dir=$tb_log_dir \
  --tasks ${task_list[*]} \
  --experiment=$exp \
  --num_train_epochs=$epochs \
  --train_batch_size=$train_bs \
  --eval_batch_size=$eval_bs \
  --eval_steps=$evalstep \
  --data_type=$data_type \
  --use_unlabel=$use_unlabel \
  --pseudo_tau=$pseudo_tau \
  --warmup_epoch=$warmup_epoch \
  --ungamma=$ungamma \
  --unlabel_amount=$unlabel_amount \
  --meantc=$meantc \
  --consistency=$consistency \
  --num_aug=$num_aug \
  --consistency_rampup=$consistency_rampup \
  --test_overfit=$test_overfit \
  --ema_decay=$ema_decay \
  --num_label=$num_label \
  --use_task_pmt=$use_task_pmt \
  --freeze_plm=$freeze_plm \
  --input_aug=$input_aug \
  --model_aug=$model_aug \
  --stu_feedback=$stu_feedback \
  --rdrop=$rdrop \
  --test_all=$test_all \
  --unlabel_ratio=$unlabel_ratio \
  --gen_replay=$gen_replay \
  --add_unlabel_lm_loss=$add_unlabel_lm_loss \
  --add_label_lm_loss=$add_label_lm_loss \
  --lr=$lr \
  --accum_grad_iter=$accum_grad_iter \
  --feedback_threshold=$feedback_threshold \
  --lm_lambda=$lm_lambda \
  --kneighbors=$kneighbors \
  --construct_memory=$construct_memory \
  --KD_term=$KD_term \
  --diff_question=$diff_question \
  --KD_temperature=$KD_temperature \
  --add_confidence_selection=$add_confidence_selection \
  --debug_use_unlabel=$debug_use_unlabel \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --max_input_len=$max_input_len \
  --max_ans_len=$max_ans_len \
  --backward_augment=$backward_augment \
  --back_kneighbors=$back_kneighbors \
  --similarity_tau=$simi_tau \
  --select_adapter=$select_adapter \
  --pseudo_data_ratio=$pseudo_data_ratio \
  --forward_augment=$forward_augment \
  --evaluate_zero_shot=$evaluate_zero_shot  \
  --random_initialization=$random_initialization \
  > $log 2> $err 
  #! not log test
