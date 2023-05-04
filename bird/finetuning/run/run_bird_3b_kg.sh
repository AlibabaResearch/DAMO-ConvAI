gpu=0
run_name='T5_3b_bird_knowledge'
output_dir='output/T5_3b_bird_kg'

export WANDB_API_KEY=''
export WANDB_PROJECT=bird


echo '''flying'''

CUDA_VISIBLE_DEVICES=4 python train_bird.py --seed 1 --cfg Salesforce/T5_3b_finetune_bird_kg.cfg \
--run_name ${run_name} --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps \
--eval_steps 1000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 1000 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 200 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --do_predict false --predict_with_generate true \
--output_dir ${output_dir} --per_device_train_batch_size 2 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true