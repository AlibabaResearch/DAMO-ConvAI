#!/bin/bash
m1=model1
m1n=model1
m2=model2
m2n=model2
eval_model=gpt-3.5-turbo-0301 # evaluaotr gpt-4 or gpt-3.5-turbo-0301

python3 -u inference-WideDeep.py \
    --all_file evaluation_data/LLMEval2_benchmark.json \
    --output log/WideDeep_${eval_model}.json \
    --eval-model $eval_model \
    --answer-model-name $m1n $m2n 
