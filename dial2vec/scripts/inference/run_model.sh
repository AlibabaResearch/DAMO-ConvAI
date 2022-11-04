#!/bin/bash

datasets=('bitod' 'doc2dial' 'sgd' 'selfdialog' 'mwoz' 'metalwoz')
n_datasets=${#datasets[@]}

model=$1

for i in $(seq 0 $(expr ${n_datasets} - 1))
do
  echo "Running ${model} on ${datasets[i]} for inference..."
  sh scripts/inference/run_${model}.sh ${datasets[i]}
done
