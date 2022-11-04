#!/bin/bash

datasets=('bitod' 'doc2dial' 'sgd' 'selfdialog' 'mwoz' 'metalwoz')
epochs=(10 5 5 2 2 2)
n_datasets=${#datasets[@]}

for i in $(seq 0 $(expr ${n_datasets} - 1))
do
  echo "Training dial2vec on ${datasets[i]}..."
  sh scripts/train/run_plato.sh ${datasets[i]} "train" ${epochs[i]}
done
