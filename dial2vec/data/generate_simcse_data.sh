#!/bin/bash
datasets=('bitod' 'doc2dial' 'metalwoz' 'mwoz' 'selfdialog' 'sgd')
data_directory="datasets"
output_directory="./github/SimCSE/data"

for dataset in ${datasets[*]}
do
  nohup python3 -u generate_simcse_data.py \
                   --dataset ${dataset} \
                   --data_directory ${data_directory} \
                   --output_directory ${output_directory} \
                   > dial2vec_to_simcse_${dataset}.log 2>&1 &
done
