#!/bin/bash
datasets=('bitod' 'doc2dial' 'metalwoz' 'mwoz' 'selfdialog' 'sgd')

for dataset in ${datasets[*]}
do
  echo "Making dataset: [${dataset}]"
  python3 generate_clustering.py --dataset ${dataset}
done
