#!/bin/bash

# control every stage whether requestion codex or use saved results 

col_select=0
row_select=0
decompose=0
parsing_excution_filling=0
reasoning=0
dataset_split=test
parsing_num_rows=5


cd scripts/wtq
################### col #################################
echo "Select Col"
if [ $col_select = 1 ];
then
    echo "Select Col -> Requset Codex"
    python run_col.py --dataset_split ${dataset_split}
fi

# ################### row ##################################
echo "Select Row"
if [ $row_select = 1 ];
then
    echo "Select Row -> Requset Codex"
    python run_row.py --dataset_split ${dataset_split}
fi
# # ###########################################################
cd ../../results/wtq/span/
python gen_sub_table.py --dataset_split ${dataset_split}
cd ../../../scripts/wtq
################### decompose #############################

echo "Decompose Question"
if [ $decompose = 1 ];
then
    echo "Decompose Question -> Requset Codex"
    python run_cloze.py --dataset_split ${dataset_split}
fi

cd ../../results/wtq/cloze/
python filter_cloze.py --dataset_split ${dataset_split}
cp wikitq_${dataset_split}_decomposed.jsonl ../../../text2sql/templates
################### parsing-excution-filling #############################
cd ../../../text2sql

echo "Run parsing-excution-filling"
if [ $parsing_excution_filling = 1 ];
then
    echo "Reasoning -> Requset Codex"
    python run_wtq_${dataset_split}.py --parsing_num_rows ${parsing_num_rows}
fi
cd results
python filling.py --dataset wikitq --dataset_split ${dataset_split}
cp wikitq_${dataset_split}_exec.jsonl ../../results/wtq/cloze
cd ../../scripts/wtq
# ################### reasoning #############################

echo "Run Final Reasoning"
if [ $reasoning = 1 ];
then
    echo "Reasoning -> Requset Codex"
    python run_end2end.py --dataset_split ${dataset_split}
fi
#################### eval #################################
cd ../../results/wtq
python eval.py --split ${dataset_split}