 #!/bin/bash


# control every stage whether requestion codex or use saved results 
col_select=0
row_select=0
decompose=0
parsing_excution_filling=0
reasoning=0
parsing_num_rows=5


cd scripts/tabfact
################### col #################################
echo "Select Col"
if [ $col_select = 1 ];
then
    echo "Select Col -> Requset Codex"
    python run_col.py
fi

################### row ##################################
echo "Select Row"
if [ $row_select = 1 ];
then
    echo "Select Row -> Requset Codex"
    python run_row.py
fi
###########################################################
cd ../../results/tabfact/span/
python gen_sub_table.py
cd ../../../scripts/tabfact
################### decompose #############################
echo "Decompose Question"
if [ $decompose = 1 ];
then
    echo "Decompose Question -> Requset Codex"
    python run_cloze.py
fi
cd ../../results/tabfact/cloze/
python filter_cloze.py
cp tabfact_decomposed.jsonl ../../../text2sql/templates
cd ../../../
################### parsing-excution-filling #############################
echo "Run parsing-excution-filling"
cd text2sql
if [ $parsing_excution_filling = 1 ];
then
    echo "Reasoning -> Requset Codex"
    python run_tabfact_test.py --parsing_num_rows ${parsing_num_rows}

fi
cd results
python filling.py --dataset tabfact --dataset_split test
cp tabfact_test_exec.jsonl ../../results/tabfact/cloze
cd ../../scripts/tabfact
################### reasoning #############################

echo "Run Final Reasoning"
if [ $reasoning = 1 ];
then
    echo "Reasoning -> Requset Codex"
    python run_end2end.py
fi

#################### eval #################################
cd ../../results/tabfact
python eval.py
