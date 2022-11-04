#! /bin/sh

python eval/x.py

for i in `ls output_dse_model |
grep "meta" | awk -F "." '{print $3}' | sort -n`; do `python eval/common_encoding.py --step $i --dataset $1 &&
python eval/eval_selection.py --dataset $1 >> result_selection.txt`; done;

for i in `ls output_dse_model |
grep "meta" | awk -F "." '{print $3}' | sort -n`; do `python eval/common_encoding_cos.py --step $i --dataset $1 &&
python eval/eval_sts.py --dataset $1 >> result_sts.txt`; done;
