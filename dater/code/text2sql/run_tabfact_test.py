import os
from pathlib import Path
import argparse


ROOT_DIR = str(Path("."))

# ROOT_DIR = os.path.join(os.path.dirname(__file__))
# print(os.path.dirname(__file__))
# print(ROOT_DIR)
# ROOT_DIR = '.'
# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsing_num_rows',type=int,default=10000)
    args = parser.parse_args()
    os.system(fr"""{TOKENIZER_FALSE} python -m scripts.annotate_sql_program_tabfact --dataset tab_fact \
    --dataset_split test \
    --prompt_file templates/prompts/tab_fact_parsing.txt \
    --parsing_num_rows {args.parsing_num_rows}\
    --n_parallel_prompts 1 \
    --n_processes 30 \
    --max_generation_tokens 150 \
    --temperature 0.4 \
    --sampling_n 10 \
    -v""")


    ## exe fraction
    os.system(fr"""{TOKENIZER_FALSE}python -m scripts.execute_sql_program_tabfact --dataset tab_fact \
    --dataset_split test \
    --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
    --input_program_file sql_program_tab_fact_test.json \
    --output_program_execution_file sql_program_tab_fact_test_exec.json \
    --allow_none_and_empty_answer \
    --vote_method answer_biased \
    --answer_biased 1 \
    --answer_biased_weight 3 \
    """)
