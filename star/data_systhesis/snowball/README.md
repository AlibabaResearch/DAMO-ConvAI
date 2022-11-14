# Snowball Framework for [Logic-Consistency Text Generation from Semantic Parses](https://aclanthology.org/2021.findings-acl.388/)

This is the official pytorch implementation of the Snowball Framework in the Findings of ACL 2021 paper:
- "[Logic-Consistency Text Generation from Semantic Parses](https://aclanthology.org/2021.findings-acl.388/)", Findings of ACL 2021

Please cite the papers if you use our data or code.
```
@inproceedings{shu-etal-2021-logic,
    title = "Logic-Consistency Text Generation from Semantic Parses",
    author = "Shu, Chang  and
      Zhang, Yusen  and
      Dong, Xiangyu  and
      Shi, Peng  and
      Yu, Tao  and
      Zhang, Rui",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.388",
    doi = "10.18653/v1/2021.findings-acl.388",
    pages = "4414--4426",
}

```

Please feel free to contact Chang Shu or Yusen Zhang for any question.

### Dependency

The model is tested in Python 3.7, Pytorch 1.0.1 and Huggingface Transformers 3.2.0 (Later version may cause unexpected errors). 

**Please make sure to use the previous version of BART model**

bart-base: `https://huggingface.co/facebook/bart-base/tree/d7db7a74f47943f8488724cb52b03cd4dd82fab4`

bart-large: `https://huggingface.co/facebook/bart-large/tree/2ee0f9b9079f8312f1a8fc76cd1de7166ae332ea`

Since our hyper-parameters were tuned on the previous version of BART and Huggingface keeps updating the structure and parameters of the BART models, using the latest version of BART models may not be able to reproduce the results we reported in our paper.


We recommend using `conda` and `pip`:

```
conda create -n relogic python=3.7
conda activate relogic
pip install -r requirements.txt
```

### Run Snowball Framework
First, make sure the SQL-to-Text and Logic-to-Text datasets are correctly located as:

```
data
├── logic2text
│   ├── eval
│   │   ├── dev.json
│   │   └── test.json
│   ├── preprocessed
│   │   ├── dev.json
│   │   ├── mutation.json
│   │   ├── test.json
│   │   ├── train.json
│   │   └── translated_all.json
│   └── raw
│       ├── dev.json
│       ├── propcess_raw.py
│       ├── test.json
│       └── train.json
├── preprocessed_data
│   ├── bart_parser_label_mapping.json
│   ├── bart_parser_label_mapping_2.json
│   ├── bart_parser_pretrain_label_mapping.json
│   ├── rat_label_mapping.json
│   ├── sp_BIOES_label_mapping.json
│   └── sql_tagging.json
└── spider
    ├── eval
    │   ├── dev.json
    │   └── test.json
    ├── newdata
    │   ├── dev.json
    │   └── test.json
    ├── preprocessed
    │   ├── dev.json
    │   ├── mutation.json
    │   ├── out_of_domain.json
    │   ├── test.json
    │   └── train.json
    └── raw
        ├── README.txt
        ├── dev.json
        ├── dev_gold.sql
        ├── out_of_domain.json
        ├── tables.json
        ├── test.json
        ├── train.json
        ├── train_gold.sql
        ├── train_others.json
        └── train_spider.json
```

- All scripts for running experiments on SQL-to-Text are located in: `/scripts/spider`

Example:
```
#!/bin/bash

export LC_ALL="en_US.utf8"
export RAW_DIR=data/spider/raw/
export PREPROCESS_DIR=data/spider/preprocessed
export EVAL_DIR=data/spider/eval
export PRETRAIN_DIR=saves/spider_snow_ball_base

output=saves/spider_snow_ball_base

CUDA_VISIBLE_DEVICES=0 python -m run_snowball \
   --output_dir=${output} \
   --tokenizer_name facebook/bart-base \
   --config_name facebook/bart-base \
   --translated_logic \
   --gen_do_test \
   --gen_do_eval \
   --gen_do_eval \
   --eval_do_test \
   --eval_do_eval \
   --gen_do_out_domain_test \
   --snow_ball_mode scratch \
   --pretrain_dir $PRETRAIN_DIR \
   --raw_dir $RAW_DIR \
   --preprocess_dir $PREPROCESS_DIR\
   --evaluator_dir $EVAL_DIR\
   --num_snowball_iterations 5 \
   --gen_learning_rate 1e-5 \
   --gen_num_train_epochs 10 \
   --gen_save_epochs 5 \
   --gen_eval_epochs 1 \
   --gen_logging_steps 25 \
   --gen_per_device_train_batch_size 24 \
   --gen_per_device_eval_batch_size 24\
   --gen_evaluate_during_training \
   --gen_seed 42 \
   --eval_learning_rate 5e-7 \
   --eval_num_train_epochs 5 \
   --eval_save_epochs 5 \
   --eval_eval_epochs 1 \
   --eval_logging_steps 25 \
   --eval_per_device_train_batch_size 8 \
   --eval_per_device_eval_batch_size 8\
   --eval_evaluate_during_training \
   --eval_seed 42 \
   --overwrite_output_dir
```
- Similarly, All scripts for running experiments on Logic-to-Text are located in: `/scripts/logic2text`

- The logs, generated text, and saved models of each Snowball training iterations are saved in `/saves` directory. For instance, the following directories contain all the experiments results of self-augmentation, evaluator and generator on SQL-to-Text dataset with BERT-base model in the first Snowball iteration.

```
── spider_snow_ball_large
    ├── augmentation
    │   ├── 0
	.....
    ├── evaluator
    │   ├── 0
    │   │   ├── pred
    │   │   └── test
	.....
    └── generator
        ├── 0
        │   ├── eval
        │   ├── out_domain_test
        │   └── test
	....
```

Those scripts reproduce the experiment results reported in the paper:
|            |      SQL2Text       |          |          |          |      |      |       |    |      |
|------------|:-------------------:|----------|----------|----------|------|------|-------|----|------|
| Metrics    |         BLEC        |          |          |          |      |      | Human |    |      |
| Snowball   | -                   | 1        | 2        | 3        | 4    | 5    |   -   | 4  | κ    |
| BART-base  | 76.4                | 78.6     | 78.5     | **84.1** | 79.7 | 78.1 | 22    | 45 | 0.69 |
| BART-large | 91.8                | 91.3     | **93.7** | 91.8     | 93.2 | 93.0 | 75    | 74 | 0.7  |
|            | Logic2Text          |          |          |          |      |      |       |    |      |
| Metrics    |         BLEC        |          |          |          |      |      | Human |    |      |
| Snowball   | -                   | 1        | 2        | 3        | 4    | 5    |   -   | 4  |   κ  |
| BART-base  | 87.9                | 86.1     | **88.6** | 87.4     | 87.7 | 87.8 | 83    | 85 | 0.48 |
| BART-large | 86.7                | **87.8** | 85.2     | 87.1     | 86.0 | 88.5 | 86    | 78 | 0.48 |


In the paper, we only report the best results we observed on the test set. From the practical perspective, you may estimate the best number of Snowball iteration on unseen datasets based on the model performance on the Dev set. For instance, the comparison between the results on Dev and Test split of Logic-to-Text is as follow:
|          | Metrics    | BLEC |          |          |      |      |      | Human |    |      |
|----------|------------|:----:|----------|----------|------|------|------|:-----:|----|------|
|          | Snowball   | -    | 1        | 2        | 3    | 4    | 5    |   -   | 4  |   κ  |
| Dev Set  | BART-base  | 89.3 | 91.0     | **91.6** | 91.2 | 88.8 | 89.7 |       |    |      |
|          | BART-large | 87.3 | 88.4     | **90.3** | 88.8 | 87.4 | 87.5 |       |    |      |
| Test Set | BART-base  | 87.9 | 86.1     | **88.6** | 87.4 | 87.7 | 87.8 | 83    | 85 | 0.48 |
|          | BART-large | 86.7 | **87.8** | 85.2     | 87.1 | 86.0 | 88.5 | 86    | 78 | 0.48 |




### BLEC Auto Evaluation
File path:
- Logic2text: preprocess/Logic2Text/snowball_evaluation/generator test.py
- Spider: preprocess/sql_auto_evaluation/eval.py

Running:
- prepare the json file: the file being tested is a dict with file name as key and the test file extracted from the snowball running results as value, such as the following example.
```json
{
  "~/relogic-snowball/saves/logic2text_snow_ball_base/generator/0/test/epoch_10.json": [
    {
      "idx": 0,
      "logic": "both ( ( ( pts ) of ( ( year ) of ( all rows ) that fuzzy matches ( 1995 ) ) ) is greater than ( ( pts ) of ( ( year ) of ( all rows ) that fuzzy matches ( 1991 ) ) ) ) and ( both ( ( ( pts ) of ( ( year ) of ( all rows ) that fuzzy matches ( 1995 ) ) ) is equal to ( 13 ) ) and ( ( ( pts ) of ( ( year ) of ( all rows ) that fuzzy matches ( 1991 ) ) ) is equal to ( 1 ) ) are true ) are true",
      "pred": "in the 1995 season, jim leffler scored more points than he did in the 1991 season.",
      "label": "mark blundell scored more points in his 1995 formula one race than he did in his 1991 formula one race."
    },
    {
      "idx": 1,
      "logic": "both ( the unique values of ( ( position ) of ( ( college ) of ( all rows ) that fuzzy matches ( saskatchewan ) ) that fuzzy matches ( k ) ) ) and ( ( ( player ) of ( ( position ) of ( ( college ) of ( all rows ) that fuzzy matches ( saskatchewan ) ) that fuzzy matches ( k ) ) ) is the same as ( matt kellett ) ) are true",
      "pred": "matt kellett was the only player drafted by the washington redskins from saskatchewan college.",
      "label": "the only kicker drafted by saskatchewan college in the 1998 cfl draft was matt kellett."
    },
    ...
}
```
- modify the path in Python file, e.g. eval.py, and generate test.py. And directly run that python file
- the algorithm is based on rules which will take less than half minute to run all the tests!
- for the cleaner and independent version of BLEC, please visit [here](https://github.com/chatc/BLEC)

