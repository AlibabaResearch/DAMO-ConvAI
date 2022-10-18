
# S²SQL: Injecting Syntax to Question-Schema Interaction Graph Encoder for Text-to-SQL Parsers

The PyTorch implementation of paper S²SQL: Injecting Syntax to Question-Schema Interaction Graph Encoder for Text-to-SQL Parsers （ACL 2022 Findings)

Please star this repo and cite paper if you want to use it in your work.

## Step 1: Env Setup
Our experimental environment is consistent with [LGESQL](https://github.com/rhythmcao/text2sql-lgesql), so we can basically follow their settings.

Firstly, create conda environment text2sql:
In our experiments, we use `torch==1.6.0` and `dgl==0.5.3` with CUDA version `10.1`

We use `NVIDIA V100-32GB` for all experiments
```
conda create -n text2sql python=3.6
source activate text2sql
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
Next, download dependencies:
```bash
 python -c "import stanza; stanza.download('en')"
 python -c "import nltk; nltk.download('stopwords')"
```
Download electra-large-discriminator from Hugging Face Model Hub, into the pretrained_models directory.
```bash
 mkdir -p pretrained_models && cd pretrained_models
 git lfs install
 git clone https://huggingface.co/google/electra-large-discriminator
``` 

## Step 2: Data Preparation
Download, unzip and rename the [spider.zip](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0) into the directory data.

Merge the data/train_spider.json and data/train_others.json into one single dataset data/train.json.

Preprocess the train and dev dataset, including input normalization, schema linking, graph construction and output actions generation.
```
 ./run/run_preprocessing.sh
```

## Step 3: Training and Evaluation
Training and eval S²SQL models with ELECTRA:

```bash
#msde: mixed static and dynamic embeddings
#mmc: multi-head multi-view concatenation
./run/run_lgesql_plm.sh [mmc|msde] electra-large-discriminator
./run/run_evaluation.sh 
```

## Acknowledgements
We would like to thank Tao Yu, Yusen Zhang for running evaluations on our submitted models.

We are also grateful to [LGESQL](https://github.com/rhythmcao/text2sql-lgesql) and [RATSQL](https://github.com/microsoft/rat-sql) that inspires our works.

