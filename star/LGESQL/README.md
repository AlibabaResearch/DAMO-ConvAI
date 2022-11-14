# Results of STAR + LGESQL
This section presents the results on CoSQL and SParC datasets with STAR fine-tuned with LGESQL.
## Create conda environment
The following commands.

Create conda environment `lgesql`:
  - In our experiments, we use **torch==1.7.0** with CUDA version 11.0:

        conda create -n lgesql python=3.6
        source activate lgesql
        pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

  - Next, download dependencies:

        python -c "import nltk; nltk.download('punkt')"
        python -c "import stanza; stanza.download('en')"
        python -c "import nltk; nltk.download('stopwords')"

## Using our checkpoint to evaluation:
  - Download our processed datasets [CoSQL](https://drive.google.com/file/d/1suuQnHVPxZZKRiUBvsUIlw7BnY21Q_6u/view?usp=sharing) or [SParC](https://drive.google.com/file/d/1DrGBq7WGdieanq90TjkiO5JgZMwcDGUu/view?usp=sharing) and unzip them into the `cosql/data` and `sparc/data` respectively. Make sure the datasets are correctly located as:
```
data
├── database
├── dev_electra.json
├── dev_electra.bin
├── dev_electra.lgesql.bin
├── dev_gold.txt
├── label.json
├── tables_electra.bin
├── tables.json
├── train_electra.bin
├── train_electra.json
└── train_electra.lgesql.bin
```
  - Download our processed checkpoints [CoSQL](https://drive.google.com/file/d/1y4edJJ2xoA_JUGCoegEd8xLopAaUuvmp/view?usp=sharing) or [SParC](https://drive.google.com/file/d/1UDs956PgVlZT1hZ4pRm3Mox3Hs5u42sF/view?usp=sharing) and unzip them into the `cosql/checkpoints` and `sparc/checkpoints` respectively. Make sure the checkpoints are correctly located as:
```
checkpoints
├── model_IM.bin
└── params.json
```
  - Execute the following command and the results are recorded in result_XXX.txt(it will take 10 to 30 minutes on one Tesla V100-PCIE-32GB GPU):

        sh run/run_evaluation.sh

## Train from scratch
  - You can train STAR yourself by following the process in the `pretrain` file or download our pre-trained [STAR](https://drive.google.com/file/d/1zfvNpofVzLixzzFyqLO0NP-WQSKKENIC/view?usp=sharing) and unzip it into the `pretrained_models/sss` directory. Make sure the STAR are correctly located as:
```
pretrained_models
└── sss
      ├── config.json
      ├── pytorch_model.bin
      └── vocab.txt
```
  - You can preprocess the data with the `process_data&&label.py` file and refer to the methods in LGESQL, or download our processed data as described above directly. 
  - Traning:
  (it will take 4 days on one Tesla V100-PCIE-32GB GPU)

        sh run/run_lgesql_plm.sh
