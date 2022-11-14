# 🌟 STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-red.svg">
    </a>
  	<a href="https://github.com/huggingface/transformers/tree/main/examples/research_projects/tapex">
      <img alt="🤗 transformers support" src="https://img.shields.io/badge/🤗 transformers-master-green" />
    </a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg">
    </a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg">
    </a>
    <br />
</p>

This is the official project containing source code for the EMNLP 2022 paper "STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing"

You can use our checkpoint to evaluation directly or train from scratch with our instructions.

1. File `data_systhesis` contains code to generate conversational text-to-SQL data.
2. File `pretrain` contains code to pre-train STAR model.
3. File `LGESQL` contains fine-tune and evaluation code.

The relevant models and data involved in the paper can be downloaded through [Baidu Netdisk](https://pan.baidu.com/s/1uA63h4zpwyDSqY5cprbeJQ?pwd=6666), or downloaded through Google Drive in the corresponding folder.

## Citation
```
@article{cai2022star,
  title={STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing},
  author={Cai, Zefeng and Li, Xiangyu and Hui, Binyuan and Yang, Min and Li, Bowen and Li, Binhua and Cao, Zheng and Li, Weijie and Huang, Fei and Si, Luo and others},
  journal={arXiv preprint arXiv:2210.11888},
  year={2022}
}
```

## 🪜 Pretrain


### Create conda environment

The following commands.

Create conda environment `star`:

- In our experiments, we use **torch==1.7.0** with CUDA version 11.0
- We use four GeForce A-100 for our pre-trained experiments.

  conda create -n star python=3.6
  conda activate star
  pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt

### Unzip pretraining dataset

Download and move the pretaining data file [pretrain_data.txt](https://drive.google.com/file/d/1YF7Kx0TZMyS_5BJ8GmsFXfuraiogxBID/view?usp=sharing) into the directory `datasets`.

### Training


```python
python pretain_inbatch.py
```

It may takes two days on four Tesla V100-PCIE-32GB GPU.

### Saving STAR model

```python
python save_model.py
```

Then you can get the trained model and its configuration (at least containing `model.bin` and `config.json`) under `pretrained/sss` direction.

## 🚪 Fine-tuning and Evaluation

This section presents the results on CoSQL and SParC datasets with STAR fine-tuned with LGESQL.

### Create conda environment
The following commands.

Create conda environment `lgesql`:
  - In our experiments, we use **torch==1.7.0** with CUDA version 11.0:
    ```
    conda create -n lgesql python=3.6
    source activate lgesql
    pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```
  - Next, download dependencies:
    ```
    python -c "import nltk; nltk.download('punkt')"
    python -c "import stanza; stanza.download('en')"
    python -c "import nltk; nltk.download('stopwords')"
    ```
### Using our checkpoint to evaluation:
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
    ```
    sh run/run_evaluation.sh
    ```

### Train from scratch
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
    ```
    sh run/run_lgesql_plm.sh
    ```

