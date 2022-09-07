# SPACE1.0
## Introduction
**SPACE** (**S**emi-supervised **P**re-tr**A**ined **C**onversation Mod**E**l) is a collection of pretrained conversational models (PCMs) built by Conversational AI Team, Alibaba DAMO Academy. The series of SPACE1.0 / SPACE2.0 / SPACE3.0 have achieved 
state-of-the-art results on various task-oriented dialog benchmarks, such as  MultiWOZ and DialoGLUE.

This repository contains code and data for SPACE1.0 (i.e., GALAXY). The corresponding paper has been published at **AAAI 2022**: "*GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection*".

Full version with Appendix is here: [[PDF]](https://arxiv.org/abs/2111.14592)


## Main Results
SPACE1.0 perform end-to-end dialog modeling and achieve new state-of-the-art results on four TOD benchmark datasets: MultiWOZ2.0, MultiWOZ2.1, In-Car Assistant and CamRest.

| End-to-End Modeling | Inform | Success |  BLEU | Combined Score |
|:-------------------:|:------:|:-------:|:-----:|:--------------:|
|     MultiWOZ2.0     |  94.40 |  85.30  | 20.50 |     110.35     |
|     MultiWOZ2.1     |  95.30 |  86.20  | 20.01 |     110.76     |

| End-to-End Modeling | Match | SuccF1 |  BLEU | Combined Score |
|:-------------------:|:-----:|:------:|:-----:|:--------------:|
|   In-Car Assistant  | 85.26 |  83.60 | 23.03 |     107.46     |
|       CamRest       | 98.50 |  87.73 | 24.15 |     117.26     |

:bangbang: New SOTA results on MultiWOZ (End-to-End Modeling & Policy Optimization) evaluated by the [standardized scoring scripts](https://github.com/Tomiinek/MultiWOZ_Evaluation), which are officially recommended for the fair evaluations. We also add new results to the [official leaderboard](https://github.com/budzianowski/multiwoz) and new predictions to [this repository](https://github.com/Tomiinek/MultiWOZ_Evaluation/tree/master/predictions).

| MultiWOZ | Inform | Success |  BLEU | Combined Score |
|:-------------------:|:------:|:-------:|:-----:|:--------------:|
|     End-to-End Modeling     |  85.40 |  75.70  | 19.64 |     100.2     |
|     Policy Optimization     |  92.80 |  83.50  | 19.92 |     108.1     |

## Requirements
```
- torch == 1.8.0+cu111
- scikit-learn == 0.23.1
- numpy == 1.18.5
- nltk == 3.5
- spacy == 2.3.5
- scipy == 1.5.0
- regex == 2020.6.8
- tqdm == 4.60.0
```
We use the tokenization tool in SpaCy and you can directly install python packages by commands: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.

## Preparation
### Path Definition
Define your own paths `<YOUR_PROJECT_PATH>` and `<YOUR_SAVE_PATH>` in scripts as follows: 
```sh
PROJECT_NAME="SPACE1.0"  # project name (fixed)
PROJECT_ROOT=<YOUR_PROJECT_PATH>/${PROJECT_NAME}  # root path of this project
SAVE_ROOT=<YOUR_SAVE_PATH>/${PROJECT_NAME}  # root path of model's output
```

### Data Preparation
Download data from this [link](https://drive.google.com/file/d/1oi1w_zNH-GAMfav6slVXIF1usALHJJNQ/view?usp=sharing). 

The downloaded zip file `data.zip` contains pre-training corpora and four TOD benchmark datasets: MultiWOZ2.0, MultiWOZ2.1, In-Car Assistant and CamRest, which have already been processed. You need to put the unzipped directory `data` into the project directory `SPACE1.0` for the subsequent training.

## Pre-training
### Pre-training Corpora
- [UniDA](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE1.0/Pre-training%20Data.zip): a new labeled dialog dataset consisting of 975,780 utterances, which are annotated with 20 frequently-used DAs, according to our proposed comprehensive unified DA taxonomy for task-oriented dialog.
- [UnDial](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE1.0/Pre-training%20Data.zip): a large-scale unlabeled dialog dataset consisting of 35M utterances with careful processing, ranging from online forum chatting logs to customer service conversations.

### Pre-trained Checkpoint
- [SPACE1.0](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE1.0/model.zip): an uncased model with DA classification head (12-layers, 768-hidden, 12-heads, 109M parameters)

You need to unzip the downloaded model file `model.zip`, then put the unzipped directory `model` into the project directory `SPACE1.0` for the further fine-tuning.

### Training
We pre-train the SPACE1.0 on limited labeled dialogs (**UniDA**) and large-scale unlabeled dialog corpora (**UnDial**) via semi-supervised learning.
You can pre-train SPACE1.0 from scratch by running the following scripts:

```sh
# Step 1: Preprocess pre-training corpora
sh scripts/pre_train/preprocess.sh

# Step 2.1: Multi-GPU training on one machine
sh scripts/pre_train/train_single.sh

# Step 2.2: Multi-GPU training across multiple machines (distributed training)
sh scripts/pre_train/train_multi.sh
```
> **NOTE**: For multi-GPU training, you only need to choose Step 2.1 or Step 2.2.
> It is worth noting that if you choose Step 2.2, you should have a well-equipped GPU cluster to support such training.

## Fine-tuning
### Fine-tuned Checkpoints
Download checkpoints from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE1.0/outputs.zip). 

The downloaded zip file `outputs.zip` contains our best fine-tuned checkpoints on different datasets: 
- the **7-th** epoch on MultiWOZ2.0 (**60** training epochs in total)
- the **5-th** epoch on MultiWOZ2.1 (**60** training epochs in total)
- the **89-th** epoch on In-Car Assistant (**100** training epochs in total)
- the **18-th** epoch on CamRest (**60** training epochs in total)

If you want to reproduce our reported results, you should put the unzipped directory `outputs` into the directory `${SAVE_ROOT}` (set in scripts). 
Then you can directly run the inference scripts of different datasets for the reproduction, which will be introduced later.

### Training
We fine-tune the SPACE1.0 on the four TOD datasets and focus on the end-to-end dialog modeling (**E2E**) task.
You can fine-tune SPACE1.0 from scratch by running the following training scripts:

```sh
# Training on MultiWOZ2.0 (8 GPUs)
sh scripts/multiwoz2.0/train.sh

# Training on MultiWOZ2.1 (8 GPUs)
sh scripts/multiwoz2.1/train.sh

# Training on In-Car Assistant (1 GPU)
sh scripts/kvret/train.sh

# Training on CamRest (1 GPU)
sh scripts/camrest/train.sh
```
> **NOTE**: For MultiWOZ2.0 and MultiWOZ2.1, we also maintain the DA prediction task to alleviate the model discrepancy between pre-training and fine-tuning. On the other hand, we discard this task on the In-Car Assistant and CamRest due to the lack of useful DAs in these two datasets.
Besides, we support both multi-GPU and single-GPU training, you can jointly tune the hyper-parameter `${BATCH_SIZE}$` and `${GRADIENT_ACCUMULATION_STEPS}$` to maintain originally offered batch size when single-GPU training.

### Inference
After collecting some fine-tuned checkpoints (by directly using ours or fine-tuning SPACE1.0 from scratch by yourself), you can do the inference on the test sets of these datasets by running the following inference scripts:

```sh
# Inference on MultiWOZ2.0 (1 GPU)
sh scripts/multiwoz2.0/infer.sh

# Inference on MultiWOZ2.1 (1 GPU)
sh scripts/multiwoz2.1/infer.sh

# Inference on In-Car Assistant (1 GPU)
sh scripts/kvret/infer.sh

# Inference on CamRest (1 GPU)
sh scripts/camrest/infer.sh
```
> **NOTE**: For reproduction, all the best hyper-parameters have already been set in corresponding scripts and you can follow them to run.
If you fine-tune SPACE1.0 from scratch by yourself, the 4-th/60 to 7-th/60 training epochs could offer you the best inference performance on MultiWOZ2.0/2.1.

## References
- For the data preparation and evaluation on MultiWOZ2.0/2.1, we refer to the code of [UBAR](https://github.com/TonyNemo/UBAR-MultiWOZ).
- For the data preparation and evaluation on In-Car Assistant/CamRest, we refer to the code of [LABES](https://github.com/thu-spmi/LABES).

## Citation
If you use our code or find SPACE1.0 useful in your work, please cite our paper as:

```bib
@article{he2022galaxy,
  title={GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection},
  author={He, Wanwei and Dai, Yinpei and Zheng, Yinhe and Wu, Yuchuan and Cao, Zheng and Liu, Dermot and Jiang, Peng and Yang, Min and Huang, Fei and Si, Luo and others},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## Contact
- For help or issues using SPACE1.0, please refer to the maintained [Repository and Issues](https://github.com/siat-nlp/GALAXY).  
- For personal communication related to SPACE1.0, please contact Wanwei He (`ww.he@siat.ac.cn`).
