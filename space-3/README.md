# SPACE
This repository contains code and data for the **SIGIR'2022** paper "**Unified Dialog Model Pre-training for Task-Oriented Dialog
Understanding and Generation**".

## Abstract
Recently, pre-training methods have shown remarkable success in task-oriented dialog (TOD) systems. However, most existing pre-trained models for TOD focus on either dialog understanding or dialog generation, but not both. In this paper, we propose FORTUNE, a novel unified pre-trained dialog model learning from large-scale dialog corpora with limited annotations, which can be effectively fine-tuned on a wide range of downstream dialog tasks. Specifically, FORTUNE consists of four successive components in a single transformer to maintain a task-flow in TOD systems: (i) a dialog encoding module to encode dialog history, (ii) a dialog understanding module to extract semantic vectors from either user queries or system responses, (iii) a dialog policy module to generate a policy vector that contains high-level semantics of the response, and (iv) a dialog generation module to produce appropriate responses. We design a dedicated pre-training objective for each component. Concretely, we pre-train the dialog encoding module with span mask language modeling to learn contextualized dialog information. To capture the structured dialog semantics, we pre-train the dialog understanding module via a novel tree-induced semi-supervised contrastive learning objective with the help of extra dialog annotations. In addition, we pre-train the dialog policy module by minimizing the L2 distance between its output policy vector and the semantic vector of the response for policy optimization. Finally, the dialog generation model is pre-trained by language modeling. Results show that FORTUNE achieves state-of-the-art performance on eight downstream dialog benchmarks, including intent prediction, dialog state tracking, and end-to-end dialog modeling. We also show that FORTUNE has a stronger few-shot ability than existing models under the low-resource setting.

## Main Results
SPACE performs end-to-end dialog modeling, dialog state tracking and intent prediction, which achieves new state-of-the-art results on all eight benchmark datasets including: BANKING77, CLINC150, HWU64, CamRest, In-Car Assistant, MultiWOZ2.0, MultiWOZ2.1 and MultiWOZ2.2.

| Intent Prediction | BANKING77 | CLINC150 | HWU64 |
|:-------------------:|:-----:|:-----:|:-----:|
|       Accuracy      | 94.94 | 97.89 | 94.14 |

| Dialog State Tracking | Joint Goal Accuracy |
|:-------------------:|:-----:|
|       MultiWOZ2.2       | 57.50 |

| End-to-End Modeling | Inform | Success |  BLEU | Combined Score |
|:-------------------:|:------:|:-------:|:-----:|:--------------:|
|     MultiWOZ2.0     |  95.30 |  88.00  | 19.30 |     110.95     |
|     MultiWOZ2.1     |  95.60 |  86.10  | 19.91 |     110.76     |

| End-to-End Modeling | Match | SuccF1 |  BLEU | Combined Score |
|:-------------------:|:-----:|:------:|:-----:|:--------------:|
|   In-Car Assistant  | 85.26 |  83.16 | 22.92 |     107.13     |
|       CamRest       | 97.74 |  88.24 | 23.68 |     116.67     |

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
- transformers == 2.9.1
```
We use the tokenization tool in SpaCy and you can directly install python packages by commands: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.

## Preparation
### Path Definition
Define your own paths `<YOUR_PROJECT_PATH>` and `<YOUR_SAVE_PATH>` in scripts as follows: 
```sh
PROJECT_NAME="SPACE"  # project name (fixed)
PROJECT_ROOT=<YOUR_PROJECT_PATH>/${PROJECT_NAME}  # root path of this project
SAVE_ROOT=<YOUR_SAVE_PATH>/${PROJECT_NAME}  # root path of model's output
```

### Data Preparation
Download data-split1 from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/data.zip). 

The downloaded zip file `data.zip` contains pre-training corpora (including BANKING77, CLINC150 and HWU64) and four extra task-oriented (TOD) benchmark datasets: CamRest, In-Car Assistant, MultiWOZ2.0 and MultiWOZ2.1, which have already been processed. You need to put the unzipped directory `data` into the project directory `SPACE` for the subsequent training.

Download data-split2 from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/trippy/data.zip).

The downloaded zip file `data.zip` contains one TOD benchmark dataset: MultiWOZ2.2, which have already been processed. You need to put the unzipped directory `data` into the directory `SPACE/trippy` for the subsequent training.

### Directory Structure
Upon completion, your SPACE folder should contain the following:
```
SPACE/
├── data  # pre-training corpora and downstream TOD datasets
├── db  # database in multiwoz2.0 dataset
├── model  # pre-trained checkpoints and vocabulary
├── outputs  # fine-tuned checkpoints
├── scripts  # inference bashes
├── space  # model and modules
├── tools  # misc tools
└── trippy  # separated code for dialog state tracking (dst)
    ├── data  # multiwoz2.2 datasets
    ├── dataset_config  # data configuration
    ├── model  # pre-trained checkpoint
    ├── outputs  # fine-tuned checkpoints for multiwoz2.2
    └── scripts  # inference bashes for multiwoz2.2
```

## Pre-training
### Pre-training Corpora
- [AnPreDial](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/AnPreDial.zip): a new labeled dialog dataset annotated with semantic trees, which contains 32 existing labeled TOD datasets with 3
million turns, ranging from single-turn QA to multi-turn dialogs.
- [UnPreDial](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/UnPreDial.zip): a large-scale unlabeled dialog dataset consisting of 19M utterances with careful processing from 21 online dialog corpora, ranging from online forums to conversational machine reading comprehension.

### Pre-trained Checkpoint
- [SPACE](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/model.zip): an uncased model (12-layers, 768-hidden, 12-heads, 110M parameters)

You need to unzip the downloaded model file `model.zip`, then put the unzipped directory `model` into the project directory `SPACE` for the further fine-tuning.

### Training
We pre-train the SPACE on limited labeled dialogs (**AnPreDial**) and large-scale unlabeled dialog corpora (**UnPreDial**) via semi-supervised learning.
You can pre-train SPACE from scratch by running the following scripts:

```sh
# Step 1: Preprocess pre-training corpora
sh scripts/pre_train/preprocess.sh

# Step 2: Begin pre-training
sh scripts/pre_train/train.sh
```

## Fine-tuning
### Fine-tuned Checkpoints
Download checkpoints-split1 from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/outputs.zip). 

The downloaded zip file `outputs.zip` contains our best fine-tuned checkpoints on the following seven datasets: 
- MultiWOZ2.0, MultiWOZ2.1, In-Car Assistant, CamRest (**End-to-End Modeling**)
- BANKING77, CLINC150, HWU64 (**Intent Prediction**)

If you want to reproduce our reported results, you should put the unzipped directory `outputs` into the directory `${SAVE_ROOT}` (set in scripts). 

Download checkpoints-split2 from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/trippy/outputs.zip). 

The downloaded zip file `outputs.zip` contains our best fine-tuned checkpoints on one dataset: 
- MultiWOZ2.2 (**Dialog State Tracking**)

If you want to reproduce our reported results, you should put the unzipped directory `outputs` into the directory `SPACE/trippy`.
Then you can directly run the inference scripts of different datasets for the reproduction, which will be introduced later.

### Training
We fine-tune the SPACE on the eight TOD datasets. You can fine-tune SPACE from scratch by running the following training scripts.

For end-to-end dialog modeling (**E2E**) task:
```sh
# Training on MultiWOZ2.0
sh scripts/multiwoz2.0/train.sh

# Training on MultiWOZ2.1
sh scripts/multiwoz2.1/train.sh

# Training on In-Car Assistant
sh scripts/kvret/train.sh

# Training on CamRest
sh scripts/camrest/train.sh
```

For intent prediction (**IP**) task:
```sh
# Step 1: Unsupervised Training on BANKING77, CLINC150 and HWU64
sh scripts/pre_train/train_intent.sh

# Step 2.1: Supervised Training on BANKING77
sh scripts/banking/train.sh

# Step 2.2: Supervised Training on CLINC150
sh scripts/clinc/train.sh

# Step 2.3: Supervised Training on HWU64
sh scripts/hwu/train.sh
```

For dialog state tracking (**DST**) task:
```sh
# Step 1: Unsupervised Training
sh scripts/pre_train/train_dst.sh

# Step 2: Supervised Training on MultiWOZ2.2
cd SPACE/trippy/
sh scripts/multiwoz22/train.sh
```

> **NOTE**: You can skip Step 1 if you directly download the output model of Step 1. 
> For DST task, you can download the model file from this [link](http://datarepo0.oss-cn-hangzhou-zmf.aliyuncs.com/Alibaba/SPACE3.0-ALL/trippy/model.zip).
> You need to unzip the downloaded model file `model.zip`, then put the unzipped directory `model` into the directory `SPACE/trippy` for the further fine-tuning.

### Inference
After collecting some fine-tuned checkpoints (by directly using ours or fine-tuning SPACE from scratch by yourself), you can do the inference on the test sets of these datasets by running the following inference scripts.

For end-to-end dialog modeling (**E2E**) task:
```sh
# Inference on MultiWOZ2.0
sh scripts/multiwoz2.0/infer.sh

# Inference on MultiWOZ2.1
sh scripts/multiwoz2.1/infer.sh

# Inference on In-Car Assistant
sh scripts/kvret/infer.sh

# Inference on CamRest
sh scripts/camrest/infer.sh
```

For intent prediction (**IP**) task:
```sh
# Inference on BANKING77
sh scripts/banking/infer.sh

# Inference on CLINC150
sh scripts/clinc/infer.sh

# Inference on HWU64
sh scripts/hwu/infer.sh
```

For dialog state tracking (**DST**) task:
```sh
# Inference on MultiWOZ2.2
cd SPACE/trippy/
sh scripts/multiwoz22/infer.sh
```

> **NOTE**: For reproduction, all the best hyper-parameters have already been set in corresponding scripts and you can follow them to run.

## References
- For the data preparation and evaluation on BANKING77/CLINC150/HWU64, we refer to the code of [DialoGLUE](https://github.com/alexa/dialoglue).
- For the data preparation and evaluation on MultiWOZ2.2, we refer to the code of [TripPy](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public).
- For the data preparation and evaluation on MultiWOZ2.0/MultiWOZ2.1/In-Car Assistant/CamRest, we refer to the code of [GALAXY](https://github.com/siat-nlp/GALAXY).

## Citation
If you use our code or find SPACE useful in your work, please cite our paper as:

```bib
@inproceedings{he2022unified,
  title={Unified Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation},
  author={He, Wanwei and Dai, Yinpei and Yang, Min and Sun, Jian and Huang, Fei and Si, Luo and Li, Yongbin},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={187--200},
  year={2022}
}
```

## Contact
For personal communication related to SPACE, please contact Wanwei He (`ww.he@siat.ac.cn`).
