# SSLL
## Introduction
-----
SSLL (Semi-Supervised Lifelong Language Learning) is build by Conversational AI Team, Alibaba DAMO Academy.

The corresponding paper has been published at EMNLP 2022 Findings: "***Semi-Supervised Lifelong Language Learning***".

## SSLL Implementation 
-----
### Requirements
```
pip install -r requirements.txt
```
### Dataset
The datasets used in the experiments follows [LAMOL](https://arxiv.org/abs/1909.03329).

### Model Training and Evaluation
```
sh scripts/lltrain.sh 
```
The files required for training are under the folder `unifymodel`.

## Citation
---- 
If you use our code or find SSLL useful for your work, please cite our paper as:\
@inproceedings{zhao2022semi,\
  title={Semi-Supervised Lifelong Language Learning},
  author={Zhao, Yingxiu and Zheng, Yinhe and Yu, Bowen and Tian, Zhiliang and Lee, Dongkyu and Sun, Jian and Li, Yongbin and Zhang, Nevin L.},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Findings},
  year={2022}\
}

LAMOL citation:\
@inproceedings{sun2019lamol,\
  title={LAMOL: LAnguage MOdeling for Lifelong Language Learning},
  author={Sun, Fan-Keng and Ho, Cheng-Hao and Lee, Hung-Yi},
  booktitle={International Conference on Learning Representations},
  year={2020}\
}

