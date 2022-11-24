# PCLL
## Introduction
-----
PCLL (Prompt-Conditioned Lifelong Learning) is build by Conversational AI Team, Alibaba DAMO Academy.

The corresponding paper has been published at EMNLP 2022 main conference: "***Prompt Conditioned VAE: Enhancing Generative Replay for Lifelong Learning in Task-Oriented Dialogue***".

## PCLL Implementation 
-----
### Requirements
```
pip install -r requirements.txt
```
### Dataset
Download the datasets from the [Google Drive](https://drive.google.com/file/d/1R99u2BYxaSZUKkmzdrs7TtnXbUISYAeV/view?usp=share_link).
Put the required datasets for intent detection and slot filling tasks under the folder `DATA`.
The datasets process scripts are also contained `DATA`. You can read the files for more details.

### Model Training and Evaluation
```
sh scripts/intent_all_train.sh # for lifelong intent detection task
sh scripts/slot_all_train.sh # for lifelong slot filling task
```
The files required for training are `lltrain.py, mycvae/model.py, mycvae/trainer.py`

## Citation
---- 
If you use our code or find PCLL useful for your work, please cite our paper as:\
@inproceedings{zhao2022cvae,\
  title={Prompt Conditioned VAE: Enhancing Generative Replay for Lifelong Learning in Task-Oriented Dialogue},
  author={Zhao, Yingxiu and Zheng, Yinhe and Tian, Zhiliang and Gao, Chang and Yu, Bowen and Yu, Haiyang and Li, Yongbin and Sun, Jian and Zhang, Nevin L.},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022},\
}


