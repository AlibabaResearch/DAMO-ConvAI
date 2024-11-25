<!-- # MMEvol -->

# MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct

<div align="center">
<br>
<a href="https://scholar.google.com/citations?user=phg8yxoAAAAJ&hl=zh-CN&oi=ao">Run Luo</a><sup><span>1,2*</span></sup>, 
<a>Haonan Zhang</a><sup><span>3*</span></sup>,
<a>Longze Chen</a><sup><span>1,2*</span></sup>,
<a>Ting-En Lin</a><sup><span>3*</span></sup>,
<a>Xiong Liu</a><sup><span>3</span></sup>,
<a>Yuchuan Wu</a><sup><span>3</span></sup>,
<a>Min Yang</a><sup><span>1,2🌟</span></sup>,
<a>Yongbin Li</a><sup><span>3🌟</span></sup>,
<br>
<a>Minzheng Wang<sup><span>2</span></sup>,
<a>Pengpeng Zeng<sup><span>4</span></sup>,
<a>Lianli Gao<sup><span>5</span></sup>,
<a>Heng Tao Shen<sup><span>4</span></sup>,
<a>Yunshui Li<sup><span>1,2</span></sup>,
<a>Xiaobo Xia<sup><span>6</span></sup>,
<a>FeiHuang<sup><span>3</span></sup>,
<a>Jingkuan Song<sup><span>4🌟</span></sup>,
<br>



\* Equal contribution 🌟 Corresponding author

<sup>1</sup> Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences<br>
<sup>2</sup> University of Chinese Academy of Sciences<br>
<sup>3</sup> Alibaba Group
<sup>4</sup> Tongji University 
<sup>5</sup> Independent Researcher
<sup>6</sup> The University of Sydney<br>
    
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) <a href='https://arxiv.org/pdf/2409.05840'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/models/Tongyi-ConvAI/MMEvol'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a> <a href='https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'> <a href='https://mmevol.github.io/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Project-Page-green'></a></a>

</div>

<p align="center">
    <img src="mmevol_sft_data/assets/mmevol.jpg" width="100%" height="100%">
</p>

<font size=5><div align='center' >  [[📖 arXiv Paper](https://arxiv.org/pdf/2409.05840)] [[📊 Dataset](https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol)] [[🏆 Models](https://huggingface.co/models/Tongyi-ConvAI/MMEvol)]  </div></font>
MMEvol is the first method that successfully introduces Evol-Instruct into multimodal domain to improve the diversity and complexity of multimodal instruction data. Compared with previous methods like  vila2, MIMIC-IT, and MMInstruct, it can perform iterative evolution in a very elegant and simple way in a fully automatic way, breaking through human imagination of data complexity and diversity. It has no restrictions on the form of data, the type of task, or complex processing. It can quickly perform self-iterative evolution on limited image instruction data to obtain ultra-high-quality multimodal data, thereby giving multimodal models more powerful capabilities. At the same time, it can be orthogonally combined with other data flow-driven methods such as vila2, MIMIC-IT, and MMInstruct to obtain more powerful data construction effects. Everyone is welcome to experience it now!

## 🔥 Update

- [11/10]🔥MMEvol is coming! We release the [code](https://github.com/RainBowLuoCS/MMEvol), [models](https://huggingface.co/models/Tongyi-ConvAI/MMEvol), and [data](https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol) for MMEvol!
- [09/09]🔥MMEvol is coming! We release the [paper](https://arxiv.org/pdf/2409.05840) for MMEvol!

## 👀 Contents

- [Setup](#Setup)
- [Model](#model)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Citation](#citation)


## 📷 Setup

Please follow the instructions below to install the required packages.

1. Clone this repository

```bash
git clone https://github.com/RainBowLuoCS/MMEvol.git
cd MMEvol
```

2. Install Package

```Shell
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

1. Install additional packages for training

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 📷 Hyperparameters

Both hyperparameters used in pretraining and finetuning are provided below.

| Hyperparameter | Global Batch Size | LLM lr | Projector lr | Vision Tower lr | Epochs | Max length | Weight decay |
| -------------- | ----------------: | -----: | -----------: | --------------: | -----: | ---------: | -----------: |
| PT             |               256 |      0 |         1e-3 |               0 |      1 |       4096 |            0 |
| FT             |               128 |   2e-5 |         2e-5 |            2e-6 |      1 |       4096 |            0 |

## 🔍 Model

Here are the pretrained weights and instruction tuning weights

| Model            | Pretrained Projector | Base LLM  | PT Data                                                      | IT Data | Download |
| ---------------- | -------------------- | --------- | ------------------------------------------------------------ | ------- | -------- |
| MMEvol-Qwen2-7B  | [mm_projector]()     | Qwen2-7B  | [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | MMEvol  | [ckpt]() |
| MMEvol-LLaMA3-8B | [mm_projector]()     | LLaMA3-8B | [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | MMEvol  | [ckpt]() |

### Performance

**VLMEvalKit Support (OpenCompass)**

| Model            | MME_C | MMStar | HallBench | MathVista_mini | MMMU_val | AI2D | POPE | BLINK | RWQA |
| ---------------- | ----- | ------ | --------- | -------------- | -------- | ---- | ---- | ----- | ---- |
| MMEvol-LLaMA3-8B | 47.8  | 50.1   | 62.3      | 50.0           | 40.8     | 73.9 | 86.8 | 46.4  | 62.6 |
| MMEvol-Qwen2-7B  | 55.8  | 51.6   | 64.1      | 52.4           | 45.1     | 74.7 | 87.8 | 47.7  | 63.9 |

**VLMEvalKit Not Support (VQADataSet)**

| Model            | VQA_v2 | GQA  | MIA  | MMSInst |
| ---------------- | ------ | ---- | ---- | ------- |
| MMEvol-LLaMA3-8B | 83.4   | 65.0 | 78.8 | 32.3    |
| MMEvol-Qwen2-7B  | 83.1   | 65.5 | 77.6 | 41.8    |

## 💡Preparation

### Dataset

Please follow [LLaVA](https://github.com/haotian-liu/LLaVA) to prepare the corresponding images and data.

###  data structure

```
datasets
├── json
│   ├── allava_vflan.json
│   ├── arxivqa.json
│   ├── cambrain_math_code.json
│   ├── data_engine.json
│   ├── shargpt_40k.json
│   ├── tabmwp.json
│   ├── wizardlm_143k.json
│   ├── mmevol_seed_no_evol_163k.json
│   ├── mmevol_evol_480k.json
│   └── mix_evol_sft.json
├── ai2d
│   ├── abc_images
│   ├── annotations
│   ├── images
│   ├── questions
│   └── categories.json
├── alfword
│   ├── alf-image-id-0
│   ├── alf-image-id-1
│   ├── alf-image-id-2
│   ├── alf-image-id-3
│   └── alf-image-id-4
├── allava_vflan
│   └── images
├── arxivqa
│   └── images
├── chartqa
│   ├── test
│   ├── train
│   └── val
├── coco
│   ├── train2014 
│   ├── train2017
│   ├── val2014
│   └── val2017
├── clevr
│   ├── CLEVR_GoGenT_v1.0
│   └── CLEVR_v1.0
├── data_engine
│   ├── partI
│   ├── partII 
│   └── partIII
├── design2code
│   └── images  
├── docvqa
│   ├── test
│   ├── train
│   └── val
├── dvqa
│   └── images
├── geo170k
│   ├── images/geo3k
│   └── images/geoqa_plus
├── geoqa+
│   └── images 
├── gpt4v-dataset
│   └── images 
├── gqa
│   └── images 
├── hfdata
│   └── ....
├── llava
│   └── llava_pretrain/images
├── llavar
│   └── finetune
├── mathvision
│   └── images
├── ocr_vqa
│   └── images
├── Q-Instruct-DB
│   ├── livefb_liveitw_aigc
│   └── spqa_koniq
├── sam
│   └── images
├── scienceqa
│   └── images
├── share_textvqa
│   └── images
├── synthdog-en
│   └── images
├── tabmwp
│   └── tables
├── textbookqa
│   └── tqa_train_val_test
├── textvqa
│   └── train_images
├── vg
│   ├── VG_100K
│   └── VG_100K_2
├── vizwiz
│   └── train
├── web-celebrity
│   └── images
├── web-landmark
│   └── images
└── wikiart
│   └── images

```

mmevol_evol_480k.json is the 480k evolution data evolved from the seed data mmevol_seed_no_evol_163k.json. You can freely combine other data such as allava_vflan.json for instruction ftuning (IT) training according to your personal preferences, or directly use our mixed mix_evol_sft.json for training.

## 📈 Train

### Pretrain

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and organize the data following [Preparation](#preparation) before training . Make sure set up the corresponding train script with correct setting (data path, weight path, and hyper-paramaters)

```bash
bash scripts/v1_6/train/llama3/pretrain.sh
bash scripts/v1_6/train/qwen2/pretrain.sh
```

### Visual Instruction Tuning

Please make sure you download and organize the data following [Preparation](#preparation) before training. Make sure set up the corresponding train script with correct setting (data path, weight path, and hyper-paramaters)

```bash
bash scripts/v1_6/train/llama3/finetune.sh
bash scripts/v1_6/train/qwen2/finetune.sh
```


## 📈 Evaluation

## opencompass

First, enter the `vlmevalkit` directory and install all dependencies:

```bash
cd vlmevalkit
pip install -r requirements.txt
```

<br />

Then, run `script/run_inference.sh`, which receives three input parameters in sequence: `MODELNAME`, `DATALIST`, and `MODE`. `MODELNAME` represents the name of the model, `DATALIST` represents the datasets used for inference, and `MODE` represents evaluation mode:

```bash
chmod +x ./script/run_inference.sh
./script/run_inference.sh $MODELNAME $DATALIST $MODE
```

<br />

The two available choices for `MODELNAME` are listed in `vlmeval/config.py`:

```bash
ungrouped = {
    'MMEvol-Llama3-V-1_6': partial(LLaVA_Llama3_V, model_path="checkpoints/xxx/checkpoint-14000"),
    'MMEvol-Qwen2-V-1_6': partial(LLaVA_Qwen2_V, model_path="checkpoints/xxx/checkpoint-14000"),
}

```

<br />

All available choices for `DATALIST` are listed in `vlmeval/utils/dataset_config.py`. While evaluating on a single dataset, call the dataset name directly without quotation marks; while evaluating on multiple datasets, separate the names of different datasets with spaces and add quotation marks at both ends:

```bash
$DATALIST="MME MMMU_DEV_VAL MathVista_MINI RealWorldQA MMStar AI2D_TEST HallusionBench POPE BLINK"
```

<br />

While scoring on each benchmark directly, set `MODE=all`. If only inference results are required, set `MODE=infer`. In order to reproduce the results in the table displayed on the homepage (columns between MME and RealWorldQA), you need to run the script according to the following settings:

```bash
# run on all 9 datasets
./script/run_inference.sh MiniCPM-Llama3-V-2_5 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench POPE BLINK" all

# The following are instructions for running on a single dataset
# MME
./script/run_inference.sh MMEvol-Llama3-V-1_6 MME all
# MMMU_DEV_VAL
./script/run_inference.sh MMEvol-Llama3-V-1_6 MMMU_DEV_VAL all
# MathVista_MINI
./script/run_inference.sh MMEvol-Llama3-V-1_6 MathVista_MINI all
.....

```

<br />

## vqadataset

For VQA and GQA dataset,  please follow [LLaVA](https://github.com/haotian-liu/LLaVA)  for evaluation.

For [MIA](https://github.com/apple/ml-mia-bench)  and [MMSInst](https://multi-modal-self-instruct.github.io/) , first download the dataset and then run the following scripts for evaluation

```bash
cd mmevol
# test
python llava/eval/model_vqa_mia.py
python llava/eval/model_vqa_mminst.py
# eval
python llava/eval/mia_eval.py
python llava/eval/mminst_eval.py
```

<br />

## 👀 Visualization

The Tongyi-ConvAI generates this dataset for multi-modal supervised fine-tuning. This dataset was used to train **Evol-Llama3-8B-Instruct** and **Evol-Qwen2-7B** reported in [our paper](https://arxiv.org/pdf/2409.05840). To create this dataset, we first selected 163K Seed Instruction Tuning Dataset for Evol-Instruct, then we enhance data quality through an iterative process that involves a refined combination of fine-grained perception, cognitive reasoning, and interaction evolution. This process results in the generation of a more complex and diverse image-text instruction dataset, which in turn empowers MLLMs with enhanced capabilities. Below we showcase the detailed data distribution of the SEED-163K, which is prepared for multi-round evolution mentioned above. More details can be found in our paper. 

<div align=center>
<img width="90%" src="mmevol_sft_data/assets/mmevol.jpg"/>
</div>

<div align='center' >
<details>
<summary> Click to expand more examples</summary>
<p align="center">
    <img src="mmevol_sft_data/assets/mmevol.jpg" width="60%" height="60%">
    <img src="mmevol_sft_data/assets/mmevol.jpg" width="60%" height="60%">
    <img src="mmevol_sft_data/assets/mmevol.jpg" width="60%" height="60%">
    <img src="mmevol_sft_data/assets/mmevol.jpg" width="60%" height="60%">
</details>
</div>

### Schedule

- [ ] Release MMEvol-10M
- [x] Release training & evaluation code
- [x] Release model weight
- [x] Release evolved dataset MMEvol-480K

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{luo2024mmevol,
  title={Mmevol: Empowering multimodal large language models with evol-instruct},
  author={Luo, Run and Zhang, Haonan and Chen, Longze and Lin, Ting-En and Liu, Xiong and Wu, Yuchuan and Yang, Min and Wang, Minzheng and Zeng, Pengpeng and Gao, Lianli and others},
  journal={arXiv preprint arXiv:2409.05840},
  year={2024}
}
```

### Contact

if you have any question, please consider following concat for help

- Run Luo — r.luo@siat.ac.cn

- Haonan Zhang — zchiowal@gmail.com

## Acknowledgement

\- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their brilliant contributions to the community! We just can't wait to use LLaVA-NeXT.

\- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): the amazing open-sourced suit for evaluating various LMMs!

