
# Code for "Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions"

This repository contains the official implementation for reproducing the experiments in our paper.


## 🛠️ Setup Instructions

### 0.1 Environment

- Python 3.10 is required.
- Install dependencies:

```bash
pip install -r requirements.txt
```


### 0.2 Base Model

- Download **Qwen2.5-VL-3B-Instruct** from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
- Place it in a directory **outside** this project (e.g., `../llm_path/Qwen/Qwen2.5-VL-3B-Instruct`), so the full path is:
  ```
  ../llm_path/Qwen/Qwen2.5-VL-3B-Instruct/
  ```


### 0.3 Data

We provide related scripts for downloading and processing required datasets in the `./data` folder.


---

## Part 1: Latent Action Space Learning

### Run Pretraining

```bash
bash pretrain.sh
```

---

## Part 2: Latent Action Reinforcement Learning  
(Example: **MMRole**)

### 📌 Preliminary Setup

Before running RL, configure API access for:

| Component | Location | Task |
|---------|----------|------|
| Reward Model (RM) | `eval_results/api_utils.py` | Fill in your API key / endpoint for reward scoring |
| LLM-as-a-Judge (final eval) | `sampling_results/api_utils.py` | Configure judge model|

---

### 2.1 Training

Run RL on **MMRole**:

```bash
bash run_MMRole_RL.sh
```

This script:
- Loads the pretrained `PolicyActionVLM` from Part 1.
- Optimize the latent action policy via RL.
- Generates evaluation results and saved to `sampling_results/*.json`.

---

### 2.2 Evaluation

Run automatic evaluation using LLM-as-a-Judge:

```bash
cd sampling_results
python MMRole_Eval.py
```



---
**Reference**
```bibtex
@misc{li-2026-controlling,
  title         = {Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions},
  author        = {Yongqi Li and Hao Lang and Tieyun Qian and Yongbin Li},
  year          = {2026},
  eprint        = {2601.07516},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.07516}
}
```