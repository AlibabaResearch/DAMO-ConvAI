# Examples

This directory contains example scripts for training LLMs/VLMs using MCoreAdapter.
These examples leverage the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) library, specifically utilizing its data processing and training argument capabilities.

## Quick Start

### Prerequisites

1. Install LLaMA-Factory first:

```bash
pip install git+https://github.com/hiyouga/LLaMA-Factory.git@b83a38eb98965fa698936a911b73aac017e73e88
```

2. Install TransformerEngine:

```bash
pip install transformer_engine[pytorch]
```

### Fine-Tuning

```bash
# sft qwen3-8b
bash train/run_sft_train.sh
# dpo llama3.1-8b
bash train/run_dpo_train.sh
# sft qwen2.5-vl-7b
bash train/run_vl_sft.sh
```
