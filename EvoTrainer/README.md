# EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning

[[Paper]](https://arxiv.org/abs/2606.03108) | English | [中文](README_zh.md)

EvoTrainer is an autonomous training framework that co-evolves LLM policies and training-side diagnostic harnesses through empirical feedback. It diagnoses rollout-level evidence, revises diagnostics, backtests interventions, and accumulates reusable skills.

Built on top of the [ROLL](https://github.com/alibaba/ROLL) framework (Reinforcement Learning Optimization for Large-Scale Learning).

<p align="center">
  <img src="docs/static/teaser.png" width="700" alt="EvoTrainer Framework Overview"/>
</p>

## Key Results

| Domain | Base (no RL) | Human-Engineered RL | EvoTrainer |
|--------|:---:|:---:|:---:|
| SWE-9B (BC%) | 30.19 | 33.77 | **38.16** (+7.97) |
| SWE-4B (BC%) | 24.68 | 31.17 | **31.49** (+6.81) |
| Math AIME 2024 (Avg@8) | 77.50 | 80.83 | **84.17** (+6.67) |
| Math AIME 2025 (Avg@8) | 67.50 | 71.67 | **73.33** (+5.83) |
| Math CNMO 2024 (Avg@8) | 75.00 | 77.78 | **81.94** (+6.94) |
| Coding (Avg@8) | 46.71 | 50.71 | **51.29** (+4.58) |

## Datasets

### Training Data

| Domain | Dataset | Size | Source |
|--------|---------|------|--------|
| Math | Big-Math-RL-Verified | 6,429 problems | [HuggingFace: SynthLabsAI/Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) |
| Coding | TACO-verified | 11,897 problems | [HuggingFace: BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) |
| SWE | swe-rebench-v6 (train split) | 8,622 instances | [HuggingFace: nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) |

### Evaluation Data

| Domain | Benchmark | Size | Source |
|--------|-----------|------|--------|
| Math | AIME 2024 | 30 problems | [HuggingFace: math-ai/aime24](https://huggingface.co/datasets/math-ai/aime24) |
| Math | AIME 2025 | 30 problems | [HuggingFace: math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| Math | CNMO 2024 | 18 problems | Chinese National Math Olympiad 2024 |
| Coding | LiveCodeBench-v6 (AtCoder subset) | 175 problems | [GitHub: LiveCodeBench](https://github.com/livecodebench/livecodebench) |
| SWE | swe-rebench-v6 (test split) | 77 instances (Python) | [HuggingFace: nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) |

### Data Format

Training data should be in JSONL format. Each line is a JSON object with the following fields:

**Math / Coding:**
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "domain": "math_rule" | "code_sandbox",
  "tag": "<dataset_tag>"
}
```

**SWE (Agentic):**
```json
{
  "meta": {
    "instance_id": "repo__issue_id",
    "messages": [{"role": "user", "content": "<problem_statement>"}],
    "repo": "owner/repo",
    "base_commit": "<commit_hash>"
  }
}
```

## Evaluation Protocol

All domains use **Avg@8** (mean over 8 independent rollouts per item) with temperature=1.0 and seed 42.

| Domain | Environment | Evaluation Method |
|--------|------------|-------------------|
| **SWE** | `roll/pipeline/agentic/env/qwen_mcp_swe/` | BC% — Binary Correctness via hidden fail-to-pass unit tests in Docker |
| **Math** | `roll/pipeline/agentic/env/math_laj/` | Correctness judged by a frozen Qwen3.5-4B LLM judge |
| **Coding** | `roll/pipeline/agentic/env/algo_coding/` | stdin/stdout execution against test cases in Docker sandbox |

## Installation

```bash
pip install -e .
pip install -r requirements_common.txt

# Choose one backend:
pip install -r requirements_torch280_vllm.txt   # vLLM backend
# or
pip install -r requirements_torch280_sglang.txt  # SGLang backend
```

## Quick Start

### 1. Prepare Model and Data

```bash
# Set environment variables
export MODEL_PATH=/path/to/your/base/model        # e.g., Qwen3.5-4B or Qwen3.5-9B
export DATA_DIR=/path/to/your/data                 # directory containing training JSONL files
export OUTPUT_DIR=/path/to/output                  # training output directory
```

### 2. Launch Agentic Training

```bash
python examples/start_agentic_pipeline.py --config_path examples/qwen35-4b-agentic/train_swe_v1.yaml
```

### 3. Configuration

Training configurations are in `examples/qwen35-{4b,9b}-agentic/`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rollout_batch_size` | Number of prompts per generation batch | 32 |
| `sequence_length` | Maximum sequence length | 131072 |
| `ppo_epochs` | PPO update epochs | 1 |
| `adv_estimator` | Advantage estimator (grpo/gae) | grpo |
| `pg_clip` / `pg_clip_high` | Clip-Higher bounds | 0.2 / 0.27 |
| `reward_clip` | Reward clipping threshold | 2 |
| `custom_envs.*.env_config.max_steps` | Max agent interaction turns | 100 |

## Reproducing SWE-9B v8

The released configuration `examples/qwen35-9b-agentic/train_swe_v1.yaml` provides the v1 baseline. To reproduce v8 (our best SWE version, BC%=38.16), apply the following modifications to `roll/pipeline/agentic/env/qwen_mcp_swe/env.py` and the training yaml:

### 1. Reward: Add Instruction-Following LLM Judge

In `env.py`, after computing `correctness_reward`, add an IF scoring call:

```python
# v8: Instruction-Following LLM Judge
if self.config.enable_instruction_following:
    if_score = await self.instruction_following_judge(trajectory)  # Returns [0.0, 1.0]
    final_reward = correctness_reward * 1.0 + if_score * 0.1 + sbe_reward + ett_reward
```

Add to `config.py`:
```python
enable_instruction_following: bool = True
instruction_following_model: str = "/path/to/reward_model"  # e.g., Qwen3.5-27B
```

### 2. Reward: Add SBE (Search-Before-Edit) and ETT (Edit-Then-Test)

In `env.py`, implement behavioral bonus signals:

```python
# SBE: +0.1 if agent searches before first edit
sbe_reward = 0.1 if first_search_turn < first_edit_turn else 0.0

# ETT: +0.15 if agent runs tests after editing
ett_reward = 0.15 if has_test_after_last_edit else 0.0
```

### 3. Filter: StdGroupFilter with EMA Threshold

In the training yaml, add under `train_env_manager`:

```yaml
train_env_manager:
  group_filter_mode: "std"
  group_keep_ratio: 0.9
  group_ema_decay: 0.95
  group_min_std_threshold: 1e-6
```

### 4. Training Configuration Changes (yaml)

```yaml
# Key differences from v1 → v8:
exp_name: "train_qwen35_9b_v8"
custom_envs:
  qwen_mcp_swe_rebench:
    env_config:
      enable_instruction_following: true
      instruction_following_model: /path/to/Qwen3.5-27B
      reward_coefficients:
        correctness: 1.0
        instruction_following: 0.1
        search_before_edit: 0.1
        edit_then_test: 0.15
```

For the complete v8 configuration with all parameters, see `examples/qwen35-9b-agentic/train_swe_v8.yaml`.

## Project Structure

```
├── roll/
│   ├── pipeline/
│   │   ├── agentic/           # Agentic RL pipeline (core of this work)
│   │   │   ├── env/           # Environment implementations (SWE, Math, Coding)
│   │   │   ├── env_manager/   # Environment management and orchestration
│   │   │   ├── llm_proxy/     # LLM inference proxies
│   │   │   └── tools/         # Tool implementations (MCP, code execution)
│   │   ├── rlvr/              # RLVR pipeline (from ROLL)
│   │   ├── dpo/               # DPO pipeline (from ROLL)
│   │   ├── distill/           # Distillation pipeline (from ROLL)
│   │   └── sft/               # SFT pipeline (from ROLL)
│   ├── distributed/           # Distributed training infrastructure (from ROLL)
│   ├── configs/               # Configuration system
│   ├── datasets/              # Data loading and sampling
│   ├── models/                # Model providers
│   ├── third_party/           # Backend adapters (vLLM, SGLang, Megatron, DeepSpeed)
│   └── utils/                 # Utilities
├── examples/                  # Training configurations and launch scripts
├── skills/                    # EvoTrainer Skill Library & Analysis Playbook
├── docker/                    # Docker environment setup
├── mcore_adapter/             # Megatron-Core model adapter
└── data/                      # Data templates and samples
```

## Case Study: SWE-9B Iteration

<p align="center">
  <img src="docs/static/swe9b_evolution.png" width="600" alt="SWE-9B Version Evolution"/>
</p>

For details on how EvoTrainer diagnosed bottlenecks, designed interventions, and evolved from v1 (31%) to v8 (38%), see the [paper](https://arxiv.org/abs/2606.03108) §5 or `skills/ANALYSIS_PLAYBOOK.md` and `skills/domain/swe/training_9b.md` in this repository.

## Cross-Domain Version Trajectories

<p align="center">
  <img src="docs/static/trajectories_paper.png" width="700" alt="Cross-Domain Version Trajectories"/>
</p>

EvoTrainer adapts to domain-specific bottlenecks rather than applying a single universal recipe:

- **SWE** evolves toward behavior-sensitive reward design (SBE, ETT) and an instruction-following judge to rescue dead groups.
- **Math** evolves toward computation-aware tool augmentation (Code Interpreter) after reward-side improvements plateau.
- **Coding** evolves toward execution-aligned reward shaping (shaped CR) and cross-domain skill reuse (StdGroupFilter transferred from SWE).

*The retained strategies diverge substantially across domains, reflecting different bottleneck structures rather than a single fixed RL template.*

## Citation

```bibtex
@article{chen2026evotrainer,
  title={EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning},
  author={Chen, Guhong and Shi, Yingcheng and Li, Yongbin and Li, Binhua and Xu, Xander and Wei, Hu and Ni, Shiwen and Yang, Min and Ye, Jieping},
  journal={arXiv preprint arXiv:2606.03108},
  year={2026}
}
```

## Acknowledgments

This project is built on top of the [ROLL](https://github.com/alibaba/ROLL) framework. We thank the ROLL team for providing the distributed RL training infrastructure. 

> **Note**: Some proprietary prompt templates, tool schemas, and SWE task scaffolding have been redacted in this release. All code and data have been reviewed for compliance with external disclosure policies. For any discrepancies between this repository and the published results, the [paper](https://arxiv.org/abs/2606.03108) should be considered authoritative.

## License

Apache License 2.0
