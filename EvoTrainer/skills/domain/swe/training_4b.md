> **Disclosure Notice**: Some numerical results in this document have been redacted or approximated
> for compliance with the organization's external disclosure policy. The methodology descriptions,
> diagnostic logic, and version evolution narratives remain accurate and complete.
> Exact experimental numbers are available in the published paper.


# Qwen3.5-4B Agentic RL Training — AI Iteration Guide

## Project Goal

Train Qwen3.5-4B with agentic RL so the model becomes better at fixing software issues (SWE-rebench dataset).

**Primary metric**: validation `correctness_reward` (fail-to-pass test pass rate; higher is better).
**Optimization focus**: reward design (NOT hyperparameter tuning — the existing hyperparameters are already well-tuned).

---

## 1. AI Action Boundary

### Allowed
- Modify `examples/qwen35-4b-agentic/train_swe_v{N}.yaml` and `.sh` (new versions)
- Modify reward code under `roll/pipeline/agentic/reward_manager/`
- Modify reward-related code under `roll/pipeline/agentic/`
- Modify prompt templates under `data/agent_templates/qwen_mcp_swe_rebench/`
- Create new reward workers or reward-helper modules
- Run analysis scripts, inspect logs and results

### Not allowed
- Modify framework core code under `roll/distributed/`, `roll/configs/`, `roll/models/`, etc.
- Modify other users' directories under `examples/`
- Change multiple unrelated things at once (each version focuses on one hypothesis)
- Delete or modify historical version output directories
- Modify the validation data path (`eval_data_path` must stay identical across versions)

### Safety principles
- All changes must be reversible via `git checkout` or `git reset`
- Before modifying reward code, confirm existing logic is not broken (add new branches gated by yaml flags)
- When uncertain, surface a `git diff` to the user for confirmation

---

## 2. Directory Layout

### Code

```
examples/qwen35-4b-agentic/
├── train_swe_v1.yaml          # v1 config (baseline)
├── train_swe_v1.sh            # v1 launch script
├── train_swe_v{N}.yaml        # per-version configs
└── train_swe_v{N}.sh          # per-version launch scripts

roll/pipeline/agentic/
├── agentic_config.py            # config definitions (incl. RewardNormalizationConfig)
├── agentic_pipeline.py          # training pipeline main logic
├── env/qwen_mcp_swe/env.py      # reward core logic (_evaluate_correctness_rebench / obtain_outcome_reward)
├── reward_manager/
│   └── fail_to_pass.py          # pytest output parser
├── utils.py                     # reward normalization (agentic_reward_norm)
└── env_manager/                 # environment management

data/agent_templates/qwen_mcp_swe_rebench/    # Agent prompt templates
```

### Experiment outputs (one tree per version)

```
<EXP_ROOT>/evotrainer_qwen35_4b_v{N}/
├── logs/                         # training logs
├── models/                       # checkpoints
├── rollouts/qwen_mcp_swe_rebench/
│   ├── train/                    # train rollouts
│   └── val/                      # val rollouts (one subdir per step)
├── tensorboard/
└── experiment_info.txt
```

### Data and models (do not modify)

| Path | Description |
|------|-------------|
| `<DATA_DIR>/swe-rebench-with-catalog-v6-train.sorted.jsonl` | Training data |
| `<DATA_DIR>/swe-rebench-with-catalog-v6-test.repeat_8.jsonl` | Validation data (77 problems × 8 rollouts = 616 records) |
| `<MODEL_DIR>/Qwen3.5-4B/` | Pretrained model |

---

## 3. Launch Command

```bash
cd <PROJECT_ROOT>
bash train_swe_v{N}.sh
```
