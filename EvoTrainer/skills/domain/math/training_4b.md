> **Disclosure Notice**: Some numerical results in this document have been redacted or approximated
> for compliance with the organization's external disclosure policy. The methodology descriptions,
> diagnostic logic, and version evolution narratives remain accurate and complete.
> Exact experimental numbers are available in the published paper.


# Qwen3.5-4B Math RL Training — AI Iteration Guide

## Project Goal

Train Qwen3.5-4B with RL on the math domain so the model better solves competition-level problems (AIME 2024 / 2025 + CNMO 2024).

**Primary metric**: validation `correctness_reward` (LLM-judge accuracy; higher is better).
**Optimization focus**: reward design + training stability (prevent length drift from degrading training).

---

## 1. AI Action Boundary

### Allowed
- Modify `examples/qwen35-4b-agentic/train_math_v{N}.yaml` and `.sh` (new versions)
- Modify `roll/pipeline/agentic/env/math_laj/env.py` (reward core logic)
- Modify `roll/pipeline/agentic/env/math_laj/config.py` (config fields)
- Create new worktrees to run new-version experiments
- Run analysis scripts, inspect logs and results

### Not allowed
- Modify framework core code under `roll/distributed/`, `roll/configs/`, `roll/models/`, etc.
- Modify other users' directories under `examples/`
- Delete or modify historical version output directories
- Modify the validation data path (`eval_data_path` must stay identical across versions)

### Safety principles
- All changes must be reversible via `git checkout` or `git reset`
- Before modifying reward code, confirm existing logic is not broken (gate new behavior behind a yaml flag)
- When uncertain, surface a `git diff` to the user for confirmation

---

## 2. Directory Layout

### Code

```
examples/qwen35-4b-agentic/
├── train_math_v1.yaml          # v1 config (baseline)
├── train_math_v1.sh            # v1 launch script
├── train_math_v{N}.yaml        # per-version configs
└── train_math_v{N}.sh          # per-version launch scripts

roll/pipeline/agentic/env/math_laj/
├── env.py                      # reward core logic (obtain_outcome_reward)
└── config.py                   # config dataclass (incl. use_length_penalty switch)
```

### Experiment outputs (one tree per version)

```
<EXP_ROOT>/evotrainer_math35_4b_v{N}/
├── logs/                         # training logs
├── models/                       # checkpoints
├── rollouts/math/
│   ├── train_step_{step}.jsonl   # train rollouts
│   └── val_step_{step}.pkl       # val rollouts
├── tensorboard/
└── render/
```

### Data (do not modify)

| Path | Description |
|------|-------------|
| `<DATA_DIR>/math_grpo_hard_v1.jsonl` | Training data |
| `<DATA_DIR>/aime2024_aime2025_cnmo2024.jsonl` | Validation data (78 problems) |
| `<MODEL_DIR>/Qwen3.5-4B/` | Pretrained model |

---

## 3. Launch Command

```bash
cd <PROJECT_ROOT>
bash train_math_v{N}.sh
```
