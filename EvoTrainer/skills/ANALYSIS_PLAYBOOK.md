# EvoTrainer Analysis Playbook

> **Disclosure Notice**: Some numerical thresholds and domain-specific examples in this document
> have been approximated for compliance with the organization's external disclosure policy.
> The methodology, diagnostic logic, and workflow structure remain accurate and complete.

---

## 1. Overview

This playbook defines the **standardized diagnostic and intervention workflow** that the AI trainer agent autonomously executes within EvoTrainer. The agent follows this loop for each version transition, generating analysis scripts, forming hypotheses, and designing interventions without human intervention:

```
Train → Diagnose → Hypothesize → Design Intervention → Backtest → Execute → Verify → Accumulate Skill
```

The core principle: **every version transition must be evidence-driven, single-variable, and falsifiable.**

---

## 2. The Iteration Loop (9 Steps)

### Step 1: Collect Rollout Evidence
- Run validation after each `eval_steps` interval
- Extract per-instance rollout data from val pkl files
- The AI agent generates and runs analysis scripts following the conventions in §9

### Step 2: Score Layer Analysis
- Compute: Avg@8 (primary metric), shaped CR mean, final_reward mean
- Compare against previous version's best step
- Identify: improvement / stagnation / regression

### Step 3: Signal Layer Analysis
- Dead Group Ratio (DGR): % of GRPO groups with zero variance
- Reward distribution: is there meaningful spread within groups?
- Gradient health: are advantages non-degenerate?
- The AI agent extracts TB metrics using the patterns defined in §9.6

### Step 4: Behavior Layer Analysis
- Domain-specific behavioral metrics (see §4 for per-domain details)
- Identify behavioral drift: is the model learning the wrong shortcut?
- Check for reward hacking patterns

### Step 5: Hypothesis Formation
- Based on Steps 2-4, form a **single testable hypothesis**
- Format: "If we change X, then metric Y should improve because Z"
- The hypothesis must be **falsifiable** with a clear success criterion

### Step 6: Intervention Design
- **Single variable only** (|ΔX| = 1)
- Document: what changes, what stays frozen, what the success condition is
- If multiple changes seem needed, prioritize and sequence them across versions

### Step 7: Backtest (when applicable)
- Simulate the proposed intervention on historical rollout data
- Verify the hypothesis directionally before committing GPU hours
- Tool: reward backtesting on cached pkl files

### Step 8: Execute Training
- Launch new version with the single-variable intervention
- Monitor TensorBoard for early anomalies (entropy collapse, KL explosion, etc.)

### Step 9: Verify & Record
- Run full val analysis on the new version
- Compare against hypothesis success criterion
- Record outcome in version history (success / failure / partial)
- If successful: accumulate the insight as a reusable Skill

---

## 3. Harness Evolution Protocol

The harness itself evolves across versions through a meta-level loop. The full protocol — when to evolve, how to add new diagnostic metrics, the escalation strategy for consecutive failures, and cross-domain transfer — is defined in [`skills/harness/agentic-rl-harness.md`](./harness/agentic-rl-harness.md) under the **Harness Evolution Protocol** section.

The iteration loop (§2) hands off to this protocol at two key points:

- **Step 4 (Behavior Layer Analysis)** — when no existing metric captures the observed failure mode, consult “How to Add a New Diagnostic Metric”
- **Step 9 (Verify & Record)** — newly validated metrics are documented under the appropriate harness dimension

---

## 4. Domain-Specific Diagnostic Dimensions

### 4.1 SWE (Software Engineering)

**Five diagnostic dimensions + one sampling dimension:**

| Dimension | Key Metrics | Alert Condition |
|-----------|------------|-----------------|
| 1. Reward Signal | BC%, CR distribution, DGR | DGR > 50% or BC% declining |
| 2. Trajectory Health | n_turns, truncation_rate, tool_call_success | Trunc > 20% or turns exploding |
| 3. Behavioral Pattern | edit_ratio, test_ratio, search_ratio | edit_ratio < 15% (action timidity) |
| 4. Training Stability | entropy, KL, pg_loss, grad_norm | Entropy collapse or KL spike |
| 5. Filter Effectiveness | group_keep_ratio, EMA threshold trend | Keep < 30% (over-filtering) |
| 6. Rollout Sampling | Random sample 5-10 rollouts per category | Manual inspection of failure modes |

### 4.2 Math (Mathematical Reasoning)

| Dimension | Key Metrics | Alert Condition |
|-----------|------------|-----------------|
| 1. Solve Rate | Avg@8 per subset (AIME/CNMO), format_reward | Score declining from pretrained |
| 2. Length Dynamics | response_length trend, length vs. correctness correlation | corr(len, reward) > 0.3 (length gaming) |
| 3. KL Health | token-level KL, init_kl_coef effectiveness | KL ineffective (add_token_level_kl bug) |
| 4. Training Stability | entropy, pg_loss convergence | Loss = 0 (dead training) |

### 4.3 Coding (Competitive Programming)

| Dimension | Key Metrics | Alert Condition |
|-----------|------------|-----------------|
| 1. Pass Rate | Avg@8, per-difficulty breakdown | Score below pretrained |
| 2. Format Gate | format_violation_rate, FmtFail root cause | FmtFail > 30% (gate misconfigured) |
| 3. Truncation | truncation_rate, overlong_ratio | Trunc > 15% (budget misaligned) |
| 4. Training Stability | entropy, DGR, reward spread | DGR > 60% (dead signal) |

---

## 5. Skill Accumulation Protocol

When a version succeeds, the insight is formalized as a **reusable Skill**:

### 5.1 Skill Format

```markdown
# Skill: [Descriptive Name]

## Trigger Condition
When should this skill be applied? (e.g., "DGR > 50% and reward is discrete")

## Intervention
What to change (specific code/config modification)

## Expected Outcome
What metric should improve, by approximately how much

## Validation
How to verify the skill worked (success criterion)

## Domain Applicability
Which domains has this been validated on? Transfer notes.
```

### 5.2 Skill Categories

1. **Reward Design**: Shaping reward signals (e.g., tiered CR, continuous CR, IF Judge)
2. **Filter Mechanism**: Trajectory/group filtering strategies (e.g., StdGroupFilter, EMA Top-p)
3. **Training Stability**: Preventing collapse (e.g., KL configuration, Clip-Higher)
4. **Evaluation Protocol**: Measurement corrections (e.g., format gate fixes, val coverage)
5. **Cross-Domain Transfer**: Mechanisms validated in one domain, adapted for another

### 5.3 Skill Storage

Skills are accumulated in their **owning domain’s md** by default — there is no separate `skill-library/` directory.

| Skill scope | Destination |
|---|---|
| **Domain-specific lesson** (default) | `skills/domain/{domain}/training_*B.md` — under “Known Experience” (create the section if absent) |
| **Cross-domain diagnostic/intervention rule** (only after validation on ≥2 domains) | `skills/harness/agentic-rl-harness.md` — Diagnostic Rule Engine or as a new dimension |

A skill always starts in its owning domain. Promotion to `harness.md` only happens after the rule has been confirmed to generalize — typically via the fallback transfer flow described in §5.4.

### 5.4 Cross-Domain Knowledge Transfer (fallback only)

The trainer agent should always **prioritize domain-internal iteration**. Cross-domain transfer is a fallback — *not* a default lookup:

1. **First**: solve the current bottleneck using the domain’s own prior experience (its `training_*B.md`, harness diagnosis, version history).
2. **Only after multiple internal attempts fail**: browse other domains’ `skills/domain/*/training_*B.md` for analogous patterns — similar diagnostic signatures (e.g., DGR > 50%) or mechanisms that solved related symptoms (e.g., StdGroupFilter for low-information groups).
3. **Document the transfer**: if an external-domain skill is adopted, note the adaptation in the current domain’s md. If it generalizes successfully, promote the rule to `harness.md` so future domains benefit automatically.

**Why fallback only**: cross-domain transfer carries mis-application risk — a SWE-specific reward shape may not fit Math. Forcing the agent to exhaust domain-internal options first keeps the experimentation loop low-noise.

---

## 6. Anti-Patterns (Learned from Failed Versions)

| Anti-Pattern | Symptom | Root Cause | Prevention |
|-------------|---------|-----------|-----------|
| Reward Hacking | Score up but behavior degrades (observed: 1-turn exits) | Efficiency factor exploitable by minimal-action strategy | Never reward brevity directly; use behavioral floor constraints |
| Exploration Collapse | Avg@8 drops to 0% within ~3 steps | Data over-filtering combined with aggressive KL (>0.05) | Cap KL at 0.002; monitor entropy every eval_step; auto-halt if entropy < threshold |
| Action Timidity | edit_ratio declining steadily | Penalty for wrong edits too strong | Balance reward for action vs. inaction |
| Dead Training | pg_loss = 0, ratio = 1.0, score frozen | Discrete reward causes >50% groups to have identical outcomes | Introduce continuous reward components or external judge to inject within-group variance |
| Length Gaming | Response length grows ~3x without score improvement | Positive correlation between length and reward is not bounded | Apply Clip-Higher (cap reward at solved-length) or DAPO-style overlong shaping |

---

## 7. Version Transition Decision Framework

Before creating v(N+1), answer these questions:

1. **What is the single largest bottleneck in vN?** (from Step 2-4 analysis)
2. **Is there a testable hypothesis?** (from Step 5)
3. **Can it be isolated as a single variable?** (|delta X| = 1)
4. **Is backtesting possible?** (from Step 7)
5. **What is the falsifiable success criterion?**
6. **What is the rollback plan if it fails?**

If any answer is "no" or "unclear", do more diagnosis before proceeding.


---

## 7.1 Human-in-the-Loop Configuration

EvoTrainer supports two execution modes:

| Mode | Description | When to use |
|------|-------------|-------------|
| **Fully Autonomous** | Agent executes the entire 9-step loop without pausing for confirmation. | Established domains with validated harness; low-risk incremental iterations. |
| **Human-Gated** | Agent pauses at designated checkpoints and presents its analysis/plan to the human operator for approval before proceeding. | New domains, high-cost training runs, or when exploring unfamiliar intervention types. |

### Configurable Checkpoints

The following steps can optionally require human confirmation (set via `human_gate: true` in the iteration config):

| Checkpoint | What the agent presents | Human decides |
|-----------|------------------------|---------------|
| **After Step 4** (Diagnosis complete) | Full diagnostic report: score/signal/behavior summary, identified bottleneck, candidate hypotheses | Whether the diagnosis is correct; which hypothesis to pursue |
| **After Step 6** (Intervention designed) | Proposed single-variable change, success criterion, rollback plan | Whether to approve the intervention or request modifications |
| **After Step 7** (Backtest results) | Backtest outcome on historical data, projected impact | Whether to proceed to full training or revise the design |
| **After Step 9** (Version complete) | Training results, comparison to success criterion, proposed skill to accumulate | Whether to accept the version, reject it, or request additional analysis |

### Default Recommendation

- **First 3 versions of a new domain**: Human-Gated (to validate the harness itself is correct)
- **After harness is validated**: Fully Autonomous (the agent has proven its diagnostic accuracy)
- **After 2 consecutive failures**: Automatically escalate to Human-Gated regardless of configuration

### Implementation

In the training configuration yaml, set:

```yaml
evotrainer:
  autonomy_level: "full"          # Options: "full", "human_gated"
  gate_after_diagnosis: false     # Pause after Step 4
  gate_after_design: true         # Pause after Step 6 (recommended)
  gate_after_backtest: false      # Pause after Step 7
  gate_after_completion: true     # Pause after Step 9 (recommended)
  auto_escalate_after_failures: 2 # Switch to human_gated after N failures
```

When a gate is triggered, the agent outputs a structured report and waits for one of:
- `APPROVE` — proceed to next step
- `REVISE: <feedback>` — agent incorporates feedback and re-runs the gated step
- `ABORT` — halt iteration, rollback to previous version

---

## 8. Changelog

| Version | Changes |
|---------|---------|
| v1.0 | Initial 9-step loop |
| v1.1 | Added four-layer framework (Score/Signal/Behavior/Version) |
| v1.2 | Added cross-domain harness evolution protocol |
| v2.0 | Consolidated cross-domain harness evolution; added script generation guide |

---

## 9. Analysis Script Generation Guide

EvoTrainer does not ship pre-built analysis scripts. Instead, the AI trainer agent **generates domain-appropriate analysis scripts on demand** based on the current diagnostic needs. This is by design: each version may require different analysis dimensions.

### 9.1 Script Naming Convention

```
scripts/{domain}/analyze_v{N}_{focus}.py
```

Examples:
- `scripts/swe/analyze_v4_reward.py` — Reward distribution analysis for SWE v4
- `scripts/math/analyze_v3_length.py` — Length drift analysis for Math v3
- `scripts/coding/analyze_v6_format.py` — Format gate violation analysis for Coding v6

### 9.2 Standard Script Template

Every analysis script should follow this structure:

```python
"""
Version: v{N}
Domain: {swe|math|coding}
Focus: {what this script analyzes}
Data source: {pkl files from rollouts/{env_name}/val/}
"""
import os
import json
import pickle
import numpy as np
from collections import defaultdict

# === Configuration ===
EXP_DIR = os.environ.get("EXP_DIR", "/path/to/experiments/your_exp_name")
VAL_DIR = os.path.join(EXP_DIR, "rollouts/{env_name}/val")

def get_available_steps(val_dir):
    """List all evaluation steps that have pkl files."""
    steps = set()
    for f in os.listdir(val_dir):
        if f.endswith(".pkl"):
            # Format: {env_name}_{group}_{idx}_{date}.pkl
            # or step-based naming
            steps.add(extract_step(f))
    return sorted(steps)

def load_step_data(val_dir, step):
    """Load all rollout results for a given step."""
    results = []
    for f in os.listdir(val_dir):
        if f.endswith(".pkl") and matches_step(f, step):
            with open(os.path.join(val_dir, f), "rb") as fr:
                data = pickle.load(fr)
                results.append(data)
    return results

def extract_metrics(results):
    """Extract domain-specific metrics from rollout data."""
    metrics = {
        "n_total": len(results),
        "binary_correctness": [],
        "final_reward": [],
        # Add domain-specific fields here
    }
    for data in results:
        detailed = data.get("detailed_reward_info", {})
        if isinstance(detailed, str):
            detailed = json.loads(detailed)
        metrics["binary_correctness"].append(
            detailed.get("binary_correctness", 0.0)
        )
        metrics["final_reward"].append(data.get("final_reward", 0.0))
    return metrics

def print_summary(metrics, step):
    """Print human-readable summary."""
    n = metrics["n_total"]
    bc = np.mean(metrics["binary_correctness"])
    fr = np.mean(metrics["final_reward"])
    print(f"Step {step}: n={n}, Avg@8={bc*100:.2f}%, final_reward={fr:.4f}")

# === Main ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default=EXP_DIR)
    parser.add_argument("--mode", choices=["summary", "detailed", "compare"], default="summary")
    args = parser.parse_args()

    val_dir = os.path.join(args.exp_dir, "rollouts/{env_name}/val")
    steps = get_available_steps(val_dir)

    for step in steps:
        results = load_step_data(val_dir, step)
        metrics = extract_metrics(results)
        print_summary(metrics, step)
```

### 9.3 Analysis Modes

Each script should support at minimum these modes:

| Mode | Purpose | Output |
|------|---------|--------|
| `summary` | Per-step Avg@8 and key metrics | One line per step |
| `detailed` | Per-instance breakdown | Full table with instance-level data |
| `compare` | Cross-version comparison | Side-by-side metrics for vN vs vN-1 |
| `failure_analysis` | Deep dive into failing instances | Grouped by failure category |

### 9.4 When to Generate a New Script

Generate a new analysis script when:
- A new version completes training and needs diagnostic evaluation
- A hypothesis requires a metric not covered by existing scripts
- Cross-version comparison is needed to validate an intervention
- A new diagnostic dimension is added to the harness

### 9.5 Data Source Reference

| Domain | Val directory structure | Key pkl fields |
|--------|----------------------|----------------|
| SWE | `rollouts/qwen_mcp_swe_rebench/val/*.pkl` | `final_reward`, `detailed_reward_info.binary_correctness`, `detailed_reward_info.correctness_reward`, `history[].parsed_messages` |
| Math | `rollouts/math/val/*.pkl` | `final_reward`, `detailed_reward_info.correctness_reward`, `history[].response_ids` |
| Coding | `rollouts/coding/val/*.pkl` | `final_reward`, `detailed_reward_info.correctness_reward`, `detailed_reward_info.format_reward` |

### 9.6 TensorBoard Metric Extraction

For training-time metrics (entropy, KL, pg_loss), use TensorBoard event files:

```python
from tensorboard.backend.event_processing import event_accumulator

def extract_tb_scalar(logdir, tag):
    """Extract a scalar time series from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]
```

Common tags:
- `actor/entropy` — Policy entropy (monitor for collapse)
- `actor/pg_loss` — Policy gradient loss
- `actor/approxkl` — Approximate KL divergence
- `val/env/{env_name}/score/mean` — Validation score
