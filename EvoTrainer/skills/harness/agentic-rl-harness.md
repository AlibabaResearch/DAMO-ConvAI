> **Disclosure Notice**: Some numerical results in this document have been redacted or approximated
> for compliance with the organization's external disclosure policy. The methodology descriptions,
> diagnostic logic, and version evolution narratives remain accurate and complete.
> Exact experimental numbers are available in the published paper.


# Agentic RL Diagnostic Harness — Design Specification

## Overview

This document defines the full diagnostic harness for agentic RL training. The harness has one goal: **at every eval step, automatically generate a training health report covering five diagnostic dimensions plus one sampling dimension, with anomaly alerts and intervention suggestions.**

**Core tool**: `scripts/generate_health_report.py` — the unified diagnostic script that produces the complete report in one invocation.

---

## Data Source

All metrics are extracted from rollout pkl files; no training-code modification is required.

Each pkl file exposes the following signals:


```
Top-level fields:
  final_reward          — final shaped reward (float)
  detailed_reward_info  — dict: correctness_reward, binary_correctness, trajectory_reward
  judge_info            — auxiliary judge info (mostly empty; safe to ignore)
  meta_data             — dict: source, id, native_id, meta
    meta.ut_info        — dict: fail_to_pass, pass_to_pass, pass_to_fail, fail_to_fail (ground-truth test sets)
    meta.source_instance_id — instance ID
  history               — list of turns; each turn contains:
    messages / parsed_messages — full conversation (system + user + assistant + tool)
      assistant.content     — model text output (reasoning trace)
      assistant.tool_calls  — list of {function: {name, arguments(dict)}}
    infer_logprobs      — list[float], per-response-token logprobs
    prompt_ids          — list[int], prompt token ids
    response_ids        — list[int], response token ids
    reward              — float, this turn's mcp_tool_call_success
    penalty             — float (= correctness_reward; carries no extra information)
    info                — dict (currently empty)
    metrics             — dict: mcp_tool_call_success
```

---

## Five Automated Diagnostic Dimensions + One Sampling Dimension

### Dimension 1: Result Metrics

**Purpose**: Is training improving, and on what?

| # | Metric | Data source | Computation |
|---|--------|-------------|-------------|
| 1.1 | BC% | `detailed_reward_info['binary_correctness']` | `sum(bc==1.0) / len(pkls)` |
| 1.2 | CR distribution | `detailed_reward_info['correctness_reward']` | Discrete: bucket by exact value. Continuous: bucket by [0, 0.01), [0.01, 0.3), [0.3, 0.5), [0.5, 0.8), [0.8, 1.0), {1.0} |
| 1.3 | CR flow | Pkl files from two step directories, aligned by `source_instance_id` | Build the CR transition matrix step_A → step_B (used during manual analysis) |
| 1.4 | Dead-data ratio | Group by `source_instance_id`, collect all per-group cr values | `count(std(group_cr) < 0.001) / num_instances` |
| 1.5 | Reward Spread | Same grouping as 1.4 | `mean(std(group_cr) for all groups)` — average within-group variance |

**Alerts**:
- Dead-data ratio > 50% → CRITICAL
- Dead-data ratio > 30% → WARNING
- Reward Spread < 0.05 → WARNING (weak GRPO signal)

### Dimension 2: Behavior Metrics

**Purpose**: What is the model *doing*; is the behavior reasonable?

| # | Metric | Data source | Computation |
|---|--------|-------------|-------------|
| 2.1 | Tool-call frequency | Iterate `history[i]['parsed_messages']`, taking role=assistant `tool_calls[j]['function']['name']` | Count by tool name and divide by total rollouts to get per-rollout frequency. Note: `tool_calls` may be None |
| 2.2 | Tool-failure rate | `history[i]['metrics']['mcp_tool_call_success']` | Per-turn judgment: a turn with at least one tool call and success<1.0 counts as a failure. `fail_turns / total_turns_with_tools`. Note: cannot pinpoint which specific tool failed within a turn |
| 2.3 | Self-test rate | Iterate tool_calls for `run_in_terminal`, check whether `arguments['command']` contains pytest/test keywords | `count(rollouts with a test command) / total_rollouts`. Note: `arguments` is a dict |
| 2.4 | Edit coverage | Check whether the rollout contains any `search_replace` / `create_file` call | `count(rollouts with edits) / total` |
| 2.5 | Solution diversity | Group by instance (8 rollouts per group); for each rollout, take the **set of edited file paths** | `1 - avg(pairwise Jaccard similarity)` — measures whether different rollouts of the same instance edit different files |
| 2.6 | Search-before-edit rate | Check whether `read_file` / `search_codebase` / `grep_code` precedes the first `search_replace` | `count(search-before-edit rollouts) / count(rollouts with edits)` |

**Alerts**:
- Self-test rate < 50% → WARNING
- Edit coverage < 60% → WARNING
- Solution diversity < 15% → WARNING (strategy ossification)

#### Dimension 3: Efficiency Metrics

**Purpose**: Is the model working efficiently, or is it idling / inflating?

| # | Metric | Data source | Computation |
|---|--------|-------------|-------------|
| 3.1 | Avg / median turns | `len(d['history'])` | mean / median / max |
| 3.2 | First-effective-edit turn | Index of the first turn containing search_replace / create_file | `mean(first_edit_turn)`, excluding rollouts with no edits |
| 3.3 | Response-token length | `len(history[i]['response_ids'])` | P50 / P90 / mean distribution |
| 3.4 | Behavior volume by CR group | Combine Dim-1 CR with Dim-2/3 metrics | Per CR group: count / avg_turns / edit% / test% |

**Alerts**:
- Avg turns < 5 → CRITICAL (abandonment behavior)
- Avg turns > 50 → WARNING (Echo Trap)

### Dimension 4: Model State Metrics

**Purpose**: Is the model's "thinking state" healthy?

| # | Metric | Data source | Computation |
|---|--------|-------------|-------------|
| 4.1 | Mean logprob | `history[i]['infer_logprobs']` (list[float]) | Per turn: `mean(infer_logprobs)` (filter 0.0 padding); then take the global mean |
| 4.2 | Logprob trend | Per-step values of 4.1 | Surfaced automatically in the trend summary |

**Alerts**:
- Mean logprob > -0.05 → CRITICAL (exploration collapse)
- Mean logprob > -0.1 → WARNING (over-confident)

### Dimension 5: Test Structure Metrics

**Purpose**: Understand task difficulty and pass patterns from the ground-truth test structure.

| # | Metric | Data source | Computation |
|---|--------|-------------|-------------|
| 5.1 | f2p count distribution | `meta_data['meta']['ut_info']['fail_to_pass']` (list) | Count distribution of `len(fail_to_pass)` |
| 5.2 | Pass rate by f2p count | Combine 5.1 (f2p count) with 1.1 (BC) | Group by f2p=1/2/3/4+...; report BC% per group |

**Alerts**:
- Instances with f2p ≥ 3 keep passing at rate 0 → consider excluding or lowering their rollout budget

### Dimension 6: Trajectory Quality Metrics

**Purpose**: Is the model's reasoning process sound? (Requires sampled inspection.)

This dimension is **not fully automated** — it does targeted text analysis on selected cases.

**Sampling strategy**:
1. CR=0.5 rollouts with low self-test rate (no pytest invocation) → why no testing?
2. CR=0.5 rollouts with turns > 30 → going in circles?
3. Instances regressing from CR=1.0 to CR=0.5 → what wrong lesson did the model pick up?
4. Rollouts where every tool call succeeded yet CR=0.0 → right actions, wrong direction?

**What to look for**:
- Did the model read the test file (to understand the goal)?
- Do the edits map to the issue description?
- Are there repetitive / circular patterns?
- Does the model recover effectively after errors?
- Is the reasoning chain coherent?

---

## Mapping to the Three Invariants

The five harness dimensions cover the three core invariants from `knowledge.md`:

| Invariant | Covering dimensions | Core metrics |
|-----------|--------------------|--------------|
| Exploration space does not collapse | Dim 2 (solution diversity) + Dim 4 (logprob) | 2.5 solution diversity, 4.1 mean-logprob trend |
| Learning signal does not degenerate | Dim 1 (GRPO signal) + Dim 5 (test structure) | 1.4 dead-data ratio, 1.5 reward spread |
| Distribution shift stays bounded | Dim 3 (efficiency) + Dim 2 (behavior change) | 3.1 turns trend, 2.3 self-test-rate change |

---

## Diagnostic Rule Engine

Empirical iteration experience, formalized as automated diagnostic rules:

### Rule 1: Insufficient Learning Signal
```
IF dead-data ratio > 30% AND a single discrete CR value > 50% of the distribution
THEN diagnosis  = inadequate signal granularity
     intervention = improve reward continuity (staircase → continuous)
     reference    = continuous-CR designs in prior versions
```

### Rule 2: Echo Trap / Turn Inflation
```
IF avg turns rise across 3 consecutive eval steps
   OR avg turns of CR=0.0 rollouts > overall × 1.5
THEN diagnosis  = Echo Trap precursor
     intervention = introduce an efficiency factor / context management / max-turns cap
     reference    = previous efficiency-factor and max-turns experiments
```

### Rule 3: Exploration Collapse
```
IF tool-sequence diversity drops > 10pp
   AND mean logprob rises > 50%
THEN diagnosis  = exploration-space collapse
     intervention = raise sampling temperature / add exploration bonus / KL constraint
```

### Rule 4: Reward Hacking
```
IF shaped reward (trajectory_reward) keeps rising
   BUT binary_correctness stays flat or declines
THEN diagnosis  = reward hacking
     intervention = simplify reward (drop process reward) / strengthen KL constraint
     reference    = past rounds where shaped reward decoupled from BC%
```

### Rule 5: Behavioral Degradation (Laziness)
```
IF run_in_terminal per-rollout drops > 30%
   AND self-test rate drops > 15pp
   AND BC% does not improve materially
THEN diagnosis  = the model has learned to "do less" rather than "do better"
     intervention = inject behavior-quality signal into the reward (self-test bonus) / restructure reward
```

### Rule 6: Poor Edit Quality
```
IF search_replace failure rate > 15%
THEN diagnosis  = inadequate edit-matching capability
     intervention = strengthen edit-format training in the SFT stage / reinforce edit conventions in the prompt
```

---

## Usage

### Unified Diagnostic Script

`scripts/generate_health_report.py` covers all metrics in dimensions 1–5:

```bash
# Full report for a single step
python3 scripts/generate_health_report.py \
    --exp_dir <EXP_ROOT>/<EXP_NAME>/rollouts/qwen_mcp_swe_rebench \
    --step 100 --split val

# Multi-step trend report
python3 scripts/generate_health_report.py \
    --exp_dir <DIR> \
    --steps 0,25,50,75,100 --split val

# Auto-detect all available steps
python3 scripts/generate_health_report.py \
    --exp_dir <DIR> --split val

# Emit JSON (convenient for programmatic comparison)
python3 scripts/generate_health_report.py \
    --exp_dir <DIR> --step 100 --json_out metrics.json
```

---

## Relation to Existing Tools

| Script | Role | Status |
|--------|------|--------|
| `scripts/generate_health_report.py` | **Main entry point**, automates dimensions 1–5 | **Implemented** |
| `scripts/analyze_val_results.py` | Early dimension-1 partial analysis | Superseded by health report |
| `scripts/analyze_cr_distribution.py` | Early GRPO signal analysis | Superseded by health report |
| `scripts/analyze_behavior.py` | Early behavior-pattern analysis | Superseded by health report |
| `scripts/inspect_trajectory.py` | Sampling tool for dimension 6 | Planned |

---

## Harness Evolution Protocol

The harness is not built once and frozen — it must evolve along with training versions. This is what distinguishes EvoTrainer from a "fixed metrics + fixed scripts" pipeline: a **meta-level iteration loop**.

### When to evolve the harness

Trigger an evolution when any of the following holds:

- A failure mode is observed that **no existing metric can capture**
- A successful intervention surfaces a new diagnostic dimension worth tracking
- A cross-domain transfer suggests a metric from domain A may apply to domain B

### How to add a new diagnostic metric

1. **Identify the gap**: which behavioral signal are we missing?
2. **Define the metric**: name, formula, data source (pkl fields / TB scalars)
3. **Set thresholds**: what counts as healthy vs. unhealthy?
4. **Implement extraction**: add to `scripts/{domain}/analyze_val.py` or create a new script
5. **Retrospective validation**: replay the metric across previous versions and confirm it discriminates successful vs. failed runs
6. **Document**: register the metric under the appropriate dimension in this file

### Worked example: the evolution of DGR

**Context**: In early SWE versions, training appeared active (non-zero pg_loss) yet BC% barely improved. Existing metrics (entropy, reward mean) showed no anomaly.

**Gap identified**: no metric captured "how many GRPO groups produce zero gradient signal".

**New metric**: Dead Group Ratio (DGR) = count(groups where std(reward)==0) / total_groups.

**Retrospective validation**: replaying DGR on early-version data revealed >60% dead groups — explaining "why loss was active but training was ineffective".

**Triggered intervention**: this metric directly motivated the IF LLM Judge variant — scoring instruction-following to inject within-group variance. DGR dropped from ~60% to ~28%.

**Distilled skill**: "When DGR > 50% and the reward is discrete, introduce an auxiliary continuous signal (e.g. an LLM judge) to break group homogeneity."

### Escalation protocol after consecutive failures

When the agent encounters **three or more consecutive failed versions** (each scoring below the current best):

1. **Pause and diagnose the diagnosis itself**: is the harness measuring correctly? Re-validate the eval pipeline (data coverage, metric computation, environment determinism).
2. **Broaden the hypothesis space**: if every previous hypothesis targeted the same layer (e.g. all reward changes), explicitly switch layer (signal / behavior / infrastructure).
3. **Consult external literature**: search published methods for the observed failure pattern; cross-reference the skill library for analogous solved cases in other domains.
4. **Simplify, do not complexify**: the next intervention after three failures should be **subtraction** (remove a component) rather than **addition** (more complexity).
5. **Roll back to the last known-good version**: if the baseline has drifted, explicitly reset to the last version that improved and re-branch from there.

### Cross-domain harness transfer

Different domains evolve different harness components:

| Domain | Characteristic metrics | Unique challenges |
|--------|------------------------|-------------------|
| **SWE** | BC%, edit_ratio, test_ratio, n_turns, tool_call_success | Long trajectories (30–50 turns), binary outcome, Docker timeouts |
| **Math** | solve_rate, format_compliance, length_drift, KL_divergence | Length explosion, policy drift from extended CoT |
| **Coding** | pass_rate, format_gate_violation, truncation_rate | Format-gate sensitivity, code-block detection |

Transfer protocol:

- When a mechanism (e.g. StdGroupFilter) succeeds in one domain, evaluate its applicability to others
- Adaptation may be required (e.g. SWE uses an LLM Judge for IF; Math/Coding may not need it)
- Record the transfer decision and any domain-specific modification in the version log

---

## Script Evolution History (an instance of harness evolution)

EvoTrainer's diagnostic scripts were not built in one shot — they went through several "scattered → consolidated" evolutions, instantiating the *analyzer specialization* mechanism described in §3.3 of the paper.

### Stage 1: Scattered scripts (single-dimension analysis)

Early on, every analytical need had its own standalone script:

- `analyze_val_results.py` — focused on the **result dimension** (BC%, CR distribution)
- `analyze_cr_distribution.py` — focused on the **GRPO signal** (dead groups, CR flow)
- `analyze_behavior.py` — focused on **behavior patterns** (edit/test/search ratios)

Output formats were not unified, so cross-version comparison required manual stitching.

### Stage 2: Diagnostic gaps surface

As versions advanced, failure modes appeared that single-dimension scripts could not capture:

- The same group of rollouts needed joint diagnosis across "behavior + GRPO signal + result"
- Cross-step trend comparison required a unified time axis
- Cross-dimension threshold rules were needed for anomaly alerts (e.g. dead-group ratio > 50%)

This was the harness's first explicit "diagnostic gap" — the per-dimension scripts no longer sufficed.

### Stage 3: Unified diagnosis (`generate_health_report.py`)

The five automated dimensions (plus the sampling dimension) were rewritten as a single diagnostic script:

- **Unified input**: a single `--exp_dir` argument; the script auto-discovers every step
- **Unified output**: one "health report" per step, covering every dimension
- **Built-in alerting**: CRITICAL / WARNING thresholds aligned with the rule engine
- **Trend visualization**: single-step report + multi-step trend report + JSON output (for programmatic comparison)

The three earlier scripts are no longer invoked, but remain in the repo as "evolution records".

### Evolution pattern

This path — **multiple single-dimension scripts → diagnostic gap surfaces → consolidation into a unified report** — is the canonical pattern for EvoTrainer harness evolution:

1. When a new diagnostic dimension is added, validate it quickly with a standalone script
2. When several dimensions need joint diagnosis or cross-version comparison, fold them into the main report
3. The main report itself expands as new dimensions are added (e.g. the dimension-6 sampling tool `inspect_trajectory.py` is queued for integration)

This "validate-independently, then consolidate" pattern guarantees that every step of harness evolution leaves a reusable intermediate artifact, avoiding the risk of a one-shot large refactor.

---

## Two-Level Filtering Mechanism

A two-stage filtering scheme that addresses dead data and low-quality trajectories.

### Level 1: per-trajectory filtering (returns -999 inside `env.py`)

Inside `obtain_outcome_reward`, the following trajectories are filtered out:
- **Abandonment**: turns ≤ 3 with no search_replace/create_file
- **Idle**: no edit tool was used and correctness = 0
- **Echo Trap**: turns > 60 with correctness = 0
- **Already filtered**: response hit max_new_tokens, trajectory truncated, evaluation failed

How the -999 sentinel is handled:
1. `env.py` returns -999
2. `mcp_swe_env_manager` rewrites it to 0.0
3. In `get_agentic_sample_level_mask`, scores outside [-5, 5] cause the entire response_mask of that sample to be zeroed

### Level 2: group-variance filtering (built-in `std_filter`)

**No code change required** — just add to the yaml config:

```yaml
# Add inside the reward config or custom_envs
query_filter_config:
  type: std_filter
  filter_args:
    std_threshold: 0.01  # tunable
```

Implementation: `roll/distributed/scheduler/user_defined_rollout_loop.py` L47-50

### How the two levels work together

1. First, filter degenerate individual trajectories (level 1, in env.py)
2. Then, if the surviving rollouts in a group still have reward variance < threshold, drop the entire group (level 2, std_filter)
3. This is more principled than running group-level filtering alone: removing the "drag-along" trajectories first can actually increase within-group variance

---

## Reward-Dimension Reference

A reference set of reward signals, useful as inspiration for future reward shaping:

```
custom_ut_exists_reward       — whether a custom unit test was run
custom_ut_pass_reward         — whether the custom unit test passed
search_before_action_reward   — whether the model searched before acting
info_gather_before_edit_reward — whether information was gathered before editing
parallel_tool_reward          — degree of tool-call parallelism
fix_match_reward              — whether the fix matches the target
fail_to_fail_reward           — fail_to_fail test outcome
task_manager_reward           — task-management ability
instruction_following_reward  — instruction-following degree
search_ratio_reward           — search ratio
file_edit_frequency_reward    — file-edit frequency
delete_then_create_reward     — detection of delete-then-create patterns
hybrid_dep_score / hybrid_diversity / hybrid_efficiency — hybrid measures
```
