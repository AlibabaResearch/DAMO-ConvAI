> **Disclosure Notice**: Some numerical results in this document have been redacted or approximated
> for compliance with the organization's external disclosure policy. The methodology descriptions,
> diagnostic logic, and version evolution narratives remain accurate and complete.
> Exact experimental numbers are available in the published paper.


# Qwen3.5-9B Agentic RL Training — AI Iteration Guide

## Project Goal

Train Qwen3.5-9B with agentic RL so the model becomes better at fixing software issues on the SWE-rebench dataset.

**Primary metric**: validation `binary_correctness` (fail-to-pass test pass rate).
**Optimization strategy**: build on the mature reward structure accumulated from 4B iterations, replicate the best configuration on 9B, and observe scale effects.
**Reference docs**: `skills/domain/swe/training_4b.md` (full 4B iteration experience), `skills/harness/agentic-rl-harness.md` (six-dimensional harness diagnostic framework).

---

## 1. AI Action Boundary

### Allowed
- Modify `examples/qwen35-9b-agentic/train_swe_v{N}.yaml` and `.sh` (new versions)
- Modify reward logic in `roll/pipeline/agentic/env/qwen_mcp_swe/env.py`
- Modify dataclass fields in `roll/pipeline/agentic/env/qwen_mcp_swe/config.py`
- Modify filter logic in `roll/pipeline/agentic/env/qwen_mcp_swe/group_filter.py`
- Run analysis scripts, inspect logs and results

### Not allowed
- Modify framework core code under `roll/distributed/`, `roll/configs/`, `roll/models/`, etc.
- Modify other users' directories under `examples/`
- Change multiple unrelated things at once (each version focuses on one hypothesis)
- Modify the validation data path (`eval_data_path` must stay identical across versions)
- Modify the 9B hardware / parallelism parameters (these are pre-tuned and out of optimization scope)

---

## 2. Directory Layout

### Code

```
examples/qwen35-9b-agentic/
├── train_swe_v1.yaml          # v1 config (baseline)
├── train_swe_v1.sh            # v1 launch script
├── train_swe_v{N}.yaml        # per-version configs
└── train_swe_v{N}.sh          # per-version launch scripts

roll/pipeline/agentic/
├── env/qwen_mcp_swe/env.py    # **reward core logic**
├── env/qwen_mcp_swe/config.py # environment config dataclass
├── agentic_pipeline.py          # GroupFilter / StdGroupFilter
└── reward_manager/fail_to_pass.py     # pytest output parser

scripts/generate_health_report.py      # (not included in open-source release)
scripts/backtest_reward.py             # (not included in open-source release)
scripts/backtest_std_threshold.py      # (not included in open-source release)
```

### Experiment outputs (one tree per version)

```
<EXP_ROOT>/evotrainer_qwen35_9b_v{N}/
├── logs/                         # training logs
├── models/                       # checkpoints
├── rollouts/qwen_mcp_swe_rebench/
│   ├── train/                    # train rollouts
│   └── val/                      # val rollouts (one subdir per step)
├── tensorboard/
└── mcps/
```

### Data and models (do not modify)

| Path | Description |
|------|-------------|
| `<DATA_DIR>/swe-rebench-with-catalog-v7-train.9b-sorted.no-easy-update.jsonl` | Training data (later versions) |
| `<DATA_DIR>/swe-rebench-with-catalog-v6-test.repeat_8.jsonl` | Validation data (616 records) |
| `<DATA_DIR>/swe_issue_fix_template.jsonl` | User template |
| `<MODEL_DIR>/Qwen3.5-9B/` | Pretrained model |
| `<MODEL_DIR>/Qwen3.5-27B/` | Reward model |

---

## 3. Naming Convention (strictly enforced)

| Item | Format | Example |
|------|--------|---------|
| yaml filename | `train_swe_v{N}.yaml` | `train_swe_v1.yaml` |
| sh filename | `train_swe_v{N}.sh` | `train_swe_v1.sh` |
| exp_name | `evotrainer_qwen35_9b_v{N}` | `evotrainer_qwen35_9b_v1` |
| Experiment output dir | `.../evotrainer_qwen35_9b_v{N}/` | |
| Git branch | `exp/9b-v{N}-{description}` | `exp/9b-v2-group-filter` |
| Worktree | `<PROJECT_ROOT>` | |

**Forbidden**: any extra suffix (e.g. `_optimized`, `_dense_reward`).

---

## 4. Key Differences vs 4B

### Hardware-scale comparison

| Item | 4B | 9B |
|------|-----|-----|
| Total GPU | 80 (10 nodes) | 160 (20 nodes) |
| Actor train | 0–31 (32 GPU) | 0–63 (64 GPU) |
| Actor infer | 32–63 (32 GPU) | 64–127 (64 GPU) |
| Reward compute | 64–79 (16 GPU) | 128–159 (32 GPU) |
| Reference | shared with training | shared with training (0–63) |
| Model parallel (train) | TP=2, CP=8, PP=1 | TP=1, SP=true, CP=16, PP=4 |
| Model parallel (infer) | vLLM TP=2 | vLLM TP=2 |

### Training-parameter differences

| Parameter | 4B | 9B | Notes |
|-----------|-----|-----|-------|
| rollout_batch_size | 32 | 64 | Rollouts per step |
| gradient_accumulation_steps | 32 | 64 | Gradient accumulation steps |
| train_env_manager.group_size | 4 | 8 | Rollouts per group (affects GRPO statistics) |
| ratio_type | — | gspo | 9B uses GSPO |
| dual_clip_loss | — | true | Pairs with GSPO |
| pg_clip_low / pg_clip_high | 0.2 | 0.2 / 0.27 | Asymmetric clip |
| pipeline_model_parallel_size | 1 | 4 | Model parallelism |

### Training parameters that stay identical (4B/9B shared)

| Parameter | Value | Notes |
|-----------|-------|-------|
| learning_rate | 1e-6 | Identical across scales |
| init_kl_coef | 0.0 | Safe under flat CR |
| adv_estimator | grpo | Identical across scales |
| reward_clip | 2 | Identical across scales |
| eval_steps | 25 | Identical across scales |
| max_steps | 1000 | Identical across scales |
| reward_normalization | mean_std + traj_group_id | Identical across scales |

---

# 5. Research Direction and AI Self-Iteration Protocol

> Detailed paper-level discussion is in Section 6 "Related Literature Review and Adaptation Assessment".

### Research positioning

**Core problem**: in agentic RL training, how to diagnose and intervene on model behavioral degradation.

**Differentiation from prior work**:

| Dimension | RAGEN-2 (paper.md) | FIPO (paper2.md) | Our direction |
|-----------|--------------------|-------------------|---------------|
| Setting | Generic agents (Sokoban, FrozenLake, etc.) | Math CoT (single-turn reasoning) | **Real-world software engineering** (SWE-bench, multi-tool multi-step) |
| Degradation type | Reasoning Template Collapse | Reasoning length plateau | **Action Pattern Collapse** (action-sequence templating) |
| Core metric | MI proxy (I(X;Z)) | Future-KL | **Behavior-pattern metric family** (edit_ratio, tool entropy, bigram repetition) |
| Intervention | SNR-Aware Filtering (filter prompt groups by reward variance) | Dense advantage (token-level weighting) | **To explore**: dynamic reward / behavior-aware filtering / staged curriculum |
| Reward | Mostly binary / sparse | Binary ORM | **Discrete staircase** (0/0.3/0.5/1.0) + multi-dimensional composition |

**Empirically verified key insights**:
- Success vs. failure differ not in *thinking* quality (both ~150 chars) but in **action patterns**
- Discrete staircase rewards push ZeroVar% as high as ~63% (much higher than continuous-reward settings)
- During training, edit_ratio drops and first_edit_turn rises — the model is "procrastinating"
- Echo trap (run→run idle loops) is a degradation mode unique to agentic settings

### AI self-iteration decision framework

Each version follows this closed loop:

```
1. Diagnose
   ├── Run test2.py to get all metrics
   ├── Focus on: BC% trend, behavior changes, ZeroVar%, stop_reason distribution
   └── Compare against baseline (v1) and the previous version

2. Hypothesize
   ├── Extract one core problem from the metric anomalies
   ├── State it in one sentence (e.g. "edit_ratio keeps dropping because the reward does not credit edits")
   └── Check whether paper.md / paper2.md theories apply

3. Intervene — only one variable per version
   ├── Reward dimension: add / remove / adjust a reward component
   ├── Filtering dimension: tune the group-filter threshold / strategy
   ├── Algorithm dimension: switch the PO type (GRPO/GSPO/DAPO/...)
   └── Curriculum dimension: dynamic reward scheduling

4. Verify
   ├── The primary metric (BC%) must not regress
   ├── The targeted metric must improve (e.g. edit_ratio recovers)
   ├── Side-effect check: do other dimensions worsen?
   └── Observe at least 50–100 steps of trend

5. Record
   └── Update the version history table: hypothesis, intervention, result, conclusion
```

### Candidate research routes (priority order)

**Route A: Action Pattern Collapse — diagnosis and intervention**
- Experiment 1 (v1 baseline): demonstrate action collapse exists (edit_ratio↓, tool entropy↓, bigram rep↑)
- Experiment 2 (v2): mitigate with SNR-Aware Filtering (validate RAGEN-2's method in agentic setting)
- Experiment 3 (v3): our approach — behavior-aware filtering / dynamic reward (beyond RAGEN-2)
- Experiment 4 (v4): ablations to validate the contribution of each component

**Route B: GRPO signal degeneration under discrete reward**
- Focus on high ZeroVar% → fewer effective gradient signals → strategy degradation
- Compare against continuous reward
- Propose solutions (finer staircase, dynamic threshold, hybrid reward)

**Route C: multi-dimensional degradation diagnostic framework**
- Combine A + B into a complete agentic-RL diagnostic framework
- Includes a metric system + automated alerts + intervention recommendations
- Suitable for a systems paper (e.g. benchmark / framework paper)

### Experimental design principles required by the paper

1. **Controlled variables**: only one change per version, otherwise attribution is impossible
2. **Baseline comparison**: every version must compare against v1 (flat CR, no filter)
3. **Multi-step trends**: show metric evolution over training, not just final values
4. **Per-CR-group analysis**: split by success / partial / fail and track behavior differences across training
5. **Statistical significance**: the val set has 616 records (77 problems × 8 rollouts); report std

---

## 6. Related Literature Review and Adaptation Assessment

> *This section, together with §5 (research positioning) and §13 (SeeUPO adaptation), captures outputs of the trainer's search-retrieval module — corresponding to 𝒮_search in §3.4 of the paper. Distinct from §11 / §12 which document our own empirical findings.*

### 6.1 Papers studied in depth

#### RAGEN-2: Template Collapse and SNR-Aware Filtering

**Paper**: RAGEN-2 — "Reasoning Template Collapse" in agentic RL training.

**Core problem**: in multi-turn agent RL, reasoning quality degrades but entropy monitoring cannot detect it.

**Key concepts**:
- **Template Collapse**: reasoning is still diverse (high H(Z|X)) but input-irrelevant (low I(X;Z)); entropy monitors miss it
- **SNR mechanism**: low reward variance → weak task gradient → KL/entropy regularization dominates → cross-input differences are flattened → templating
- **SNR-Aware Filtering**: filter low-signal prompt groups by reward variance to reduce input-irrelevant updates

**MI Proxy diagnostics**:
- In-Batch Cross-Scoring: condition reasoning on every prompt under the current model and compute the conditional likelihood
- Retrieval-Acc: can the reasoning recover its source prompt? Approaches random 1/P under template collapse
- MI-ZScore-EMA: an EMA-smoothed z-score of the matched-marginal difference

**Relation to our work**:
- ✅ SNR-Aware Filtering has been instantiated as our StdGroupFilter (EMA Top-p 0.9)
- ✅ The reward-variance → task-gradient mechanism explains our high Dead%
- ⚠️ RAGEN-2's setting is a generic agent (Sokoban, FrozenLake, etc.) with short trajectories of 3–15 turns
- ⚠️ Template Collapse focuses on reasoning quality; what we observe is Action Pattern Collapse (action-sequence templating)
- ❌ MI Proxy needs extra forward passes (P×G inferences); the cost is unacceptable on 9B

**Innovation directions worth borrowing**:
- Action Pattern Collapse is a dimension RAGEN-2 does not cover; this is our differentiation point
- Our behavior-metric family (edit_ratio, tool entropy, bigram rep) is functionally equivalent to RAGEN-2's MI proxy but with zero compute cost

---

#### FIPO: Future-KL Dense Advantage

**Paper**: FIPO — dense credit assignment via Future-KL, breaking GRPO's "reasoning-length ceiling".

**Core problem**: in GRPO/DAPO, outcome reward is broadcast uniformly to every token; this coarse credit assignment causes a reasoning-length plateau.

**Core method**:
- **Future-KL**: `FutureKL_t = Σ_{k=t}^{T} M_k · γ^{k-t} · Δlog p_k` — accumulated policy drift from t to the end
- **Influence weight**: `f_t = clip(exp(FutureKL_t), 1-ε_low, 1+ε_high)`
- **Dense Advantage**: `Ã_t = Â_t · f_t` — different tokens receive learning signals of different strength
- Stability design: extreme-ratio filtering (M_k) + influence-weight clipping + soft decay window (γ=2^{-1/τ}, τ=32)

**Key results**:
- Qwen2.5-32B + DAPO: AIME2024 50% → 56%
- Reasoning length grows from ~4k to 10k+ tokens (plateau broken)
- Uncontrolled FutureKL leads to length collapse (catastrophic)

**Relation to our work**:
- ⭐ Future-KL can be adapted as per-turn advantage weighting (use turn-level Δlog p instead of token-level)
- ⭐ The "key tokens decide reasoning direction" insight maps to our "edit→test" key-action structure
- ⚠️ O(B·L²) compute; expensive over our 45-turn long trajectories
- ⚠️ FIPO targets math CoT single-turn reasoning, not multi-tool multi-step
- ❌ Length growth is not our goal (we want precise edits, not longer thinking)

**Borrowable directions**:
- Future-KL → Per-Turn Influence Weight: compute each turn's contribution to the final CR
- We don't need the full O(L²); a turn-level approximation (45 turns, not tens of thousands of tokens) suffices

---

#### SeeUPO: Critic-free multi-turn convergence and per-turn credit assignment

**Paper**: SeeUPO (arXiv:2602.06554) — convergence guarantees for sequence-level agentic RL.

**Core insights**:
1. GRPO's "divide by std" breaks convergence: GRAE introduces structural bias that breaks PPO's monotonic-improvement property
2. Reverse-order update: model the multi-turn trajectory as a sequential-decision multi-agent problem and update each turn in reverse
3. Batch-level normalization: replace group-level std normalization to preserve the convergence property
4. Critic-free + multi-turn convergence to global optimum

**Relation to our work**:
- ⭐ Per-turn precise credit assignment: good turns get +adv, bad turns get -adv (vs. all turns sharing one advantage today)
- ⭐ Batch-level normalization may partly explain BC% late-stage regression in earlier versions
- ⚠️ Reverse-order update needs T separate gradient updates; for SWE-bench's 45 turns this multiplies training time by ×45
- ⚠️ SeeUPO's benchmarks are short (3–5 turns, AppWorld / BFCL)
- ❌ The ROLL framework does not support turn-by-turn independent batch construction

**Adaptation: Per-Turn Advantage Weighting**:
```
Standard GRPO:   advantage[i,t] = (R[i] - R̄) / σ                # same for every token in the trajectory
Per-Turn AW:      advantage[i,t] = (R[i] - R̄) / σ × w[turn_t]    # weighting differs per turn

Designing w[turn_t]:
  - turn_t is an edit immediately followed by a test → w > 1 (good behavior, amplify)
  - turn_t is an edit with no following test            → w < 1 (bad behavior, attenuate)
  - turn_t is a test                                     → w > 1 (verification, amplify)
  - turn_t is pure search                                → w ≈ 1 (neutral)
```
- Compute overhead is essentially zero (just multiply the advantage by a weight)
- Distinguishes good and bad turns within the same trajectory
- No extra T backward passes needed

---

### 6.2 Adaptation assessment of 18 agentic-RL credit-assignment papers

Assessed against our constraints — SWE-bench's 45-turn trajectories, GRPO without a critic, ROLL's chained-rollout framework:

#### ⭐⭐⭐⭐⭐ Best fit: HCAPO (Hindsight Credit Assignment)

- **Core idea**: infer per-turn contributions from the CR outcome (hindsight credit assignment)
- **Why it fits best**:
  - No extra model / tree structure
  - Naturally suits 45-turn long trajectories
  - Combines well with our existing step-level tracking (edit_streak, test_regression)
  - Can extract "which turns were the pivots" from successful (CR=1.0) trajectories
- **Possible implementation**: for successful trajectories, each turn's credit relates to "would CR flip if this turn's behavior were removed?"

#### ⭐⭐⭐⭐ High fit: SORL (training-collapse defense)

- **Core idea**: an SO-GRPO variant that targets training collapse directly
- **Adaptation points**:
  - Adaptive clipping prevents positive-feedback loops (which may be related to BC% late-stage regression in earlier versions)
  - SORL's collapse-detection mechanism can augment our six-dimensional diagnosis
- **Risk**: need to confirm SO-GRPO is compatible with StdGroupFilter

#### ⭐⭐⭐ Inspirational: GiGPO (two-level advantage)

- **Core idea**: two-level advantage estimation, grouping by anchor states
- **Workaround**: SWE-bench does not satisfy the "same environment state" prerequisite; we can use turn-behavior features (has_edit + has_test) as pseudo-anchors
- **Limitation**: anchor-grouping quality depends on the accuracy of the behavior-feature design

#### ⭐⭐⭐ Inspirational: IGPO (information gain)

- **Core idea**: per-step compute the information gain of P(correct)
- **Workaround**: approximate information gain via test pass-rate change
- **Limitation**: per-step P(correct) is infeasible; the workaround's accuracy is unverified

#### ❌ Not applicable: tree-based methods

- ATPO / TreeAdv / Tree-GRPO / SEE A-R1: ROLL does not support tree sampling; tree search over 45 turns is memory-prohibitive

#### ❌ Not applicable: methods requiring an extra model

- CriticSearch: needs a frozen Critic LLM; GPU cost is too high
- AgentPRM: needs a value network, which violates the GRPO no-critic constraint

#### ❌ Not applicable: per-turn independent updates

- Turn-PPO: the per-turn MDP idea is suggestive but ROLL does not support per-turn independent updates

#### Key warnings

- All these methods benchmark on 3–15 turn short trajectories; copying them blindly is high-risk
- Earlier "T3 truncation" copy attempts collapsed; adaptation is mandatory
- Credit assignment under long (45-turn) trajectories is an unexplored area — our innovation opportunity

---

### 6.3 Mapping from paper methods to our versions

| Paper method | Our adaptation | Version | Status |
|--------------|---------------|---------|--------|
| RAGEN-2 SNR Filtering | StdGroupFilter EMA Top-p | v4 | ✅ Landed |
| RAGEN-2 MI Proxy | Behavior metric family (edit_ratio, tool_entropy, etc.) | v1+ | ✅ Landed |
| Data sorting (drop pass_rate=0) | Interleaved 1958 records | v7 | ❌ Too little data, exploration collapse |
| LLM Judge instruction_following | 27B reward model scoring instruction following | v8 | ✅ Implemented |
| SeeUPO Batch-Level normalization | Replace group-level with batch-level | future | TBD |
| SeeUPO Per-Turn Credit | Per-Turn Advantage Weighting | future | TBD |
| FIPO Future-KL | Per-Turn Influence Weight | future | Design pending |
| HCAPO hindsight | Reverse-engineer per-turn contribution from CR | future | Design pending |
| SORL adaptive clipping | Prevent positive-feedback loops | TBD | Evaluation pending |
| GiGPO two-level advantage | Pseudo-anchor grouping by behavior features | TBD | Evaluation pending |

---

## 7. Version Plan

### v1 (current): Baseline — Flat CR

**Core configuration**:
- Reward: staircase CR (0 / 0.3 / 0.5 / 1.0), no filter, no bonus
- Hardware / training parameters: inherit baseline configuration
- Purpose: establish a clean 9B baseline; observe pure flat CR behavior on 9B

**Why v1 has no group filter / bonus**:
1. We need a clean baseline to assess 9B's intrinsic Dead% and GRPO-signal distribution
2. group_size=8 (4B uses 4); the Dead% pattern may differ
3. First inspect the six-dimensional diagnosis on v1, then decide what to add in v2

**Key v1 settings**:
```yaml
exp_name: "evotrainer_qwen35_9b_v1"
# Reward: the v4 flat CR from the baseline code
# No group_filter_cls and no bonus
# init_kl_coef: 0.0 (safe under flat CR, validated by 4B experiments)
# ratio_type: gspo + dual_clip_loss: true
```

**v1 observation goals**:
- Step-0 BC% baseline and the training curve trend
- Dead% distribution (under group_size=8)
- Turn inflation (does 9B exhibit the same problem?)
- Diversity trend
- Whether GSPO introduces new training instability
- Side-by-side comparison against the 4B v4 baseline

### v2 (planned): apply the best 4B reward structure

**Expected configuration** (subject to v1 results):
- Migrate Group-variance filtering from 4B (std<0.01)
- Consider migrating bonus (cr=1.0 and turns≤threshold → +0.1)
- The threshold may need recalibration based on the 9B turn distribution

---

## 8. 4B Iteration Lessons (Reusable in 9B)

### Validated reward-design rules

1. **Flat CR (0/0.3/0.5/1.0) is the best CR shape**: healthier than continuous CR (Diversity↑) and more stable than binary
2. **Group-variance filtering is effective**: std<0.01 filters dead data and helps under any CR shape
3. **The first-level per-trajectory filter is redundant**: backtests cover only 2–4%; fully subsumed by the group filter
4. **Bonus reward is safe and effective**: +0.1 when cr=1.0 and turns≤threshold lowers Dead% by ~8pp and is hack-proof

### Three classes of toxic reward (never use)

| Class | Symptom |
|-------|---------|
| Efficiency factor | The model learns to exit in 1 turn to grab the maximum efficiency score |
| Iterative-debugging reward | Turn count explodes (e.g. 38 → 56) |
| f2p progress reward | Turn count explodes; BC% drops by ~10pp within 10 steps |

**Common pattern**: any reward that "directly optimizes a behavior metric" gets reward-hacked.

### Six-dimensional evaluation framework (mandatory)

| Dimension | Description | Key threshold |
|-----------|-------------|---------------|
| BC% | Score | High does not imply healthy; combine with other dimensions |
| Dead% | Fraction of zero-gradient groups | Should drop or stay flat |
| Diversity | Outcome diversity | **Should rise** (litmus test) |
| AvgTurns | Average dialogue turns | Inflation is a common failure mode |
| 1stEdit | Time to first edit | Steady increase = strategy-degradation early warning |
| Anomaly jumps | Step-to-step metric jumps | Should be 0 |

Tool: `python3 scripts/generate_health_report.py --exp_dir <DIR>`

### test2 diagnostic system (`scripts/test2.py`)

Designed from empirical analysis of 9B v1 step-0 rollouts; analyzes **val** data and stays consistent with `test.ipynb`.

Run: `python scripts/test2.py` or `python scripts/test2.py --no-plot`

Edit `get_path_groups()` to configure which experiment paths to analyze.

#### Core metrics (consistent with test.ipynb)

| Field | Meaning |
|-------|---------|
| `binary_correctness` (BC%) | Whether the fail-to-pass tests fully pass (0 or 1) |
| `correctness_reward` (CR) | Staircase correctness score (0 / 0.3 / 0.5 / 1.0) |
| `final_reward` (FR) | Final reward (after bonus and other add-ons) |
| `n_turns` | Number of turns in which the model issued a tool call |
| `n_tokens` | Total prompt_ids + response_ids tokens of the trajectory |

#### Behavior metrics (added, based on rollout empirical analysis)

| Field | Meaning | Design rationale |
|-------|---------|------------------|
| `edit_ratio` (EditR%) | Edit-op share = (search_replace + create_file + edit_file + delete_file) / total tool calls | SUCCESS samples have edit_ratio≈27%; PARTIAL samples ≈1% — strong discrimination |
| `test_ratio` (TestR%) | Test-op share = run_in_terminal / total tool calls | FAIL_LONG samples reach ~53% (idling); a healthy range is 25–30% |
| `search_ratio` (SrchR%) | Search-op share = (search_codebase + grep_code + read_file + ...) / total tool calls | Helps observe the explore-vs-execute balance |
| `first_edit_turn` (1stEdit) | Step index of the first edit op | Drifting later = strategy degradation (e.g. 6.9 → 10.1) |
| `has_test_run` (HasTest%) | Whether `run_in_terminal` was ever invoked (0/1) | FAIL_PARTIAL samples never run a test — directly causes failure |
| `tool_type_entropy` (ToolEnt) | Shannon entropy of the tool-type distribution (bits) | Low entropy = action-space collapse (the model uses only a few tools) |
| `max_consecutive_same_tool` (MaxConsec) | Max consecutive calls of the same tool | FAIL_LONG samples show run→run streaks of 32 — a classic echo-trap signal |
| `bigram_repetition_rate` (BigramRep) | Tool-call bigram repetition rate = 1 - unique_bigrams / total_bigrams | Higher = more templated; corresponds to RAGEN-2's Template Collapse |
| `search_before_edit` (SrchB4Ed) | Number of search/read calls before the first edit | Whether the model understands the code before acting |
| `create_file_count` | Count of `create_file` invocations | Excessive count may mean creating debug scripts instead of fixing |

#### Training-health metrics (added)

| Field | Meaning | Design rationale |
|-------|---------|------------------|
| `group_reward_variance` (GrpVar) | Mean within-group reward variance (per instance) | Core GRPO signal-quality metric; variance=0 → zero gradient |
| `zero_variance_group_ratio` (ZeroVar%) | Fraction of zero-variance groups | At 9B v1 step 0 reaches ~63% — characteristic of discrete reward |
| `stop_reason` | Distribution of trajectory termination reasons | Categories: success / partial / fail / truncation / max_tokens / echo_trap / abandon / idle |
| `git_command_usage` (Git%) | Fraction of trajectories that used `git show` / `git log` | Information-leak detection; should drop to 0 once `clear_git_log=true` |

#### Reasoning metrics (added)

| Field | Meaning | Design rationale |
|-------|---------|------------------|
| `thinking_length_mean` (ThkMean) | Average character length of the thinking content per trajectory | Empirically success/failure thinking lengths are similar (~150 chars); the difference is in action patterns |
| `thinking_length_total` (ThkTotal) | Total character length of thinking per trajectory | Combine with n_turns to assess whether long trajectories think effectively |

#### Model-state metrics (added)

| Field | Meaning | Design rationale |
|-------|---------|------------------|
| `mean_logprob` (LogP) | Average per-token logprob per trajectory (excluding padding) | Exploration-collapse detector; closer to 0 is more dangerous (e.g. -0.237 → -0.087, monotonically rising) |
| `solution_diversity` (1-Jaccard) | 1 - Jaccard similarity of edited file sets across rollouts of the same instance | Exploration-diversity gauge; lower = strategy ossification (all rollouts edit the same files) |

#### Per-CR-group behavior analysis (added)

For every step, test2 separately reports behavior metrics for the success / partial / fail groups, revealing how the behavioral gap between successful and failed runs evolves with training. This is one of the most valuable analytical dimensions for the paper.

#### Key empirical findings (9B v1 step 0)

| Sample type | Steps | Core behavior |
|-------------|-------|---------------|
| SUCCESS (CR=1.0) | 33 | edit_ratio=27%, run_in_terminal=27% — balanced edits and tests |
| PARTIAL (CR=0.5) | 42 | edit_ratio≈1% — almost no editing; 36% run_in_terminal |
| FAIL_LONG (CR=0, echo_trap) | 100 | run_in_terminal≈53%, run→run bigram = 32 — idling |
| FAIL_SHORT (CR=0, max_tokens) | 15 | Single-turn response_ids = 8192 — hits the token cap |
| FAIL_PARTIAL (CR=0.3) | 7 | 0 invocations of run_in_terminal — submitted without testing |

**Core insight**: success vs. failure differ not in *thinking* quality (both ≈150 chars) but in **action patterns** (edit_ratio, tool entropy, whether tests are run). This is a dimension RAGEN-2 does not cover.

---

## 9. Iteration Workflow

### Standard flow

1. **Analyze the previous version**: run the six-dimensional health report → form a conclusion
2. **Design the new version**: focus on one core hypothesis, control variables
3. **Create the config**: copy yaml/sh; update `exp_name` / version number
4. **Code changes**: if reward logic must change, isolate via worktree
5. **User launches training**
6. **Record results**: update the iteration log

### Mandatory config changes for a new version

**yaml file**:
- `exp_name` → `evotrainer_qwen35_9b_v{N}`

**sh file**:
- `CONFIG_NAME` → `train_swe_v{N}`
- `EXP_NAME` → `evotrainer_qwen35_9b_v{N}`
- When using a worktree: point `PYTHONPATH` and `PROJECT_ROOT` at the worktree path

### Version control

```
baseline (or master)
  ├── exp/9b-v1-trajectory-filter  ← v1
  ├── exp/9b-v2-ragen-filter       ← v2
  ├── exp/9b-v3-continuous-cr-bonus ← v3
  └── ...
```

Note: 9B branches are prefixed with `9b-` to distinguish them from other-scale experiment branches.

---

## 10. Version History

| Version | BC% | Status | Core configuration |
|---------|-----|--------|--------------------|
| v0 | 30.19 | baseline | Base model (no RL; paper Table 1 / §A.6) |
| v1 | ~31% | done | Flat CR (0/0.3/0.5/1.0), no filter, no bonus, KL=0, GSPO |
| v2 | ~33% | done | RAGEN group-level trajectory filter; otherwise same as v1 |
| v3 | ~33% | done | Continuous CR + completion bonus + StdGroupFilter (Top-K 50%) + drop global whitening |
| v4 | ~36% | done | Binary CR (0/0.1/1.0) + behavior reward + StdGroupFilter (EMA Top-p 0.9) + drop global whitening |
| v5 | ~31% | failed branch | v4 baseline + step-level penalty + soft truncation; below v4 |
| v6 | ~33% | failed branch | v4 baseline + edit_streak penalty + scores-dim fix; below v4 |
| v7 | ~32% | failed branch | v4 baseline + filtered training data; below v4, rolled back |
| v8 | ~38% | done (best) | v4 baseline + instruction-following LLM Judge; final_reward = CR×1.0 + IF×0.1 + SBE + ETT + StdGroupFilter |

---

## 11. Known Experience

- 9B hardware / parallelism parameters (TP/CP/PP/node count/gradient_accumulation, etc.) follow the baseline configuration and are out of optimization scope
- Training hyperparameters (lr=1e-6, KL=0, reward_clip=2, etc.) stay consistent with the 4B baseline
- The optimization focus is the reward structure (CR shape, filtering mechanism, bonus, etc.), in line with the 4B experimental direction
- group_size=8 (4B uses 4) may shift Dead% characteristics; verify with v1 results
- GSPO (ratio_type=gspo + dual_clip_loss) is enabled for the 9B run but absent in the 4B run; watch its effect on training stability
- `init_kl_coef=0.0` was empirically safe under flat CR in 4B experiments and is reused on 9B
- Any new field added to `env_config` must also be declared in the `QwenMCPSweEnvConfig` dataclass in `config.py` (dacite constraint)
- Dataset paths under `<DATA_DIR>/` are shared data — do not modify
- **The baseline branch is the main branch; per-version code lives in worktrees and only merges back if results are good**
- **`clear_git_log` defaults to True**, preventing the model from peeking via `git show` / `git log`
- **GRPO whitening math**: advantage = (reward - mean) / std depends only on the *grouping structure* of within-group rewards, not their absolute values. Widening CR gaps does not help under whitening; only changing the grouping structure (continuous CR) does
- **v1 core finding**: dead data (within-group reward all equal → advantage=0 → zero gradient) accounts for 55%; CR=0.5 covers 57% of rollouts


---

## 12. V4 Design Details

### V4 → V5 core changes

| Item | V4 | V5 | Why |
|------|----|----|-----|
| CR shape | 0 / 0.1 / 1.0 | **0 / 0.1 / 1.0** (unchanged) | Binary CR keeps the signal simple |
| Search-Before-Edit | +0.1 | **removed** | SBE is a binary constant — ~99% of rollouts get the full bonus, no within-group variance, a constant offset rather than a learning signal |
| Edit-Then-Test | +0.15 | **removed** | Same as SBE: binary constant, ~99% coverage |
| edit_streaks penalty | none | **-0.01 per streak** | 2+ consecutive edits without test; v4 backtest shows just 1 violation across 5 steps — safe upper bound |
| test_regression penalty | none | **-0.0 per occurrence** | pytest failures+errors increase after edit; v4 detection had no data, only tracked, not penalized |
| Hard-bad truncation | none | **2 consecutive errors** | Two consecutive tool/syntax errors → truncate the invalid suffix |
| Soft truncation | none | **20 turns no progress** | Consecutive edits without test, or pure-search streaks → truncate. Threshold=20 only mis-truncates 3–5 of CR=1.0 trajectories |
| StdGroupFilter | EMA Top-p 0.9 | **EMA Top-p 0.9** (unchanged) | Continue filtering dead data |
| Global whitening | false | **false** (unchanged) | Group whitening only |

### V5 reward formula

```
trajectory_reward = CR − 0.01 × edit_streaks − 0.0 × test_regressions
```

- **CR** = 0 / 0.1 / 1.0 (same as v4)
- **edit_streaks** = total streaks of 2+ consecutive edits without test, weight 0.01 (v4 backtest safety bound)
- **test_regressions** = number of times pytest failures rose after an edit, weight 0.0 (tracking only; tune later when data is available)
- **All-negative design**: every step-level signal can only subtract; the only positive signal is CR — avoids reward hacking

### V5 T3 truncation rationale

Truncation does not change the reward, but changes which gradient signal is in scope:
- Without truncation: good and bad behaviors share the same advantage; bad-behavior gradients dilute good-behavior gradients
- With truncation: noisy gradients from invalid suffixes are eliminated, focusing the advantage signal on the valid prefix
- **Hard-bad**: 2 consecutive tool/syntax errors → truncate immediately (clearly meaningless operations)
- **Soft**: 20 turns with no progress → truncate (pure search without edit/test, or edit without test)
- After truncation, the final pytest evaluation still runs, so CR is unaffected
- **Backtest validation** (v4 val data, 308 trajectories × 5 steps):
  - threshold=15: truncates 15–23%, but mis-truncates 6–18 of CR=1.0 (6–17%) — **too aggressive**
  - threshold=20: truncates 4–9%, only 3–5 mis-truncations of CR=1.0 (3–5%) — **safe**
  - hard_bad rate is 0% on the v4 data
- **w_es=0.01 safety check**: across 5 steps only step 50 had 1 violation (CR=0.1 vs CR=0.0 boundary); zero violations between CR=1.0 and CR=0.1

### V5 key fixes

1. **Drop the `truncated=True` legacy filter**: previously, when `rollout_cache.truncated=True`, `obtain_outcome_reward` returned -999 to discard the trajectory. In current code `truncated` is only set on `max_turn_exceed`, which already routes through `is_finished=False` first, so the -999 path is dead code — removed
2. **Drop `turn['penalty'] = final_reward`**: the `penalty` field is broadcast to every response token of that turn under the token_penalty mechanism; with `add_token_penalty=True` it overwrites advantages and breaks them. Step-level penalties are already aggregated into `episode_score` via `final_reward += penalty`; no need to double-write the `penalty` field

### Why V5 drops SBE/ETT

V4's SBE (+0.1) and ETT (+0.15) are binary constants:
- SBE: ~99% of rollouts get the +0.1; only 1/33 dead-data groups have within-group variance
- ETT: ~99% of rollouts get the +0.15; only 1/33 dead-data groups have within-group variance
- CR=0.1 + SBE(0.1) + ETT(0.15) = 0.35 is essentially constant for almost all rollouts
- These are constant offsets rather than learning signals — they cannot provide within-group discrimination

### V5 step-level signal tracking

Tracked in real time inside `step()`:
1. **edit_streaks**: classify each turn's tool type; consecutive edits without a test increment the streak
2. **test_regression**: when a turn contains `run_in_terminal`, parse pytest output via `FailToPassJudge.parse_pytest_summary()` and compare the failed+error counts
3. **hard_bad**: detect tool-call error / syntax error / import error
4. **no_progress**: count turns of consecutive pure search without edit/test, or consecutive edits without test (edit without verify)

Aggregation in `obtain_outcome_reward()`:
- `edit_streak_penalty = -0.01 × edit_streak_total`
- `test_regression_penalty = -0.0 × test_regression_count`
- `final_reward = CR + edit_streak_penalty + test_regression_penalty`

### V5 numerical validation

Backtested on the full 5-step val rollouts (308 × 5 = 1540 trajectories):
- edit_streaks vs CR: CR=1.0 averages 0.95–1.18; CR=0.1 averages 1.41–2.01; CR=0.0 averages 2.15–3.31
- w_es=0.01: across 5 steps, only step 50 has 1 violation (CR=0.1 vs CR=0.0); zero violations CR=1.0 vs CR=0.1
- w_es=0.02: step 50 has 7 violations ⚠️
- w_es=0.03: step 50 has 26 violations ⚠️
- **Between CR=1.0 and CR=0.1, no violation occurs at any weight** — the core safety constraint holds
- During v4 training BC% rose then fell (peak ~35% at step75 → ~33% at step100); root cause is action timidity (the model learned that fewer edits are safer, but overcorrected)

### V1 → V4 change list

| Item | V1 | V4 | Why |
|------|----|----|-----|
| CR shape | 0 / 0.3 / 0.5 / 1.0 | **0 / 0.1 / 1.0** | Simplify the signal; remove pseudo-dead-data (CR=0.5 covers 57% of rollouts → within-group ties → zero gradient) |
| Search-Before-Edit | none | **+0.1** | Reinforce "search before editing"; SUCCESS edit_ratio=27% vs failure ~1% |
| Edit-Then-Test | none | **+0.15** | Reinforce "verify after editing"; FAIL_PARTIAL samples submit without testing |
| Trajectory filtering | none | **StdGroupFilter EMA Top-p** | RAGEN V2-style SNR-Adaptive, tracks the std distribution adaptively |
| Global whitening | whiten_advantages=true | **false** | Group whitening only (traj_group_id), preserves the strength of the behavior reward signal |
| clear_git_log | false | **true** | Prevent peeking |
| ratio_type | gspo | **removed (use default)** | gspo errored at v3, removed at v4 |

### V4 CR discretization (0/0.1/1.0)

V1's CR shape (0/0.3/0.5/1.0) had serious problems:
- CR=0.5 (partial pass) covers 57% of rollouts → many within-group ties → std=0 → zero gradient
- CR=0.3 (full failure) rewarded failed behavior and increased the chance of within-group ties
- With group_size=8 and 4 CR levels, full within-group ties are very likely

V4 simplifies to three levels:
- **1.0**: all tests pass (fail_to_pass all green)
- **0.1**: tests ran but did not all pass (covers full failure and partial pass)
- **0.0**: did not finish / never tested / parsing failed

Key design considerations:
- The 0.1 (rather than 0) preserves a minimal signal distinguishing "tried" from "did not try", but the signal is weak and is nearly nullified after GRPO whitening
- Removing 0.3 and 0.5 substantially lowers the chance of within-group ties, reducing dead data

### V4 behavior rewards

| Reward | Condition | Value | Rationale |
|--------|-----------|-------|-----------|
| Search-Before-Edit | A search call occurs before the first edit | +0.1 | SUCCESS edit_ratio=27%, PARTIAL ~1% |
| Edit-Then-Test | A test call occurs after the last edit | +0.15 | FAIL_PARTIAL samples submit without testing; edit_then_test matters more |

**Safety analysis**:
- Search-Before-Edit: cannot be hacked by trivial search calls — the search must precede the edit and produce meaningful content
- Edit-Then-Test: cannot be hacked because after editing the model must actually run a test; the test result does not affect the reward (only whether a test was invoked)
- Both rewards confirm "behavior that already happened" rather than steering "behavior that should happen", so they cannot be reward-hacked
- The maximum total of 0.1 + 0.15 = 0.25 < 1.0 (CR=1.0), so the correctness signal still dominates

### V4 StdGroupFilter (EMA-adaptive Top-p)

Problems with V3's Top-K sliding-window scheme:
- Always cuts 50% of groups — but with behavior rewards most groups have variance
- The fixed window size (16) cannot adapt to distribution shifts

V4 adopts the RAGEN V2-style EMA-adaptive Top-p:

1. **EMA tracking**: after computing each group's within-group reward std, use an EMA to dynamically estimate the std distribution (mean + variance)
2. **Adaptive threshold**: `threshold = ema_mean - z × ema_std`, with z derived from `keep_ratio` via the inverse normal CDF
3. **Two-stage filtering**:
   - Dead-data backstop: std=0 is always filtered (independent of the threshold)
   - SNR-adaptive: std < threshold → low-signal group is filtered
4. **Warmup**: for the first 20 groups rely only on the std=0 backstop, avoiding excessive filtering early on
5. **Compatibility mode**: `filter_mode='absolute'` uses a fixed threshold; `filter_mode='top_p'` uses the EMA path

Configuration:
- `group_filter_mode: top_p` — use EMA-adaptive
- `group_keep_ratio: 0.9` — keep the 90% with the highest variance (filter the lowest 10%)
- `group_min_keep_ratio: 0.5` — safety floor: keep at least 50%
- `group_ema_decay: 0.1` — EMA decay (new data weighted 10%)
- `group_min_std_threshold: 0.01` — backwards-compatible absolute mode

`keep_ratio=0.9` → z≈1.28, meaning groups with std below `ema_mean - 1.28×ema_std` are filtered.

### V4 file locations

| File | Worktree path |
|------|--------------|
| env.py | `<PROJECT_ROOT>` |
| config.py | `<PROJECT_ROOT>` |
| StdGroupFilter | `<PROJECT_ROOT>` |
| train_swe_v4.yaml | `<PROJECT_ROOT>` |
| train_swe_v4.sh | `<PROJECT_ROOT>` |

### V5 file locations

| File | Worktree path |
|------|--------------|
| env.py | `<PROJECT_ROOT>` |
| config.py | `<PROJECT_ROOT>` |
| StdGroupFilter | `<PROJECT_ROOT>` |
| train_swe_v5.yaml | `<PROJECT_ROOT>` |
| train_swe_v5.sh | `<PROJECT_ROOT>` |

### V6 file locations

| File | Worktree path |
|------|--------------|
| env.py | `<PROJECT_ROOT>` |
| config.py | `<PROJECT_ROOT>` |
| StdGroupFilter | `<PROJECT_ROOT>` |
| train_swe_v6.yaml | `<PROJECT_ROOT>` |
| train_swe_v6.sh | `<PROJECT_ROOT>` |

### V5 config fields

```python
# config.py (QwenMCPSweEnvConfig)
clear_git_log: bool = True                          # clean git history; prevent peeking
filter_max_new_tokens_traj: bool = True             # trajectory-truncation filter
# --- v5 step-level reward ---
use_edit_streak_penalty: bool = False               # edit_streaks penalty switch
edit_streak_penalty_weight: float = 0.01            # -0.01 per streak (v4 backtest safety bound)
use_test_regression_penalty: bool = False           # test_regression penalty switch
test_regression_penalty_weight: float = 0.0         # -0.0 per regression (track-only for now)
# --- v5 T3 truncation ---
use_hard_bad_truncation: bool = False               # hard-bad truncation switch
hard_bad_threshold: int = 2                         # 2 consecutive hard errors → truncate
use_soft_truncation: bool = False                   # soft truncation switch
soft_truncation_threshold: int = 20                 # 20 turns of no progress → truncate (v4 backtest safe)
# --- v4 compat (unused in v5) ---
use_search_before_edit_reward: bool = False
search_before_edit_reward_value: float = 0.1
use_edit_then_test_reward: bool = False
edit_then_test_reward_value: float = 0.15
# --- v4 group filter ---
group_filter_mode: str = 'top_p'                    # top_p=EMA-adaptive / absolute=fixed threshold
group_keep_ratio: float = 0.9                       # keep the highest-variance 90%
group_min_keep_ratio: float = 0.5                   # safety floor
group_ema_decay: float = 0.1                        # EMA decay
group_min_std_threshold: float = 0.01               # backwards-compatible absolute mode
```

### V6 config fields

```python
# config.py (QwenMCPSweEnvConfig)
clear_git_log: bool = True                          # clean git history; prevent peeking
filter_max_new_tokens_traj: bool = True             # trajectory-truncation filter
# --- v4 behavior rewards (kept by v6) ---
use_search_before_edit_reward: bool = False         # disabled: post-whitening it is a constant offset
search_before_edit_reward_value: float = 0.1
use_edit_then_test_reward: bool = False             # disabled: same as SBE; edit_streak is a more precise signal
edit_then_test_reward_value: float = 0.15
# --- v6 edit_streak penalty ---
use_edit_streak_penalty: bool = False               # edit_streaks penalty switch
edit_streak_penalty_weight: float = 0.01            # -0.01 per streak (v4 backtest safety bound)
# --- v4 group filter ---
group_filter_mode: str = 'top_p'                    # top_p=EMA-adaptive / absolute=fixed threshold
group_keep_ratio: float = 0.9                       # keep the highest-variance 90%
group_min_keep_ratio: float = 0.5                   # safety floor
group_ema_decay: float = 0.1                        # EMA decay
group_min_std_threshold: float = 0.01               # backwards-compatible absolute mode
```

### V6 design details

#### V5 → V6 core changes

| Item | V5 | V6 | Why |
|------|----|----|-----|
| CR shape | 0 / 0.1 / 1.0 | **0 / 0.1 / 1.0** (unchanged) | Binary CR keeps the signal simple |
| Search-Before-Edit | removed | **disabled** (false) | Constant offset after whitening; no learning signal under GRPO |
| Edit-Then-Test | removed | **disabled** (false) | Same as SBE; edit_streak penalty already covers ETT's role more precisely |
| edit_streaks penalty | -0.01 per streak | **-0.01 per streak** (kept) | Targets the "edit then no test" failure mode |
| T3 truncation | hard_bad(2) + soft(20) | **none (removed)** | T3 truncation adds complexity and mis-truncation risk; v6 focuses on the edit_streak penalty effect |
| test_regression | tracked (w=0) | **none (removed)** | No data on v4; remove to reduce code complexity |
| StdGroupFilter | EMA Top-p 0.9 | **EMA Top-p 0.9** + scores-dim fix | Use `scores.sum(dim=-1)` instead of `.item()` to avoid the multi-dim tensor error |
| turn['penalty']=final_reward | not written | **not written** (inherited from v5 design) | Writing this field, under the token_penalty mechanism, replaces advantages and breaks them |

#### V6 reward formula

```
trajectory_reward = CR − 0.01 × edit_streaks
```

- **CR** = 0 / 0.1 / 1.0 (same as v4)
- **edit_streaks** = total streaks of 2+ consecutive edits without test, weight 0.01
- **SBE/ETT disabled**: constant offsets after whitening, no GRPO learning signal; the edit_streak penalty more precisely covers ETT's role

#### V6 key fixes

1. **Remove `turn['penalty'] = final_reward`** (inherited from v5): the `penalty` field is broadcast to every response token of that turn by the token_penalty mechanism in MCPSweEnvManager; writing `final_reward` overwrites advantages and breaks them
2. **StdGroupFilter scores-dim fix**: `data.batch['scores']` may be a multi-dim tensor (e.g. `[reward_dim]`); use `sum(dim=-1)` to get a scalar — otherwise `.item()` raises ValueError

#### V6 vs V5 relationship

V6 is **not** a continuation of V5; it is a new version on top of the V4 baseline:
- V5 dropped the SBE/ETT positive rewards; v6 **also disables** them (constant offsets after whitening, no GRPO learning signal)
- V5 added T3 truncation and test_regression tracking; v6 does not (removed to reduce complexity)
- V6 only inherits the edit_streak penalty mechanism and the `turn['penalty']` fix from v5
- V6's training data uses `v7-train.9b-sorted.no-easy-update.jsonl` (newer than v4/v5's training data)

#### V6 logical-conflict check

- **ETT disabled, no conflict**: edit_streak fully covers ETT's role and is more precise (per-segment counts vs. one global binary)
- **SBE disabled, independent of edit_streak**: neither is enabled; no conflict
- **`add_token_penalty=false` ensures rewards propagate normally**: v6 removed `turn['penalty']=final_reward`, so token_penalty is all zeros. With `add_token_penalty=false`, rewards take the standard path (normalize → EOS token → compute_reinforce_return → group-level whitening)

### V6 yaml key configuration

```yaml
# env_config:
clear_git_log: true
# --- v6: SBE/ETT disabled (constant offset after whitening is ineffective) ---
use_search_before_edit_reward: false
search_before_edit_reward_value: 0.1
use_edit_then_test_reward: false
edit_then_test_reward_value: 0.15
# --- v6 edit_streak penalty ---
use_edit_streak_penalty: true
edit_streak_penalty_weight: 0.01
# --- v4 group filter ---
group_filter_mode: top_p
group_keep_ratio: 0.9
group_min_keep_ratio: 0.5
group_ema_decay: 0.1

# train_env_manager:
group_filter_cls: roll.pipeline.agentic.agentic_pipeline.StdGroupFilter

# top-level:
whiten_advantages: false          # group whitening only
add_token_penalty: false          # must be false; otherwise an all-zero token_penalty causes zero gradient
```

### V4 training-process analysis (val 308 × 5 steps)

**BC% rises then falls**: step0 ≈ 30% → step75 peak ≈ 35% → step100 retreats to ≈ 33%.

**The core issue is Action Timidity, not Action Pattern Collapse**:
- EditRatio drifts down from ~22% to ~18%; SearchRatio drifts up from ~31% to ~35%
- The model learned "search is safer", but overcorrected — BC% regresses

**Behavior gap between success and failure (step 100)**:

| Behavior | CR=1.0 | CR=0.1 | CR=0.0 | Interpretation |
|----------|--------|--------|--------|----------------|
| TestRatio | ~31% | ~29% | ~21% | Successes test the most |
| EditRatio | ~17% | ~18% | ~20% | Failures edit blindly |
| ES_avg | 0.95 | 1.41 | 2.15 | Failures have 2× the edit-streak count |
| NP_max | 10.2 | 11.8 | 13.5 | Failures have longer no-progress runs |

**Good behavior = frequent testing + precise edits + no edit streaks without tests.**

**V5's edit_streak penalty targets exactly this**: punishes the "edit-without-test" failure mode while preserving sensible edits.

**Analysis script**: `python scripts/analyze_step_level.py --exp_dir <EXP_ROOT>/evotrainer_qwen35_9b_v4`.

### V4 bug-fix log

| Bug | Fix | Impact |
|-----|-----|--------|
| Insufficient EMA warmup | Warmup raised from 10 to `max(20, 1/ema_decay) = 20` | Early-training filter rate drops from 40% to ~5% |
| Compat-mode mis-trigger | filter_mode check changed from `'absolute' or min_std_threshold!=None` to `'absolute'` only | top_p mode no longer mis-routes to absolute thresholding |
| scores dim | `_compute_group_std` now uses `scores.sum(dim=-1).item()` | Fixes `ValueError: only one element tensors can be converted to Python scalars` |

### Launch commands

**V4:**
```bash
pip install demjson3
pip install fastmcp
export RAY_gcs_rpc_server_connect_timeout_s=30000
export RAY_py_gcs_connect_timeout_s=60000
cd <PROJECT_ROOT>
bash examples/qwen35-9b-agentic/train_swe_v4.sh
```

**V5:**
```bash
pip install demjson3
pip install fastmcp
export RAY_gcs_rpc_server_connect_timeout_s=30000
export RAY_py_gcs_connect_timeout_s=60000
cd <PROJECT_ROOT>
bash examples/qwen35-9b-agentic/train_swe_v5.sh
```

**V6:**
```bash
pip install demjson3
pip install fastmcp
export RAY_gcs_rpc_server_connect_timeout_s=30000
export RAY_py_gcs_connect_timeout_s=60000
cd <PROJECT_ROOT>
bash examples/qwen35-9b-agentic/train_swe_v6.sh
```

---

## 13. SeeUPO Inspiration and SWE Adaptation

### Paper

SeeUPO (arXiv:2602.06554) — proposes convergence guarantees for sequence-level agentic RL.

### SeeUPO core insights

1. **GRPO's "divide by std" breaks convergence**: GRAE (group-relative advantage estimation) divides by within-group std and introduces structural bias, breaking PPO's monotonic-improvement property — this may partly explain BC% late-stage regression in earlier versions
2. **Reverse-order update**: model the multi-turn trajectory as a sequential-decision multi-agent problem; update the policy turn by turn in reverse, with each turn having its own advantage — yielding precise per-turn credit assignment
3. **Batch-level normalization**: replace group-level std normalization with batch-level normalization to preserve the convergence property
4. **Convergence guarantee**: critic-free + multi-turn convergence to global optimum — strong theoretical value

### SeeUPO vs V5 — fundamental differences

| Dimension | SeeUPO | V5 |
|-----------|--------|-----|
| Problem solved | Algorithmic convergence | Behavioral degradation |
| Layer addressed | Algorithm structure (training loop) | Reward design (signal + truncation) |
| Credit assignment | Per-turn precise (good turn +adv, bad turn -adv) | Per-trajectory coarse (the whole trajectory's reward shifts together) |
| Setting | Short trajectories (3–5 turns) | Long trajectories (~45 turns) |
| Benchmark | AppWorld / BFCL | SWE-bench |

### Key issue: V5's limitation

V5's edit_streak penalty can only lower the entire trajectory's advantage; it cannot distinguish good and bad turns within the same trajectory:

```
Trajectory: search → edit → edit → edit (no test) → test → CR=1.0

Standard GRPO:  every turn advantage = +0.45
V5:             every turn advantage = +0.43 (slight global drop)
SeeUPO:         turn 5 (test):   advantage = +0.60 (directly drove success)
                turn 4 (edit):   advantage = +0.20 (useful but not decisive)
                turns 2–3 (edit): advantage = -0.05 (consecutive edit without test → negative credit)
                turn 1 (search): advantage = +0.30 (good start)
```

### Best adaptation for SWE: Per-Turn Advantage Weighting

**Why we don't copy SeeUPO directly**:
- SeeUPO's reverse-order update needs T separate gradient updates; SWE-bench averages 45 turns → 45 forward + backward passes → ×45 training overhead
- SeeUPO's benchmarks are 3–5 turns where the overhead is acceptable; at 45 turns, both compute cost and effectiveness are unverified
- The ROLL framework does not support turn-by-turn independent batch construction

**Our adaptation: use step-level signals to do per-turn advantage weighting**

Idea: do not change the training loop; only adjust each turn's token advantage via the per-turn step-level signal.

```
Standard GRPO:
  advantage[i, t] = (R[i] - R̄) / σ                # same for every token in a trajectory

Per-Turn Advantage Weighting:
  advantage[i, t] = (R[i] - R̄) / σ × w[turn_t]    # different turns get different weights

w[turn_t] design:
  - turn_t is an edit immediately followed by a test → w > 1 (good behavior, amplify)
  - turn_t is an edit with no following test          → w < 1 (bad behavior, attenuate)
  - turn_t is a test                                  → w > 1 (verification, amplify)
  - turn_t is pure search                             → w ≈ 1 (neutral)
```

**Implementation path**:
1. `env.py`'s `step()` already tracks the per-turn `turn_type` (has_edit, has_test, has_search)
2. In `obtain_outcome_reward()`, also compute a `per_turn_weight` list alongside `episode_score`
3. In the advantage computation inside MCPSweEnvManager, use `per_turn_weight` to adjust each turn's token advantage
4. No training-loop changes needed — only post-processing of advantages

**Advantages**:
- Compute overhead is essentially zero (just multiply the advantage by a weight)
- Distinguishes good and bad turns within the same trajectory
- Composes naturally with V5's step-level signal system
- No ×T extra backward passes

### Roadmap

| Stage | Version | Change | Goal |
|-------|---------|--------|------|
| Done | v4 | Binary CR + SBE/ETT + StdGroupFilter EMA Top-p | Current best BC% ≈ 35% |
| Rolled back | v7 | v4 baseline + filtered training data (1958 records) | exploration collapse |
| Current | v8 | v4 baseline + instruction-following LLM Judge (27B) | Use the 27B judge to inject within-group variance and revive ~45% of dead data |
| Near term | v9 | TBD by v8 results | Optional: batch-level normalization / more LLM-judge dimensions |
| Mid term | v10+ | per-turn advantage weighting | Precise per-turn credit assignment |

### Three risks of SeeUPO under SWE

1. **Long-trajectory overhead**: 45 independent gradient updates per trajectory = ×45 training time, possibly unacceptable
2. **Advantage estimation under sparse reward is hard**: SWE-bench only has the final CR — no instant rewards on intermediate turns; under reverse-order updates the early-turn advantage estimates have very high variance
3. **StdGroupFilter compatibility**: our group filter relies on episode-level reward std; per-turn updates would change the reward structure

---

## 14. V8 Design Details

### V4 → V8 core changes

| Item | V4 | V8 | Why |
|------|----|----|-----|
| CR shape | 0 / 0.1 / 1.0 | **unchanged** | Binary CR works |
| SBE/ETT | +0.1 / +0.15 | **kept** (true) | V8 does not change v4's behavior rewards; it only adds IF |
| instruction_following | none | **LLM Judge (27B)** | Core addition: inject within-group variance |
| final_reward computation | direct CR assignment | **Weighted sum**: `1.0*CR + 0.1*IF` | Coefficients are tunable |
| StdGroupFilter | EMA Top-p 0.9 | **unchanged** | |
| Global whitening | false | **unchanged** | |
| add_token_penalty | true | **true** (kept from v4) | V8 builds on v4, not v6 |

### V8 core mechanism

**Problem**: V4 has ~50% dead data (every rollout in a GRPO group has the same reward → variance=0 → advantage=0 → zero gradient).

**Solution**: add an instruction_following dimension; use a 27B reward model to score each trajectory's instruction-following degree (0.0–1.0), creating variance among trajectories with the same correctness.

**Math**:
```
# v4: final_reward = correctness_reward
# v8: final_reward = 1.0 * correctness_reward + 0.1 * instruction_following_reward

# Example: 8 trajectories in a group, all CR=0.1
# v4: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] → variance=0 → dead data
# v8: [0.19, 0.19, 0.14, 0.19, 0.19, 0.17, 0.19, 0.14] → variance>0 → alive!
```

**Backtest** (v4_1 data, 5 train steps):
- Total GRPO groups: 40
- CR-dead groups: 20 (50%)
- IF can rescue: 9 (45% of dead)
- Effective training data: 50% → ~72%

### V8 instruction-following rules

```python
# Currently a single rule
instruction_following_requirements = {
    "no_unnecessary_md_creation": (
        "When the user's query does not explicitly request markdown files, "
        "never create extra markdown files. If the user explicitly requests "
        "markdown, create them according to the user's specification."
    )
}
```

LLM-Judge flow:
1. Build the prompt (`instruction_following_judge_prompt_template.md` + the conversation history)
2. Call the 27B reward model (`request_reward_model`)
3. Parse the JSON response (per-violation 0/1 scoring)
4. Average → `instruction_following_reward` (0.0–1.0)

### V8 final-reward computation

```python
coefficients = config.final_reward_coefficients  # {"correctness": [1.0, 1.0], "instruction_following": [0.1, 0.1]}
coef_idx = 0 if correctness_reward >= 1.0 else 1  # different weight for correct vs. incorrect
final_reward = correctness_coef * correctness_reward + instruction_following_coef * instruction_following_reward
# Then add SBE/ETT behavior rewards (if enabled)
```

### V8 new files

| File | Worktree path |
|------|--------------|
| InstructionFollowingJudge | `<PROJECT_ROOT>` |
| Prompt template | `<PROJECT_ROOT>` |
| env.py | `<PROJECT_ROOT>` |
| config.py | `<PROJECT_ROOT>` |
| train_swe_v8.yaml | `<PROJECT_ROOT>` |
| train_swe_v8.sh | `<PROJECT_ROOT>` |

### V8 new config fields

```python
# config.py (QwenMCPSweEnvConfig) — added in v8
enable_instruction_following_judge: bool = True   # LLM-Judge switch
final_reward_coefficients: Dict[str, List[float]] = field(
    default_factory=lambda: {
        "correctness": [1.0, 1.0],
        "instruction_following": [0.1, 0.1],
    }
)
INSTRUCTION_FOLLOWING_JUDGE_PROMPT_TEMPLATE: str = ...  # auto-loaded from the template file
```

### V8 launch command

```bash
pip install demjson3
pip install fastmcp
export RAY_gcs_rpc_server_connect_timeout_s=30000
export RAY_py_gcs_connect_timeout_s=60000
cd <PROJECT_ROOT>
bash examples/qwen35-9b-agentic/train_swe_v8.sh
```

---

## 15. Known Experience (Supplement)

- **Worktree must be created with `git worktree add`**; `cp -r` is not allowed — it loses git metadata
- **SBE/ETT are ineffective under GRPO whitening**: ~99% of rollouts get the full bonus, no within-group variance — a constant offset rather than a learning signal. Disabled since v6
- **`penalty` field is safe**: `env.step` returns `penalty=0.0`; in MCPSweEnvManager the token_penalty broadcast yields all zeros, not interfering with advantages. Never write `final_reward` into `turn['penalty']`, or advantages will break
- **V6 git branch**: `exp/9b-v6-edit-streak-penalty`, branched from `exp/9b-v4-binary-cr-behavior-reward`
- **V6 training data**: `swe-rebench-with-catalog-v7-train.9b-sorted.no-easy-update.jsonl`, newer than the v4/v5 training data
- **`add_token_penalty` must be `false`** (under gamma=1.0): when `turn['penalty']` is not written into `final_reward`, token_penalty is all zeros. With `add_token_penalty=true`, the all-zero token_penalty replaces the advantage → zero gradient. Correct path: reward → z-score normalize → EOS token → `compute_reinforce_return(gamma=1)` → broadcast to all tokens → group whitening → non-zero advantage
- **Do not drastically shrink the training data**: shrinking the training set sharply (e.g. to 23%) triggers exploration collapse
- **V8 core idea: use an LLM judge to manufacture variance**: heuristic rewards (SBE/ETT, etc.) become constant signals at 94–99% coverage and are nullified by GRPO whitening. The LLM judge's semantic scoring creates real variance among trajectories with the same correctness, rescuing ~45% of dead data

---

## 16. DSW Environment Notes

### Using TensorBoard inside DSW

1. **Launch command**: `--bind_all` is mandatory (do not use `--host localhost`); otherwise the DSW proxy layer cannot connect
   ```bash
   pkill -f tensorboard 2>/dev/null; sleep 1
   tensorboard --logdir <inner_dir> --port 6006 --bind_all --reload_multifile=true
   ```
2. **`logdir` must point at the inner directory**: ROLL's event files are doubly nested (e.g. `v5/v5/<date>/`); TensorBoard only scans direct children
3. **DSW access methods**:
   - Public IP: `http://<public-ip>:6006` (requires DSW port mapping + opening the security group to your IP)
   - Proxy: `https://<dsw-domain>/proxy/6006/` (no security-group changes needed, but `--bind_all` is required)
4. **Security group**: when using public-IP access, the inbound rule must allow your current public IP (note that the IP may change)
