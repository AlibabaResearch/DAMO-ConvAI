# Coding Harness Notes

This directory holds **Coding-specific harness extensions** accumulated during RL training on the competitive-programming domain (LiveCodeBench).

## Relationship to the base harness

All Coding experiments start from the canonical six-dimensional harness defined in [`../agentic-rl-harness.md`](../agentic-rl-harness.md). Most of its dimensions (result metrics, model state, learning-signal health) apply directly to Coding.

Anything written under this directory is an **extension on top of** the base harness — typically:

- Format-gate / truncation diagnostics specific to single-turn code generation
- Coding-specific metrics not yet generalized to all domains (e.g. format violation rate, FmtFail root-cause taxonomy, code-block detection)
- Coding-only intervention rules

## When something lands here

Following the iteration loop in [`../../ANALYSIS_PLAYBOOK.md`](../../ANALYSIS_PLAYBOOK.md) (§2 Step 9 + §3 Harness Evolution Protocol), a Coding-specific extension is recorded here when:

1. A failure mode is observed that the base harness cannot capture
2. A new diagnostic metric is defined and validated retroactively on Coding rollouts
3. A Coding-specific intervention rule is distilled into the rule engine

## Promotion path

If an extension turns out to generalize across ≥2 domains, promote the rule back to [`../agentic-rl-harness.md`](../agentic-rl-harness.md) per the cross-domain transfer protocol described there.

## Current state

*To be populated as Coding training iterates beyond the base harness.*
