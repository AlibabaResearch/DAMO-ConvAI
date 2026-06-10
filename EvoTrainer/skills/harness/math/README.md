# Math Harness Notes

This directory holds **Math-specific harness extensions** accumulated during RL training on the math reasoning domain (AIME / CNMO).

## Relationship to the base harness

All Math experiments start from the canonical six-dimensional harness defined in [`../agentic-rl-harness.md`](../agentic-rl-harness.md). Most of its dimensions (result metrics, model state, learning-signal health) apply directly to Math, even though they were originally distilled from agentic SWE training.

Anything written under this directory is an **extension on top of** the base harness — typically:

- New failure modes specific to math reasoning (e.g. length explosion in extended CoT, length-gaming reward hacks)
- Math-specific metrics not yet generalized to all domains (e.g. format compliance, length-vs-reward correlation, KL effectiveness)
- Math-only intervention rules

## When something lands here

Following the iteration loop in [`../../ANALYSIS_PLAYBOOK.md`](../../ANALYSIS_PLAYBOOK.md) (§2 Step 9 + §3 Harness Evolution Protocol), a Math-specific extension is recorded here when:

1. A failure mode is observed that the base harness cannot capture
2. A new diagnostic metric is defined and validated retroactively on Math rollouts
3. A Math-specific intervention rule is distilled into the rule engine

## Promotion path

If an extension turns out to generalize across ≥2 domains, promote the rule back to [`../agentic-rl-harness.md`](../agentic-rl-harness.md) per the cross-domain transfer protocol described there.

## Current state

*To be populated as Math training iterates beyond the base harness.*
