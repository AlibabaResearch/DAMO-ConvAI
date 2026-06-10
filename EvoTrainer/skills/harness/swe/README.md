# SWE Harness Notes

This directory holds **SWE-specific harness extensions** accumulated during long-horizon agentic RL training on SWE-rebench.

## Relationship to the base harness

The canonical six-dimensional harness in [`../agentic-rl-harness.md`](../agentic-rl-harness.md) was originally distilled from SWE iteration experience and therefore already covers the bulk of SWE-relevant signals (BC%, CR distribution, dead-group ratio, edit/test/search behavior, logprob, fail-to-pass test structure, trajectory sampling).

Anything written under this directory is an **extension on top of** the base harness — typically:

- New failure modes specific to SWE-rebench (e.g. Docker-specific anomalies, environment determinism issues)
- Refined alert thresholds tuned for long trajectories (30–50 turns)
- Auxiliary tools (e.g. `inspect_trajectory.py`) that complement the base health report

## When something lands here

Following the iteration loop in [`../../ANALYSIS_PLAYBOOK.md`](../../ANALYSIS_PLAYBOOK.md) (§2 Step 9 + §3 Harness Evolution Protocol), a SWE-specific extension is recorded here when:

1. A failure mode is observed that the base harness cannot capture
2. A new diagnostic metric is defined and validated retroactively on SWE rollouts
3. A SWE-specific intervention rule is distilled into the rule engine

## Promotion path

If an extension turns out to generalize across ≥2 domains, promote the rule back to [`../agentic-rl-harness.md`](../agentic-rl-harness.md) per the cross-domain transfer protocol described there.

## Current state

*To be populated as SWE training iterates beyond the base harness.*
