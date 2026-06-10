# Domain Skills

👋 This directory holds the training skill documents for the three evaluation domains in the EvoTrainer paper (SWE / Math / Coding) — telling an AI trainer agent how to actually launch RL training on each task.

---

## About the Redaction

All EvoTrainer training experiments were carried out on the team's internal compute and service clusters. The skill documents and accompanying Python scripts that emerged from this work inevitably carry traces of the internal environment — cluster paths, toolchains, ops configurations, and the implementation details of internal analysis tools.

To pass the open-source redaction review, we apply **differential preservation** to the skill files in this directory:

| Document | Treatment | Purpose |
|---|---|---|
| [`swe/training_9b.md`](./swe/training_9b.md) | **Fully preserved** ⭐ | Methodology reference: design decisions, iteration logic, harness evolution |
| [`swe/training_4b.md`](./swe/training_4b.md) | Slimmed down | Just enough to launch training |
| [`math/training_4b.md`](./math/training_4b.md) | Slimmed down | Same as above |
| [`coding/training_4b.md`](./coding/training_4b.md) | Slimmed down | Same as above |

> **Why SWE-9B as the full version?**
> Long-horizon agentic RL training carries the most complex diagnostic dimensions (the six-dimensional harness, three classes of toxic rewards, cross-domain skill reuse, and so on) and therefore offers the richest methodological exposition. Keeping it as the complete reference gives readers the most complete view of EvoTrainer's "self-evolution" mechanism.

---

## What the Slimmed Versions Contain — and Don't

### ✅ Kept

The information an AI trainer agent needs to actually launch training:

- AI trainer action boundaries (what it may / may not modify)
- Project directory layout and key files
- Data format conventions (rollout pkl field semantics)
- Iteration-loop overview
- Final-version reward design and code state
- Key empirical lessons (e.g. how to avoid common reward-hacking patterns)

### ❌ Removed

For the redaction review, the following were stripped:

- Per-version iteration details (the v1 → vN intermediate process)
- Internal cluster paths and ops scripts
- Concrete implementations of internal analysis tools
- Internal repository branch structure and Git workflow
- File references that only make sense in the internal context (experiment trackers, iteration logs, etc.)

---

## Want the Full Methodology?

Open [`swe/training_9b.md`](./swe/training_9b.md). That one keeps:

- The complete version-planning and design-decision chain
- Key diagnostic reasoning and orthogonal experiment design
- Concrete instances of harness evolution and script consolidation
- The provenance of cross-domain reusable skills (e.g. StdGroupFilter)

The methodological principles spelled out in SWE-9B — controlled variables, single-factor interventions, harness self-reflection, and so on — apply equally to the other three domains; only the per-version iteration details are absent from the slimmed versions due to compliance constraints.

---

If you hit missing information or have implementation questions while using these documents, feel free to open an Issue. Thanks for understanding 🙏
