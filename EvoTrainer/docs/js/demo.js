/**
 * EvoTrainer Demo — Terminal typewriter animation
 * Shows SWE-9B v0→v8 autonomous evolution with deep diagnostic reasoning.
 */

const SCENES = [
  {
    lines: [
      { text: "$ evotrainer diagnose --version v0 --domain swe-9b\n\n", cls: "t-dim" },
      { text: "[EvoTrainer] ", cls: "t-label" },
      { text: "Loading 308 validation rollouts (77 tasks × 4)...\n\n", cls: "" },
      { text: "[Score]    ", cls: "t-label" },
      { text: "BC% = 30.19 | No RL training applied.\n", cls: "" },
      { text: "[Signal]   ", cls: "t-label" },
      { text: "No reward signal — model relies purely on pre-training.\n", cls: "t-dim" },
      { text: "[Behavior] ", cls: "t-label" },
      { text: "Avg turns = 35.7 | First-edit turn = 4.2 (reasonable)\n", cls: "" },
      { text: "           Tool usage: grep(38%) → edit(31%) → test(12%)\n\n", cls: "t-dim" },
      { text: "Diagnosis: ", cls: "t-decision" },
      { text: "Pre-training policy shows basic SWE patterns but no\n", cls: "" },
      { text: "           optimization signal. Ready for RL initialization.\n", cls: "" },
      { text: "Action:    ", cls: "t-decision" },
      { text: "Initialize GSPO with binary correctness reward (0/1).\n", cls: "" },
      { text: "\n→ ", cls: "" },
      { text: "v1 trained. BC% = 31.04 (+0.85)", cls: "t-success" },
    ]
  },
  {
    lines: [
      { text: "$ evotrainer diagnose --version v1 v2 v3 --trajectory\n\n", cls: "t-dim" },
      { text: "[EvoTrainer] ", cls: "t-label" },
      { text: "Analyzing score-dominant iteration path.\n\n", cls: "" },
      { text: "[Score]    ", cls: "t-label" },
      { text: "v1=31.04 → v2=32.89 → v3=33.33 (diminishing returns)\n", cls: "" },
      { text: "[Signal]   ", cls: "t-label" },
      { text: "Dead Group Ratio: v1=55% → v2=52% → v3=50%\n", cls: "" },
      { text: "           Half of all rollout groups produce ", cls: "t-dim" },
      { text: "zero gradient", cls: "t-number" },
      { text: ".\n", cls: "" },
      { text: "           These groups all-pass or all-fail — no relative signal.\n\n", cls: "t-dim" },
      { text: "⚠ Saturation detected: ", cls: "t-star" },
      { text: "3 recipe changes yield <1pp each.\n", cls: "" },
      { text: "  The reward (flat CR) cannot distinguish behaviors within\n", cls: "" },
      { text: "  groups that share the same pass/fail outcome.\n\n", cls: "" },
      { text: "Diagnosis: ", cls: "t-decision" },
      { text: "Score-dominant iteration exhausted. The bottleneck is\n", cls: "" },
      { text: "           reward dimensionality, not algorithm choice.\n", cls: "" },
      { text: "Next step: ", cls: "t-decision" },
      { text: "Analyze rollout BEHAVIOR to find untapped signals.", cls: "t-success" },
    ]
  },
  {
    lines: [
      { text: "$ evotrainer analyze-rollouts --version v3 --focus behavior\n\n", cls: "t-dim" },
      { text: "[EvoTrainer] ", cls: "t-label" },
      { text: "Inspecting 40 failed rollouts with BC%=0...\n\n", cls: "" },
      { text: "[Pattern 1] ", cls: "t-label" },
      { text: "\"Blind edit\" — agent edits code WITHOUT reading context:\n", cls: "" },
      { text: "  rollout #037: ", cls: "t-dim" },
      { text: "edit→edit→edit→submit (no grep/cat before first edit)\n", cls: "t-number" },
      { text: "  rollout #112: ", cls: "t-dim" },
      { text: "edit→submit (single edit, no search at all)\n", cls: "t-number" },
      { text: "  → Hypothesis: ", cls: "t-decision" },
      { text: "reward +0.1 for search-before-edit (SBE)\n\n", cls: "" },
      { text: "[Pattern 2] ", cls: "t-label" },
      { text: "\"Edit without verify\" — agent never tests after editing:\n", cls: "" },
      { text: "  rollout #054: ", cls: "t-dim" },
      { text: "grep→edit→edit→edit→submit (never runs tests)\n", cls: "t-number" },
      { text: "  rollout #089: ", cls: "t-dim" },
      { text: "search→edit→submit (skips test, patch has syntax error)\n", cls: "t-number" },
      { text: "  → Hypothesis: ", cls: "t-decision" },
      { text: "reward +0.15 for edit-then-test (ETT)\n\n", cls: "" },
      { text: "[Backtest] ", cls: "t-label" },
      { text: "Re-scoring v3 rollouts with SBE+ETT...\n", cls: "" },
      { text: "           Groups with zero CR variance: ", cls: "t-dim" },
      { text: "38% regain non-zero variance", cls: "t-success" },
      { text: "\n           under the new reward dimensions.\n", cls: "t-dim" },
      { text: "\nAction:    ", cls: "t-decision" },
      { text: "Deploy v4 with Binary CR + SBE + ETT + EMA filter.", cls: "t-success" },
    ]
  },
  {
    lines: [
      { text: "$ evotrainer diagnose --version v5 v6 v7 --compare-to v4\n\n", cls: "t-dim" },
      { text: "[EvoTrainer] ", cls: "t-label" },
      { text: "v4 = 36.30 BC%. Evaluating 3 exploratory branches:\n\n", cls: "" },
      { text: "  Branch v5: ", cls: "t-dim" },
      { text: "+edit_streak penalty + soft truncation\n", cls: "" },
      { text: "    Result:  BC% = ", cls: "t-dim" },
      { text: "30.82", cls: "t-fail" },
      { text: " (-5.48)\n", cls: "t-fail" },
      { text: "    Root cause: penalty field written into advantage tensor,\n", cls: "t-dim" },
      { text: "    corrupting GRPO normalization. ", cls: "t-dim" },
      { text: "REJECTED.\n\n", cls: "t-fail" },
      { text: "  Branch v6: ", cls: "t-dim" },
      { text: "edit_streak (penalty sync fixed)\n", cls: "" },
      { text: "    Result:  BC% = ", cls: "t-dim" },
      { text: "33.01", cls: "t-fail" },
      { text: " (-3.29)\n", cls: "t-fail" },
      { text: "    Root cause: penalty suppresses legitimate multi-edit\n", cls: "t-dim" },
      { text: "    workflows (e.g., incremental refactoring). ", cls: "t-dim" },
      { text: "REJECTED.\n\n", cls: "t-fail" },
      { text: "  Branch v7: ", cls: "t-dim" },
      { text: "data downsizing (8622→1958 via LLM filter)\n", cls: "" },
      { text: "    Result:  BC% = ", cls: "t-dim" },
      { text: "31.55", cls: "t-fail" },
      { text: " (-4.75)\n", cls: "t-fail" },
      { text: "    Root cause: reduced training diversity causes policy\n", cls: "t-dim" },
      { text: "    to collapse onto narrow solution templates. ", cls: "t-dim" },
      { text: "REJECTED.\n\n", cls: "t-fail" },
      { text: "Decision:  ", cls: "t-decision" },
      { text: "All 3 stored as negative evidence. Revert to v4 baseline.", cls: "" },
    ]
  },
  {
    lines: [
      { text: "$ evotrainer diagnose --version v4 --deep-signal\n\n", cls: "t-dim" },
      { text: "[EvoTrainer] ", cls: "t-label" },
      { text: "v4 = 36.30 BC%. Analyzing remaining bottleneck.\n\n", cls: "" },
      { text: "[Signal]   ", cls: "t-label" },
      { text: "Dead Group Ratio still = ", cls: "" },
      { text: "50.0%", cls: "t-number" },
      { text: "\n", cls: "" },
      { text: "           Problem: binary CR (pass/fail) cannot distinguish\n", cls: "t-dim" },
      { text: "           trajectories where ALL 8 rollouts share same outcome.\n\n", cls: "t-dim" },
      { text: "[Hypothesis] ", cls: "t-label" },
      { text: "An instruction-following dimension might separate\n", cls: "" },
      { text: "             trajectories that CR alone collapses.\n\n", cls: "" },
      { text: "[Backtest]   ", cls: "t-label" },
      { text: "Re-scoring v4 dead groups with +0.1 IF reward...\n", cls: "" },
      { text: "             40 dead groups examined → ", cls: "t-dim" },
      { text: "18 regain variance (45%)", cls: "t-number" },
      { text: "\n", cls: "" },
      { text: "             These 18 groups can now produce useful gradients.\n\n", cls: "t-dim" },
      { text: "[Validation] ", cls: "t-label" },
      { text: "IF scores show low correlation with CR (r=0.12),\n", cls: "" },
      { text: "             confirming orthogonal signal axis.\n\n", cls: "" },
      { text: "Action:    ", cls: "t-decision" },
      { text: "Deploy IF LLM Judge (Qwen3-27B), weight = 0.1\n\n", cls: "" },
      { text: "→ v8 trained: BC% = ", cls: "t-dim" },
      { text: "38.16", cls: "t-number" },
      { text: " | DGR = ", cls: "" },
      { text: "27.5%", cls: "t-number" },
      { text: " (halved)\n\n", cls: "" },
      { text: "★ ", cls: "t-star" },
      { text: "EvoTrainer final: 38.16% vs Human-RL 33.77% ", cls: "t-star" },
      { text: "(+4.39 BC%)", cls: "t-success" },
    ]
  }
];

// --- State ---
let currentScene = 0;
let typing = false;
let autoPlay = false;
let autoTimer = null;

const output = document.getElementById("terminal-output");
const indicator = document.getElementById("scene-indicator");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnPlay = document.getElementById("btn-play");

function updateIndicator() {
  indicator.textContent = `Scene ${currentScene + 1} / ${SCENES.length}`;
}

function clearTerminal() {
  output.innerHTML = '<span class="cursor">_</span>';
}

async function typeScene(sceneIdx) {
  if (typing) return;
  typing = true;
  clearTerminal();
  currentScene = sceneIdx;
  updateIndicator();

  const scene = SCENES[sceneIdx];
  const cursor = output.querySelector(".cursor");

  for (const segment of scene.lines) {
    for (let i = 0; i < segment.text.length; i++) {
      const span = document.createElement("span");
      span.className = segment.cls;
      span.textContent = segment.text[i];
      output.insertBefore(span, cursor);
      const ch = segment.text[i];
      const delay = ch === "\n" ? 60 : (ch === " " ? 10 : 15);
      await sleep(delay);
    }
    await sleep(30);
  }
  typing = false;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

btnPrev.addEventListener("click", () => {
  if (typing) return;
  stopAuto();
  typeScene((currentScene - 1 + SCENES.length) % SCENES.length);
});

btnNext.addEventListener("click", () => {
  if (typing) return;
  stopAuto();
  typeScene((currentScene + 1) % SCENES.length);
});

btnPlay.addEventListener("click", () => {
  if (autoPlay) { stopAuto(); } else { startAuto(); }
});

function startAuto() {
  autoPlay = true;
  btnPlay.innerHTML = "&#9724; Stop";
  advanceAuto();
}

function stopAuto() {
  autoPlay = false;
  btnPlay.innerHTML = "&#9654; Auto";
  if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
}

function advanceAuto() {
  if (!autoPlay) return;
  typeScene((currentScene + 1) % SCENES.length).then(() => {
    if (autoPlay) { autoTimer = setTimeout(advanceAuto, 3500); }
  });
}

typeScene(0);
