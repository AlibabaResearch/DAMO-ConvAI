# API-Bank → ToolSearcher-Free Benchmark

This directory contributes a **ToolSearcher-free** variant of the API-Bank
evaluation split. The upstream benchmark uses a meta-tool called
`ToolSearcher` that the model must call at runtime to *discover* which real
tool applies to a request. Here we remove `ToolSearcher` entirely and instead
advertise **every real tool's spec directly in the system prompt**, so the
model picks the right tool without any retrieval hop.

The upstream evaluation data is unchanged in substance — same conversations,
same gold calls, same gold returns. This variant only:

1. strips `ToolSearcher` calls, their return values and the leading
   `{"apiCode":"ToolSearcher",...}` spec block from every trajectory, and
2. moves API (tool) outputs into an OpenAI-style `tool` role message
   (upstream rendered them as `system` turns; Qwen-family chat templates
   reject multiple `system` messages).

## Contents

```
scripts/
  convert_apibank.py          harvests a tool catalog + rewrites all levels
  grade_apibank.py            OpenAI-compatible grader with step-replay scoring
  validate_apibank_conversion.py  1,887 checks: only-ToolSearcher-was-removed
data/
  test-data/                  the upstream eval records (HF split)
  processed/                  converted benchmark (catalog, prompts, JSONL)
  results/                    graded outputs + this run's report.html
```

## Data

- **catalog**: 67 real tools harvested from the eight `test-data/*.json`
  files (`ToolSearcher` excluded).
- **level-1.jsonl**: 368 single-turn call tasks (no ToolSearcher in source).
- **level-2.jsonl**: 49 multi-turn dialogues (45 ToolSearcher calls removed).
- **level-3.jsonl**: 50 full multi-call conversations (114 ToolSearcher steps
  removed; trajectories collapsed to the real-tool sequence).
- **level-3-batch{,-icl}.jsonl**: 121 per-step datapoints each (gold calls
  that embed an unparseable list-literal argument are dropped → 131→121).
- **data/results/viewer.html**: trajectory viewer. Serves each graded
  conversation with its gold calls, gold returns and the model's predicted
  call, marking every prediction **SUCCESS / PARTIAL / UNSUCCESSFUL**, with
  filters by level and success state. Run `python3 -m http.server 8123
  --directory data` in this directory and open
  `http://localhost:8123/results/viewer.html`.

## Reproduce

```bash
# 1. convert (re-generates data/processed from data/test-data)
python scripts/convert_apibank.py --test-data data/test-data --out data/processed

# 2. validate that only ToolSearcher + role mapping changed
python scripts/validate_apibank_conversion.py

# 3. grade an OpenAI-compatible model (canonical API-Bank text format)
python scripts/grade_apibank.py \
    --ip <host> --port 8000 --model <model> \
    --level 3 --variant batch --mode text \
    --output data/results/l3b
```

Scoring is *step-replay*: after each predicted call the dialogue advances
using the **gold** call + gold return, so per-step accuracy is independent of
earlier model mistakes. Metrics per call: tool-name exact, args exact, args
partial (≥50% of keys match).

## Results (Qwen/Qwen3.6-35B-A3B-FP8)

| Variant | Datapoints | Gold calls | Tool-name | Args exact |
|---|---|---:|---:|---:|
| Level 1 (given-desc) | 368 | 368 | 91.3% | 78.3% |
| Level 2 (dialogues) | 49 | 64 | 53.1% | 37.5% |
| Level 3 full (convs) | 50 | 131 | 66.4% | 62.6% |
| Level 3 batch (per-step) | 121 | 121 | 86.8% | 70.2% |

Full details, before/after datapoint examples, model outputs and the
conversion-validation evidence are in `data/results/report.html`.

## Validation

`validate_apibank_conversion.py` proves the converted benchmark differs from
upstream **only** in the ToolSearcher part plus the documented role/format
changes:

- **TS-FREE** — no `ToolSearcher` string anywhere in the processed data.
- **GOLD-PRESERVED** — per datapoint, the ordered `gold_tool_calls` equal the
  upstream ordered real-tool calls (ToolSearcher dropped, duplicates collapsed).
- **TEXT-PRESERVED** — after stripping ToolSearcher artefacts and normalising
  `API-Request: [..]->{..}` into (assistant call, tool result) pairs, the
  user/assistant text stream matches upstream byte-for-byte.

All 1,887 checks pass.

## License

See the repository's Apache-2.0 `LICENSE` (top-level and `../LICENSE`).