# Agent Template — Qwen MCP SWE-rebench

This directory contains agent prompt templates and tool schema for SWE training.

## Redaction Notice

Several files in this directory have been **redacted** for the open-source release.
Due to the organization's external disclosure policy, proprietary prompt templates
and the full tool schema cannot be publicly distributed.

### What was redacted

| File | State |
|---|---|
| `system_prompt_template.md` | Content replaced with placeholder; users must supply their own system prompt |
| `system_prompt_template_alt.md` | Same as above (alternative variant) |
| `user_query_template_linux.md` | Content replaced with placeholder |
| `user_query_template_windows.md` | Content replaced with placeholder |
| `tools.json` | Reduced from 18 tools to 5 generic SWE tools (see below) |

### `tools.json` — kept tools

The released `tools.json` contains only 5 generic file/shell tools:

- `list_dir` — list directory contents
- `read_file` — read file content
- `grep_code` — regex search in code
- `search_replace` — edit files via search-and-replace
- `run_in_terminal` — execute shell commands

These omitted tools are part of the proprietary scaffold and are not released.
To reproduce the full training capability, users must define and register
their own tool implementations matching their target agent scaffold, and
extend `tools.json` accordingly.

### How to use

The 5 retained tools form a minimal, runnable SWE agent scaffold.
The training pipeline (`roll/pipeline/agentic/env/qwen_mcp_swe/`) loads
`tools.json` at runtime via `dataclass.field(default_factory=...)`. As long
as each entry follows the OpenAI `tools` schema (`type=="function"` with
`function.name`, `function.description`, `function.parameters`), training
will run without modification.

Users can extend `tools.json` with additional tools to match richer scaffolds
(e.g., language-server-based static analysis, semantic search, task management).
