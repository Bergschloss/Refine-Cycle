# Refine Cycle

![Refine Cycle — a self-improvement plugin for Hermes Agent](assets/banner.jpg)

**Self-improvement loop for Hermes Agent** — the agent reads its own trajectory,
finds repeated failures and reusable tactics, then proposes and applies the
**smallest possible edit** to its skills or memory. Every edit is journaled with
a backup, so it can be rolled back in one command.

This is a port of the `/refine` concept from
[Prime Intellect's Prime Agent](https://www.primeintellect.ai/blog/prime-agent)
(Continual Harness) built on top of the Hermes plugin system — no core changes,
pure opt-in plugin.

---

## Why

Modern agent harnesses ship static skills and prompts that never adapt to what
the agent actually learns while running. Refine Cycle closes that loop:

- **Repeated failures** (e.g. the same tool call fails the same way twice) → the
  plugin writes a skill that prevents them.
- **Reusable tactics** (a workflow that worked and the user had to explain) →
  the plugin captures them as a skill or memory entry so the agent doesn't need
  to be re-taught.

The system prompt is never touched. Only **agent-created** skills and memory
entries are editable. Built-in, pinned, and hub-installed skills are off-limits.

---

## How it works

```
trajectory (state.db) → evidence → LLM proposal → guardrails → apply → journal + backup
                                                              ↘ no_op if nothing worthwhile
```

| Stage | What happens |
|---|---|
| **1. Collect evidence** | Reads the last N messages of the session from `~/.hermes/state.db` (read-only), extracts error patterns and user corrections |
| **2. LLM proposal** | Calls the host model with structured output: one minimal `create`/`patch`/`no_op` proposal for a skill or memory |
| **3. Guardrails** | Validates: only agent-created skills, no `delete`, no reserved `hermes-` prefix, size limits, daily budget |
| **4. Apply** | Runs the edit through the standard `skill_manage` / `memory_tool` APIs (approval gate respected) |
| **5. Journal** | Appends a JSONL record with trigger, proposal, outcome and backup path |
| **6. Rollback** | `rollback <id>` restores the pre-edit state from the backup |

### Provider compatibility

The proposal is requested via `json_schema` structured output, with an automatic
fallback to `json_mode` (then raw-text JSON salvage) for providers that reject
`response_format.type=json_schema` — verified on opencode-go ("Console Go").

---

## Installation

> **Note:** this is a plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — it requires a working Hermes installation (≥ 0.17.0) and does not run standalone.

The plugin lives in `~/.hermes/plugins/refine/`.

1. Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - refine
  entries:
    refine:
      llm:
        allow_model_override: false
        allow_provider_override: false
```

2. Restart Hermes (gateway or CLI):

```bash
hermes gateway restart
```

3. Verify:

```
hermes plugins list
# refine  0.1.0  Self-improvement loop ...  user  enabled
```

---

## Usage

### Manual (slash command)

In any Hermes chat:

```
/refine
/refine focus on Gmail API failures
/refine rollback 1f2a3b4c5d6e
```

### Automatic (hook)

Enable in config — the plugin then runs after every session with enough
messages (background thread, never blocks session teardown):

```yaml
plugins:
  entries:
    refine:
      auto_enabled: true      # run refine after each session
      auto_min_messages: 15   # require at least 15 messages
```

### Agent-invocable tool

The agent itself gets a `refine_run` tool (toolset `refine`), so it can trigger
a refinement pass whenever it notices a repeated failure or a reusable tactic.

---

## Configuration

All keys live under `plugins.entries.refine`:

| Key | Type | Default | Description |
|---|---|---|---|
| `auto_enabled` | bool | `false` | Auto-run on `on_session_end` |
| `auto_min_messages` | int | `15` | Min messages for auto-analysis |
| `max_edits_per_run` | int | `1` | Max CRUD edits per single run |
| `max_edits_per_day` | int | `3` | Max edits per day (all triggers) |
| `only_agent_created` | bool | `true` | Only edit agent-created skills |
| `journal_dir` | path | `~/.hermes/plugins/refine` | Journal + backup location |

LLM trust policy (`plugins.entries.refine.llm`):

```yaml
llm:
  allow_model_override: false
  allow_provider_override: false
  allow_agent_id_override: false
```

---

## Rollback

Every applied edit writes a journal entry with a backup:

```
/refine rollback <journal_id>
```

Manual fallback: find the entry in `refine_journal.jsonl`, restore the matching
`.bak` file from `backups/`.

---

## Tests

```bash
cd ~/.hermes/plugins/refine
python3 -m tests.run_tests
```

The suite covers: trajectory collection (real `state.db`, read-only), proposal
parsing + validation, guardrails, journal roundtrip, an end-to-end create→apply→
delete cycle, and rollback error handling. A mock LLM is used — no API tokens spent.

---

## Repository layout

```
refine/
├── plugin.yaml          # Hermes plugin manifest
├── __init__.py          # register(ctx): /refine command, refine_run tool, on_session_end hook
├── config.py            # config.yaml reader (plugins.entries.refine.*)
├── core.py              # evidence collection, guardrails, apply logic, refine_run entry
├── llm.py               # structured LLM proposal + json_mode fallback + validation
├── journal.py           # JSONL journal, backups, rollback
└── tests/
    └── run_tests.py     # self-contained test suite
```

---

## Safety & limits

- **Agent-created skills only** (default). Built-in / pinned / hub-installed are
  never touched.
- **Credential scrubbing** — trajectory fragments are scrubbed
  (PATs, API keys, JWTs, private keys, `token=`/`password=` values → `[REDACTED]`)
  before they are sent to the LLM **and** before they are written to the journal.
- **No delete** — refine can only create or patch.
- **Daily budget** — max 3 applied edits per day (UTC), max 1 per run.
- **Backup before edit, rollback by ID.**
- **No system prompt access** — the base prompt stays immutable.
- **Approval gate respected** — if skill writes are gated, edits stage as
  `pending_approval` instead of being applied.
- **Read-only trajectory** — `state.db` is opened with `mode=ro`.
- Requires Hermes ≥ 0.17.0 (plugin API: `register_tool`, `register_command`,
  `register_hook`, `ctx.llm`).

---

## License

MIT © 2026 Taras Boiko
