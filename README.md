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
trajectory (state.db) → scrub → fingerprint + aggregate → signal gate ─┬→ no_op (no model call)
                                                                      └→ LLM proposal → guardrails
                                                                         → apply → journal + backup
                                                                                  → usefulness ledger
```

| Stage | What happens |
|---|---|
| **1. Collect evidence** | Reads the last N messages of the session from `<HERMES_HOME>/state.db` (read-only). Credentials are redacted at this point, so every downstream consumer gets scrubbed text |
| **2. Aggregate** | Normalizes each error to its invariant shape (ids, paths, timestamps stripped) and fingerprints it, then counts occurrences — within this session and across the last 7 days |
| **3. Signal gate** | No failure repeated and no user correction → `no_op` **without calling the model at all** |
| **4. LLM proposal** | Calls the host model with structured output: one minimal `create`/`patch`/`no_op` proposal, grounded in a listed pattern |
| **5. Guardrails** | Validates: agent-created skills only for patches, fresh name for creates, no `delete`, no reserved `hermes-` prefix, size limits, daily budget, no duplicate of a recent edit |
| **6. Apply** | Runs the edit through the standard `skill_manage` / `memory_tool` APIs (approval gate respected) |
| **7. Journal + ledger** | Appends a JSONL record with trigger, proposal, outcome and backup path, and registers the edit for later auditing |
| **8. Rollback** | `rollback <id>` restores the pre-edit state from the backup |

### Why fingerprinting

"The same failure happened again" is a question about shapes, not strings.
`HTTP 429 for /users/8821` and `HTTP 429 for /users/9134` are one failure, not two.
Normalizing away the volatile parts and hashing what remains turns a flat list of
error text into countable patterns — which is what lets the plugin *assert* that
something recurs instead of asking the model to guess from a transcript.

A pattern that appears in several **different** sessions is a much stronger signal
than one repeated twice inside a single conversation, where it is usually just a
retry loop. Both counters are tracked and shown to the model.

### Provider compatibility

The proposal is requested via `json_schema` structured output, with an automatic
fallback to `json_mode` (then raw-text JSON salvage) for providers that reject
`response_format.type=json_schema` — verified on opencode-go ("Console Go").

---

## Installation

> **Note:** this is a plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — it requires a working Hermes installation (≥ 0.17.0) and does not run standalone.

The plugin lives in `<HERMES_HOME>/plugins/refine/` — that is `~/.hermes/plugins/refine/`
on Linux and macOS, and `%LOCALAPPDATA%\hermes\plugins\refine\` on Windows. Under a
Hermes profile it follows the profile. The plugin resolves this itself via
`hermes_constants.get_hermes_home()`, so the journal and the trajectory are always
read from the same place Hermes uses.

1. Add to your Hermes `config.yaml`:

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
/refine audit
/refine rollback 1f2a3b4c5d6e
```

### Auditing what refine wrote

`/refine audit` answers the question the loop otherwise never asks — did any of
this help?

```
Refine-created entries (3):

  name                           age   uses  recurred  verdict
  gmail-scope-fix                12d     ~5        no  working
  prisma-migrate-note             9d     ~0         —  unused
  bash-path-hint                  3d     ~0       yes  did not help

Candidates for removal:
  bash-path-hint — /refine rollback 8c1d2e3f4a5b
```

The `recurred` column re-runs the fingerprint aggregation restricted to the time
*after* the skill was written and checks whether the failure it targeted came
back. That is the honest answer to "did this work?", and it is only possible
because each proposal records the fingerprint it addressed.

The audit deletes nothing. It prints the command; you decide.

Skills that were never used are also fed back into the next proposal as negative
examples, so the model stops writing more of the same shape.

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
| `only_agent_created` | bool | `true` | Only patch agent-created skills |
| `journal_dir` | path | `<HERMES_HOME>/plugins/refine` | Journal + backup location |
| `min_signal_required` | bool | `true` | Skip the model call when nothing repeated |
| `min_pattern_count` | int | `2` | Repeats before a failure counts as a signal |
| `cross_session_enabled` | bool | `true` | Aggregate failures across recent sessions |
| `cross_session_days` | int | `7` | Look-back window for cross-session patterns |
| `cross_session_max_sessions` | int | `25` | Cap on sessions scanned per run |
| `dedup_window_days` | int | `7` | Refuse an edit identical to a recent one |

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
cd <HERMES_HOME>/plugins/refine
python3 -m tests.run_tests
```

The suite covers: trajectory collection (real `state.db`, read-only), credential
scrubbing (including a regression test proving no secret survives into the tool
result), error fingerprinting and aggregation, the signal gate, guardrails and the
dedup guard, journal roundtrip, the usefulness ledger, an end-to-end
create→apply→delete cycle, and rollback error handling. A mock LLM is used — no
API tokens spent, and tests that need message rows build a throwaway SQLite file
rather than touching the real `state.db`.

---

## Repository layout

```
refine/
├── plugin.yaml          # Hermes plugin manifest
├── __init__.py          # register(ctx): /refine command, refine_run tool, on_session_end hook
├── config.py            # config.yaml reader (plugins.entries.refine.*)
├── core.py              # evidence collection, scrubbing, guardrails, apply logic
├── patterns.py          # error normalization, fingerprinting, aggregation, signal gate
├── ledger.py            # usefulness ledger + /refine audit report
├── llm.py               # structured LLM proposal + json_mode fallback + validation
├── journal.py           # JSONL journal, backups, rollback, dedup
└── tests/
    └── run_tests.py     # self-contained test suite
```

---

## What gets sent to the model

Refine reads your session content and sends part of it to whichever LLM provider
Hermes is configured to use. Specifically: aggregated error patterns, quotes from
messages where you corrected the agent, your existing skill and memory names, and
up to 8000 characters of recent trajectory as context.

Credentials are redacted first (see below), but the rest is ordinary conversation
content. If that matters for your setup, keep `auto_enabled: false` and run
`/refine` manually, so nothing leaves the machine unless you asked for it.

The signal gate limits this considerably: when nothing repeated and you corrected
nothing, the run ends as a `no_op` and **no data is sent at all**.

---

## Safety & limits

- **Credential scrubbing** — redaction happens at the single point where rows
  leave `state.db`, so every consumer downstream (the model, the journal, the tool
  result echoed back into context) gets scrubbed text. Covers GitHub/GitLab/Slack/
  OpenAI/Anthropic/HuggingFace/SendGrid/AWS/Google key formats, JWTs, private key
  blocks, `Authorization:` headers, basic-auth URLs, `.env`-style lines, and
  generic `token=`/`password=` values.
- **Signal gate** — no repeated failure and no user correction means no model call.
- **Agent-created skills only** for patches; creates must use a free name, so a
  bundled, pinned or hub-installed skill can never be overwritten.
- **No delete** — refine can only create or patch. `/refine audit` reports
  removal candidates but never acts on them.
- **Daily budget** — max 3 applied edits per day (UTC), max 1 per run.
- **No duplicate edits** — an edit identical to one applied in the last 7 days is refused.
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
