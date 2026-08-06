# Refine Cycle

![Refine Cycle — a self-improvement plugin for Hermes Agent](assets/banner.jpg)

**Self-improvement loop for Hermes Agent** — the agent reads its own trajectory,
finds repeated failures and reusable tactics, then proposes and applies the
**smallest possible edit** to its skills or memory. Mutations are prepared in a
durable journal before they run and carry conflict-aware recovery metadata.

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
trajectory (state.db) → scrub → fingerprint + aggregate → signal gate ─┬→ no_op
                                                                      └→ LLM proposal
                                                                         → guardrails + backup
                                                                         → prepared journal record
                                                                         → apply → finalized outcome
                                                                                 → usefulness ledger
```

| Stage | What happens |
|---|---|
| **1. Collect evidence** | Reads the last N messages of the session from `<HERMES_HOME>/state.db` (read-only). Credentials are redacted before downstream use |
| **2. Aggregate** | Normalizes each error to its invariant shape and records a complete 12-character fingerprint, then counts occurrences within and across sessions |
| **3. Signal gate** | No repeated failure and no explicit user correction → `no_op` without calling the model |
| **4. LLM proposal** | Calls the host model with structured output: one minimal `create`/`patch`/`no_op` proposal. Every model-bound field is sanitized. Skill patches get the current complete `SKILL.md` only when it is unchanged by sanitization and at most 15,000 characters; otherwise the patch becomes `no_op` |
| **5. Guardrails** | Validates agent-created patch targets, fresh create names, content/frontmatter, size limits, daily budget, and recent duplicates |
| **6. Prepare** | Creates a durable patch backup or exact append-recovery metadata, then appends and `fsync`s a `prepared` journal record before mutation |
| **7. Apply and reconcile** | Runs the standard host API (`patch` maps to host `edit`), proves immediate writes from actual target state, and records `applied`, `pending_approval`, or `error`. Pending approvals retain their host ID and reconcile lazily before later runs, audit, or rollback |
| **8. Rollback** | Journals `rollback_prepared` before the side effect. Immediate rollback is finalized only after target-state proof; staged rollback remains `pending_rollback` until approval reconciliation |

### Why fingerprinting

"The same failure happened again" is a question about shapes, not strings.
`HTTP 429 for /users/8821` and `HTTP 429 for /users/9134` are one failure, not two.
Normalizing away volatile parts and hashing what remains turns a flat list of
error text into countable patterns.

A pattern that appears in several **different** sessions is a stronger signal
than one repeated twice inside one conversation. Interactive prompts remain
bounded, while `/refine audit` evaluates recurrence over the complete available
post-edit period.

### Provider compatibility

The proposal is requested via `json_schema` structured output, with an automatic
fallback to `json_mode` (then raw-text JSON salvage) for providers that reject
`response_format.type=json_schema`.

---

## Installation

> **Note:** this is a plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — it requires a working Hermes installation (≥ 0.17.0) and does not run standalone.

The plugin lives in `<HERMES_HOME>/plugins/refine/` — `~/.hermes/plugins/refine/`
on Linux and macOS, and `%LOCALAPPDATA%\hermes\plugins\refine\` on Windows. Under
a Hermes profile it follows the profile. The plugin resolves this through
`hermes_constants.get_hermes_home()`.

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

2. Restart Hermes:

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

### Manual

```
/refine
/refine focus on Gmail API failures
/refine audit
/refine rollback 1f2a3b4c5d6e
```

`audit` and `rollback <12-character-id>` are exact subcommands. Other text,
including reasons beginning with those words, is passed to the proposal model as
the manual reason.

### Auditing what refine wrote

`/refine audit` reports whether refine-created entries were used and whether the
failure fingerprint recurred after the edit. Timestamp-aware host counts are
preferred. If the host exposes only an all-time aggregate, the report labels it
`all:` and does not claim post-edit use from it. Pending approvals remain marked
as pending rather than being reported as applied. On the next audit, run, or
rollback request, the plugin checks the host pending store and the actual skill
or memory target: an exact target match becomes applied, an unresolved host
record stays pending, and a removed host record without a target match becomes
rejected.

```
Refine-created entries (3):

  name                           age     uses  recurred  verdict
  gmail-scope-fix                12d        5        no  working
  prisma-migrate-note             9d       ~0         —  too early
  bash-path-hint                  3d        2       yes  did not help

Candidates for removal:
  bash-path-hint — /refine rollback 8c1d2e3f4a5b
```

The audit deletes nothing. It prints a rollback command only for recorded
candidates. Skills that remain unused are fed into later proposals as negative
examples.

### Automatic

Enable the session-end hook in config:

```yaml
plugins:
  entries:
    refine:
      auto_enabled: true
      auto_min_messages: 15
```

Automatic and manual runs share a cross-thread and cross-process mutation lock,
then recheck the daily budget inside that lock.

### Agent-invocable tool

The agent gets a `refine_run` tool (toolset `refine`) and may trigger the same
serialized flow with an optional reason.

---

## Configuration

All keys live under `plugins.entries.refine`:

| Key | Type | Default | Description |
|---|---|---|---|
| `auto_enabled` | bool | `false` | Auto-run on `on_session_end` |
| `auto_min_messages` | int | `15` | Min messages for auto-analysis |
| `max_edits_per_run` | int | `1` | Max applied or reserved edits per run |
| `max_edits_per_day` | int | `3` | Max applied, pending, or prepared edits per UTC day |
| `only_agent_created` | bool | `true` | Only patch agent-created skills |
| `journal_dir` | path | `<HERMES_HOME>/plugins/refine` | Journal, lock, ledger, and backups |
| `min_signal_required` | bool | `true` | Skip the model call when nothing repeated |
| `min_pattern_count` | int | `2` | Repeats before a failure counts as a signal |
| `cross_session_enabled` | bool | `true` | Aggregate failures across recent sessions |
| `cross_session_days` | int | `7` | Interactive cross-session look-back window |
| `cross_session_max_sessions` | int | `25` | Interactive session scan cap |
| `dedup_window_days` | int | `7` | Refuse an edit identical to a recent applied, pending, or prepared edit |

LLM trust policy (`plugins.entries.refine.llm`):

```yaml
llm:
  allow_model_override: false
  allow_provider_override: false
  allow_agent_id_override: false
```

---

## Rollback

A successful mutation returns a rollback command only when its journal record is
actually reversible:

```
/refine rollback <journal_id>
```

Create rollback deletes the skill only if its current content still exactly
matches the refine proposal. Patch rollback likewise refuses to overwrite a
later change before restoring its durable backup. Memory rollback removes only
the exact appended entry and preserves unrelated later entries.

If mutation succeeded but journal finalization failed, the returned recovery ID
points to the durable `prepared` record. Pending forward approvals consume budget
but are not advertised as reversible until the target exactly matches the
proposal. Rollback intent is journaled before its side effect; a staged rollback
returns a pending ID and is not called rolled back until the target change is
confirmed. A rejected rollback returns the entry to `applied`, so it can be
retried.

---

## Tests

```bash
cd <HERMES_HOME>/plugins/refine
python -m tests.run_tests
```

The stdlib-only suite installs an in-memory fake Hermes host before importing the
plugin. Every database, journal, backup, skill, memory file, ledger, and lock
lives under a fresh `TemporaryDirectory`; running the tests cannot touch live
`~/.hermes` or profile state. It covers proposal completion, host action mapping,
backup/journal failures, create/patch/memory rollback conflicts, failed applies,
secret sanitation and idempotent redaction, pending approval/rejection
reconciliation, staged rollback and finalization recovery, concurrent budget
checks and lock initialization races, append-only journal tail recovery,
multipass partial-success IDs, complete patch limits and metadata preservation,
streaming full-history audit aggregation, full fingerprints, command parsing,
error/correction classification, and audit scope. No model API is called.

---

## Repository layout

```
refine/
├── plugin.yaml          # Hermes plugin manifest
├── __init__.py          # command, tool, and session-end hook registration
├── config.py            # plugins.entries.refine config reader
├── core.py              # evidence, guardrails, serialized apply orchestration
├── sanitization.py      # recursive credential redaction
├── patterns.py          # normalization, fingerprints, aggregation, signal gate
├── ledger.py            # timestamp-aware usefulness ledger and audit report
├── llm.py               # structured proposal and complete patch regeneration
├── journal.py           # atomic journal, lock, backups, recovery, rollback
└── tests/
    └── run_tests.py     # hermetic fake-host regression suite
```

---

## What gets sent to the model

Refine sends sanitized aggregated error patterns, explicit correction excerpts,
existing skill and memory names/snippets, the optional manual reason/prior-pass
note, and up to 8000 characters of sanitized recent trajectory to the configured
provider. When a skill patch is selected, a second structured request receives
the target's current complete `SKILL.md` only if it is already safe and no larger
than the shared 15,000-character input/output limit. Unsafe or oversized current
skill content aborts the patch as `no_op`; it is never redacted, truncated, or
used to generate a destructive replacement.

Credentials are redacted first, but the remaining content is ordinary
conversation or skill content. Keep `auto_enabled: false` if model-bound session
analysis must be manually initiated. With the signal gate enabled, no model call
occurs when nothing repeated and no explicit correction was detected.

---

## Safety & limits

- **Credential scrubbing** covers evidence, reasons, proposals, host errors, and
  every recursively nested journal field, including quoted JSON keys and
  punctuation-heavy values.
- **Signal gate** requires a repeated failure or explicit correction.
- **Agent-created skills only** for patches; creates require a free normalized
  name and cannot use the reserved `hermes-` prefix.
- **No autonomous delete** — delete is used only by an explicit rollback of an
  unchanged skill created by refine.
- **Serialized budget** counts applied, pending-approval, and unresolved prepared
  records after acquiring the process-safe mutation lock.
- **Durable append journal** writes one locked, fsynced JSON line per state
  transition without rewriting history. A corrupt trailing line is skipped and
  isolated before the next valid record; backup and ledger replacement writes
  remain atomic.
- **Conflict-aware rollback** preserves later skill and memory changes.
- **Approval gate respected** — staged forward and rollback writes persist their
  pending IDs and are reconciled from both host-pending and exact target state.
- **Read-only trajectory** — `state.db` is opened with `mode=ro`.
- **No system prompt access** — the base prompt stays immutable.
- Requires Hermes ≥ 0.17.0 (`register_tool`, `register_command`, `register_hook`,
  and `ctx.llm`).

---

## License

MIT © 2026 Taras Boiko
