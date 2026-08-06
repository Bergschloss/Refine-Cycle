"""Core refine logic: collect trajectory evidence, run guardrails, apply edits.

This module does NOT import PluginContext — it receives a ``PluginLlm``
and works standalone (testable without a live session).
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.plugin_llm import PluginLlm

try:
    from . import config, journal, ledger, llm as _llm, patterns
except ImportError:
    import config, journal, ledger, llm as _llm, patterns  # noqa: F811 — standalone test

logger = logging.getLogger(__name__)

# ── secret scrubbing ─────────────────────────────────────────────────────────
# Trajectory fragments (tool outputs, user messages) may contain credentials.
# Scrub them before anything leaves the machine: LLM calls AND the journal.

_RED = '[REDACTED]'
_B = 'Bearer'

_SECRET_PATTERNS = [
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "[REDACTED]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED]"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), "[REDACTED]"),
    (re.compile(r"ntn_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"glpat-[A-Za-z0-9_\-]{15,}"), "[REDACTED]"),
    (re.compile(r"SG\.[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{15,}"), "[REDACTED]"),
    (re.compile(r"dop_v1_[a-f0-9]{60,}"), "[REDACTED]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "[REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED]"),
    # Credentials embedded in a URL: https://user:pass@host
    (re.compile(r"(https?://)[^/\s:@]+:[^/\s:@]+@"), r"\1[REDACTED]@"),
    # .env-style line pasted into a message: FOO_API_KEY=value
    (re.compile(r"(?m)^(\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*\s*=\s*)\S+$"), r"\1[REDACTED]"),
    # "Authorization: <B> <token>" / "<B> <token>" (space-separated value)
    (re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9_\-\.\/\+]{4,}"), "Authorization: " + _B + " " + _RED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.\/\+]{8,}"), _B + " " + _RED),
    # Generic key=value. Floor of 6 chars: shorter values are overwhelmingly
    # literals like true/null/1234, and redacting those only destroys evidence.
    (re.compile(r"(?i)(authorization|bearer|api[_-]?key|password|passwd|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-\.\/\+]{6,}"), r"\1=[REDACTED]"),
]


def scrub_text(text: str) -> str:
    """Replace known credential patterns with [REDACTED]."""
    if not text:
        return text
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def scrub_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub all string fields of a proposal dict before journaling."""
    out = dict(proposal)
    for key, val in list(out.items()):
        if isinstance(val, str):
            out[key] = scrub_text(val)
        elif isinstance(val, list):
            out[key] = [scrub_text(v) if isinstance(v, str) else v for v in val]
    return out

# ── trajectory collector ────────────────────────────────────────────────────


def _open_db() -> Optional[sqlite3.Connection]:
    """Open state.db read-only. Returns None on failure."""
    db_path = config.state_db_path()
    if not db_path.is_file():
        logger.warning("state.db not found at %s", db_path)
        return None
    try:
        uri = f"file:{db_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        con.row_factory = sqlite3.Row
        return con
    except Exception as exc:
        logger.warning("Cannot open state.db: %s", exc)
        return None


def _get_recent_session_id(con: sqlite3.Connection) -> Optional[str]:
    """Get the most recently ended session id."""
    try:
        row = con.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None
    except Exception:
        return None


def collect_evidence(session_id: Optional[str] = None, limit: int = 60) -> Dict[str, Any]:
    """Gather recent trajectory from state.db.

    Returns dict with: messages, error_count, tool_errors, user_corrections, session_id.
    """
    con = _open_db()
    if not con:
        return {
            "messages": [],
            "error_count": 0,
            "tool_errors": [],
            "user_corrections": [],
            "session_id": "",
        }

    try:
        if not session_id:
            session_id = _get_recent_session_id(con)

        if not session_id:
            return {
                "messages": [],
                "error_count": 0,
                "tool_errors": [],
                "user_corrections": [],
                "session_id": "",
            }

        rows = con.execute(
            "SELECT role, content, tool_name, timestamp "
            "FROM messages "
            "WHERE session_id = ? AND active = 1 "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

        messages: List[Dict[str, Any]] = []
        error_count = 0
        tool_errors: List[Dict[str, Any]] = []
        user_corrections: List[Dict[str, Any]] = []
        error_items: List[Dict[str, Any]] = []

        for row in reversed(rows):
            role = row["role"] or ""
            # Scrub at the single point where rows leave the DB — every
            # downstream consumer (LLM evidence, journal, tool result echoed
            # back into context) gets the redacted text automatically.
            content = scrub_text(str(row["content"] or ""))[:3000]  # truncate long outputs
            tool_name = row["tool_name"] or ""

            entry: Dict[str, Any] = {
                "role": role,
                "content": content if len(content) <= 400 else content[:400] + "…",
                "tool_name": tool_name,
            }
            messages.append(entry)

            # Detect errors
            if role == "tool" and _is_error_content(content):
                error_count += 1
                tool_errors.append({
                    "tool": tool_name,
                    "snippet": content[:300],
                })
                error_items.append({
                    "tool": tool_name,
                    "content": content,
                    "session_id": session_id,
                    "ts": row["timestamp"] or 0,
                })

            # Detect user corrections
            if role == "user" and _is_correction(content):
                user_corrections.append({
                    "snippet": content[:300],
                })

        return {
            "messages": messages[-limit:],
            "error_count": error_count,
            # tool_errors is the flat pre-aggregation view — kept for
            # compatibility. error_patterns is what callers should read.
            "tool_errors": tool_errors[-10:],
            "error_patterns": patterns.extract_patterns(error_items),
            "user_corrections": user_corrections[-5:],
            "session_id": session_id,
        }
    finally:
        con.close()


def collect_cross_session_patterns(
    days: Optional[int] = None,
    max_rows: int = 4000,
) -> List[Dict[str, Any]]:
    """Aggregate error patterns across recent sessions.

    A failure that recurs in several *different* sessions is a much stronger
    signal than one repeated twice inside a single conversation, where it is
    usually just a retry loop. Only tool rows are read, and the row cap is hard —
    this runs in a background thread at session end and must not turn into a
    full-history scan.
    """
    if not config.cross_session_enabled():
        return []

    window_days = days if days is not None else config.cross_session_days()
    con = _open_db()
    if not con:
        return []

    try:
        since = time.time() - (window_days * 86400)
        rows = con.execute(
            "SELECT session_id, tool_name, content, timestamp "
            "FROM messages "
            "WHERE role = 'tool' AND active = 1 AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (since, max_rows),
        ).fetchall()
    except Exception as exc:
        logger.warning("Cross-session query failed: %s", exc)
        return []
    finally:
        con.close()

    items: List[Dict[str, Any]] = []
    sessions_seen: set = set()
    max_sessions = config.cross_session_max_sessions()

    for row in rows:
        sid = str(row["session_id"] or "")
        if sid and sid not in sessions_seen:
            if len(sessions_seen) >= max_sessions:
                continue
            sessions_seen.add(sid)
        # Scrub here too: the choke-point rule applies to every path out of the DB.
        content = scrub_text(str(row["content"] or ""))[:3000]
        if not _is_error_content(content):
            continue
        items.append({
            "tool": row["tool_name"] or "",
            "content": content,
            "session_id": sid,
            "ts": row["timestamp"] or 0,
        })

    return patterns.extract_patterns(items)


# A successful JSON tool result usually still contains the word "error" — as a
# field name with a null value. Counting those as failures buries the real ones.
_JSON_SUCCESS_MARKERS = (
    '"success": true', '"success":true',
    '"error": null', '"error":null',
    '"error": ""', '"error":""',
    '"ok": true', '"ok":true',
)


def _is_error_content(content: str) -> bool:
    """Heuristic: does this look like a tool error?"""
    if len(content) >= 2000:
        return False  # error messages tend to be short

    lower = content.lower()

    # An explicit success marker outranks any incidental "error" keyword.
    if any(marker in lower for marker in _JSON_SUCCESS_MARKERS):
        return False

    return any(
        kw in content or kw in lower
        for kw in ["Traceback", "exit_code", " error", "failed", "ERROR", "timeout"]
    )


def _is_correction(content: str) -> bool:
    """Heuristic: is the user clearly correcting the agent?

    Deliberately narrow — generic words like "no", "не", "use", "try" match
    almost every message and produce noise. Only explicit correction phrasings
    (plus a minimum length) count as a signal.
    """
    if len(content) < 15:
        return False
    lower = content.lower()
    correction_phrases = [
        "wrong", "неправильно", "не так", "not right", "fix it", "fix the",
        "don't", "do not", "замість", "instead", "correct", "спробуй інакше",
        "try again", "stop", "не треба", "не роби", "не використовуй",
        "use the", "use this", "that's not", "that is not", "перероби",
    ]
    return any(ph in lower for ph in correction_phrases)


# ── existing skills / memories ──────────────────────────────────────────────


def list_skill_names() -> List[str]:
    """Return a list of all loaded skill names."""
    try:
        from tools.skills_tool import skills_list
        raw = skills_list()
        # skills_list returns a JSON string — parse it (dict/list also handled).
        if isinstance(raw, str):
            result = json.loads(raw)
        else:
            result = raw
        if isinstance(result, dict) and "skills" in result:
            return [s.get("name", "") for s in result["skills"]]
        if isinstance(result, list):
            return [s.get("name", "") for s in result]
    except Exception:
        pass
    return []


def list_memory_snippets() -> List[str]:
    """Return short memory snippets for context."""
    try:
        from tools.memory_tool import MemoryStore
        store = MemoryStore()
        store.load_from_disk()
        entries = store.memory_entries + store.user_entries
        return [e[:120] for e in entries[-20:]]
    except Exception:
        return []


# ── guardrails ──────────────────────────────────────────────────────────────


def _unused_skills_safe() -> List[str]:
    """Never let ledger trouble break a refine run."""
    try:
        return ledger.unused_skills()
    except Exception as exc:
        logger.debug("Cannot compute unused skills: %s", exc)
        return []


def refine_audit() -> Dict[str, Any]:
    """Read-only report: did the skills refine wrote actually help?"""
    try:
        current = collect_cross_session_patterns()
    except Exception:
        current = []
    rows = ledger.audit(current)
    return {"success": True, "rows": rows, "report": ledger.format_audit(rows)}


def _validate_proposal(proposal: Dict[str, Any]) -> Optional[str]:
    """Check guardrails. Returns None if OK, or an error reason string."""
    action = proposal.get("action", "no_op")
    if action == "no_op":
        return None  # not an error

    kind = proposal.get("kind", "")
    name = proposal.get("name", "").strip()

    if not name:
        return "Proposal missing name"

    # Size check
    content = proposal.get("content", "")
    if action == "create" and len(content) > 15000:
        return f"Content too large ({len(content)} chars)"

    if action == "patch" and len(content) > 15000:
        return f"Content too large ({len(content)} chars)"

    # Patch may only touch skills the agent itself created — never bundled
    # or hub-installed. Create is protected separately by the already-exists
    # check below (a fresh name can't be a bundled skill).
    if kind == "skill" and action == "patch" and config.only_agent_created():
        try:
            from tools.skill_usage import is_agent_created
            if not is_agent_created(name):
                return (
                    f"Skill '{name}' is bundled/hub-installed "
                    "(denied by only_agent_created)"
                )
        except ImportError:
            return "Cannot import skill_usage module"

    # Create must target a NEW skill name — never overwrite an existing one
    # (bundled, user, or agent-created). Existing skills are patched, not created.
    if kind == "skill" and action == "create" and name in list_skill_names():
        return f"Skill '{name}' already exists — use patch, not create"

    # Refuse an edit identical to one already applied recently
    if journal.was_applied_recently(proposal, config.dedup_window_days()):
        return (
            f"Identical edit already applied within "
            f"{config.dedup_window_days()} day(s)"
        )

    # Don't delete anything
    if action == "delete":
        return "Delete action not allowed from refine"

    # Don't touch skills that start with "hermes-" (reserved)
    if kind == "skill" and name.startswith("hermes-"):
        return f"Skill '{name}' has reserved prefix"

    return None


# ── apply ───────────────────────────────────────────────────────────────────


def _apply_skill(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a skill create/patch via skill_manage."""
    action = proposal["action"]
    name = proposal["name"]
    content = proposal["content"]
    category = proposal.get("category", "")

    from tools.skill_manager_tool import skill_manage

    result_str = skill_manage(
        action=action,
        name=name,
        content=content if content else None,
        category=category if category else None,
    )
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return {"success": False, "error": result_str}


def _apply_memory(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a memory create/patch via memory_tool."""
    action = proposal["action"]
    content = proposal["content"]
    kind = proposal.get("kind", "memory")
    target = "user" if kind == "user" else "memory"

    from tools.memory_tool import MemoryStore

    store = MemoryStore()
    store.load_from_disk()

    if action == "create":
        result = store.add(target, content)
    elif action == "patch":
        # For memory, a 'patch' means replace the oldest/whole entry.
        # We treat the content as a new add (since memory is append-only by nature).
        result = store.add(target, content)
    else:
        return {"success": False, "error": f"Unknown action '{action}' for memory"}

    if result.get("success"):
        store.save_to_disk(target)
    return result


# ── main entry ──────────────────────────────────────────────────────────────


def _refine_once(
    llm: PluginLlm,
    *,
    reason: str = "",
    session_id: Optional[str] = None,
    auto: bool = False,
) -> Dict[str, Any]:
    """Run a single refine pass (one proposal → one edit)."""
    trigger = "auto" if auto else "manual"
    t_start = time.time()

    # 1. Daily limit check
    if journal.daily_limit_reached():
        return {
            "success": False,
            "message": f"Daily edit limit reached ({config.max_edits_per_day()}). "
                       f"Edits today: {journal.count_today_applied()}.",
        }

    # 2. Collect evidence
    evidence = collect_evidence(session_id=session_id)
    sid = evidence.get("session_id", "")

    if len(evidence.get("messages", [])) < 3:
        return {
            "success": True,
            "message": "Not enough messages in this session to analyze.",
            "evidence": evidence,
        }

    # 3. Aggregate failures across this session and the recent window
    error_patterns = patterns.merge_patterns(
        evidence.get("error_patterns", []),
        collect_cross_session_patterns(),
    )
    evidence["error_patterns"] = error_patterns
    corrections = evidence.get("user_corrections", [])

    # 4. Signal gate — a single one-off error teaches nothing generalizable,
    # so do not spend a model call on it.
    if config.min_signal_required() and not patterns.has_signal(
        error_patterns, corrections, min_count=config.min_pattern_count()
    ):
        proposal = {
            "action": "no_op",
            "reason": (
                f"No repeated failure (min {config.min_pattern_count()}x) and no user "
                f"correction in the last {config.cross_session_days()} day(s)."
            ),
        }
        entry_id = journal.log(
            trigger=trigger,
            reason=reason or proposal["reason"],
            session_id=sid,
            proposal=proposal,
            outcome="no_op",
        )
        return {
            "success": True,
            "message": f"No actionable improvement found. {proposal['reason']}",
            "journal_id": entry_id,
            "llm_called": False,
            "evidence": evidence,
        }

    # 5. Get existing skills/memories for context
    skills = list_skill_names()
    memories = list_memory_snippets()

    # 6. Build evidence text for LLM
    ev_lines: List[str] = []
    for m in evidence.get("messages", []):
        role_tag = f"[{m['role']}]"
        if m.get("tool_name"):
            role_tag += f"({m['tool_name']})"
        ev_lines.append(f"{role_tag} {m['content'][:400]}")
    evidence_text = "\n".join(ev_lines)

    # 7. LLM proposal
    proposal = _llm.propose(
        llm=llm,
        evidence_text=evidence_text,
        existing_skills=skills,
        existing_memories=memories,
        error_patterns=error_patterns,
        user_corrections=[c.get("snippet", "") for c in corrections],
        unused_skills=_unused_skills_safe(),
        purpose="refine",
    )

    # Scrub before journaling/returning — credentials must never persist.
    proposal = scrub_proposal(proposal)

    # 6. If no_op, log and return
    if proposal.get("action") == "no_op":
        entry_id = journal.log(
            trigger=trigger,
            reason=reason or proposal.get("reason", ""),
            session_id=sid,
            proposal=proposal,
            outcome="no_op",
        )
        return {
            "success": True,
            "message": f"No actionable improvement found. {proposal.get('reason', '')}",
            "journal_id": entry_id,
            "evidence": evidence,
        }

    # 7. Guardrails
    guardrail_err = _validate_proposal(proposal)
    if guardrail_err:
        entry_id = journal.log(
            trigger=trigger,
            reason=reason,
            session_id=sid,
            proposal=proposal,
            outcome="rejected",
            error=guardrail_err,
        )
        return {
            "success": False,
            "message": f"Proposal rejected by guardrails: {guardrail_err}",
            "journal_id": entry_id,
            "proposal": proposal,
        }

    # 8. Backup
    backup_path = ""
    kind = proposal.get("kind", "skill")
    name = proposal.get("name", "")
    action = proposal.get("action", "")

    if kind == "skill" and action == "patch":
        bp = journal.backup_skill(name)
        backup_path = str(bp) if bp else ""
    elif kind == "memory":
        mem_content = journal.backup_memory("memory")
        if mem_content is not None:
            bp = journal.backups_dir() / f"{int(time.time()*1000)}_memory.bak"
            bp.write_text(mem_content, encoding="utf-8")
            backup_path = str(bp)

    # 9. Apply
    try:
        if kind == "skill":
            result = _apply_skill(proposal)
        elif kind == "memory":
            result = _apply_memory(proposal)
        else:
            result = {"success": False, "error": f"Unknown kind: {kind}"}
    except Exception as exc:
        entry_id = journal.log(
            trigger=trigger,
            reason=reason,
            session_id=sid,
            proposal=proposal,
            outcome="error",
            error=str(exc),
        )
        return {
            "success": False,
            "message": f"Apply failed: {exc}",
            "journal_id": entry_id,
            "proposal": proposal,
        }

    # 10. Journal
    outcome = "applied" if result.get("success") else "error"
    staged = result.get("staged", False)
    if staged:
        outcome = "pending_approval"

    entry_id = journal.log(
        trigger=trigger,
        reason=reason,
        session_id=sid,
        proposal=proposal,
        outcome=outcome,
        backup_path=backup_path,
        error=result.get("error", ""),
    )

    # Register in the usefulness ledger so /refine audit can judge it later.
    if outcome in ("applied", "pending_approval"):
        try:
            ledger.record_edit(proposal, entry_id)
        except Exception as exc:
            logger.warning("Cannot record edit in ledger: %s", exc)

    elapsed = time.time() - t_start
    msg_parts = [
        f"done ({elapsed:.1f}s)",
        f"action={proposal['action']} kind={kind} name={name}",
        f"outcome={outcome}",
    ]
    if result.get("staged"):
        msg_parts.append(f"pending_id={result.get('pending_id', '?')}")
    if result.get("error"):
        msg_parts.append(f"error={result['error'][:100]}")

    return {
        "success": True,
        "message": " | ".join(msg_parts),
        "journal_id": entry_id,
        "proposal": proposal,
        "result": result,
        "backup_path": backup_path,
        "evidence": {
            "session_id": sid,
            "messages": len(evidence.get("messages", [])),
            "errors": evidence.get("error_count", 0),
        },
    }


def refine_run(
    llm: PluginLlm,
    *,
    reason: str = "",
    session_id: Optional[str] = None,
    auto: bool = False,
) -> Dict[str, Any]:
    """Run up to ``max_edits_per_run`` refine passes.

    Each pass proposes one edit; passes stop early on no_op/rejection/failure
    or when the daily edit budget is exhausted. A single pass is returned
    unchanged (backwards compatible); multiple passes are aggregated.
    """
    t_start = time.time()
    runs: List[Dict[str, Any]] = []
    max_runs = max(1, config.max_edits_per_run())
    run_reason = reason

    for _ in range(max_runs):
        if journal.daily_limit_reached():
            break
        result = _refine_once(llm, reason=run_reason, session_id=session_id, auto=auto)
        runs.append(result)
        action = result.get("proposal", {}).get("action")
        applied = bool(result.get("result", {}).get("success"))
        if not result.get("success") or action in (None, "no_op") or not applied:
            break
        # Tell the next pass what was already done in this run so it doesn't
        # re-propose the same edit (saves a wasted LLM call).
        done_name = result.get("proposal", {}).get("name", "")
        done_kind = result.get("proposal", {}).get("kind", "")
        note = f"Already {action}d {done_kind} '{done_name}' in this run — propose something else or no_op."
        run_reason = f"{reason}\n{note}".strip() if reason else note

    if not runs:
        return {
            "success": False,
            "message": f"Daily edit limit reached ({config.max_edits_per_day()}).",
        }

    if len(runs) == 1:
        return runs[0]

    applied_count = sum(1 for r in runs if r.get("result", {}).get("success"))
    last = runs[-1]
    return {
        "success": any(r.get("success") for r in runs),
        "message": f"{len(runs)} pass(es), {applied_count} edit(s) applied ({time.time() - t_start:.1f}s)",
        "journal_id": last.get("journal_id", runs[0].get("journal_id", "")),
        "proposal": last.get("proposal", runs[0].get("proposal", {})),
        "results": runs,
        "evidence": runs[0].get("evidence", {}),
    }


def refine_rollback(entry_id: str) -> Dict[str, Any]:
    """Rollback a previous refine edit by journal id."""
    entry = journal.get_entry(entry_id)
    if not entry:
        return {"success": False, "error": f"Entry {entry_id} not found"}

    proposal = entry.get("proposal", {})
    kind = proposal.get("kind", "skill")

    if kind == "skill":
        return journal.rollback_skill(entry_id)
    elif kind == "memory":
        return journal.rollback_memory(entry_id)
    else:
        return {"success": False, "error": f"Unknown kind for rollback: {kind}"}
