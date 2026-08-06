"""Core refine logic: collect trajectory evidence, run guardrails, apply edits.

This module does NOT import PluginContext — it receives a ``PluginLlm``
and works standalone (testable without a live session).
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.plugin_llm import PluginLlm

try:
    from . import config, journal, llm as _llm
except ImportError:
    import config, journal, llm as _llm  # noqa: F811 — standalone test

logger = logging.getLogger(__name__)

# ── trajectory collector ────────────────────────────────────────────────────


def _open_db() -> Optional[sqlite3.Connection]:
    """Open state.db read-only. Returns None on failure."""
    db_path = Path.home() / ".hermes" / "state.db"
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

        for row in reversed(rows):
            role = row["role"] or ""
            content = str(row["content"] or "")[:3000]  # truncate long outputs
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

            # Detect user corrections
            if role == "user" and _is_correction(content):
                user_corrections.append({
                    "snippet": content[:300],
                })

        return {
            "messages": messages[-limit:],
            "error_count": error_count,
            "tool_errors": tool_errors[-10:],
            "user_corrections": user_corrections[-5:],
            "session_id": session_id,
        }
    finally:
        con.close()


def _is_error_content(content: str) -> bool:
    """Heuristic: does this look like a tool error?"""
    lower = content.lower()
    any_hit = any(
        kw in content or kw in lower
        for kw in ["Traceback", "exit_code", " error", "failed", "ERROR", "timeout"]
    )
    return any_hit and len(content) < 2000  # error messages tend to be short


def _is_correction(content: str) -> bool:
    """Heuristic: is the user correcting the agent?"""
    lower = content.lower()
    correction_phrases = [
        "не ", "no ", "wrong", "неправильно", "fix ", "don't", "do not",
        "use ", "instead", "замість", "correct", "try ", "спробуй",
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

    # Agent-created only guard
    if kind == "skill" and config.only_agent_created():
        try:
            from tools.skill_usage import is_agent_created
            if action == "patch" and not is_agent_created(name):
                return f"Skill '{name}' is not agent-created (denied by only_agent_created)"
        except ImportError:
            return "Cannot import skill_usage module"

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


def refine_run(
    llm: PluginLlm,
    *,
    reason: str = "",
    session_id: Optional[str] = None,
    auto: bool = False,
) -> Dict[str, Any]:
    """Run one refine pass.

    Args:
        llm: PluginLlm instance (from ctx.llm or standalone).
        reason: Optional user-provided reason (for manual /refine).
        session_id: Optional session to analyze (auto-detected from state.db if None).
        auto: True when triggered by on_session_end hook.

    Returns:
        A dict suitable for serialization to show the user or log.
    """
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

    # 3. Get existing skills/memories for context
    skills = list_skill_names()
    memories = list_memory_snippets()

    # 4. Build evidence text for LLM
    ev_lines: List[str] = []
    for m in evidence.get("messages", []):
        role_tag = f"[{m['role']}]"
        if m.get("tool_name"):
            role_tag += f"({m['tool_name']})"
        ev_lines.append(f"{role_tag} {m['content'][:400]}")
    evidence_text = "\n".join(ev_lines)

    # 5. LLM proposal
    proposal = _llm.propose(
        llm=llm,
        evidence_text=evidence_text,
        existing_skills=skills,
        existing_memories=memories,
        purpose="refine",
    )

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
