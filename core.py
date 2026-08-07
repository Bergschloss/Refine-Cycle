"""Core refine orchestration: evidence, guardrails, durable apply, rollback."""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.plugin_llm import PluginLlm

try:
    from . import config, journal, ledger, llm as _llm, patterns
    from .sanitization import sanitize, scrub_text
except ImportError:
    import config, journal, ledger, llm as _llm, patterns  # noqa: F811
    from sanitization import sanitize, scrub_text  # noqa: F811

logger = logging.getLogger(__name__)

# ── session identity ───────────────────────────────────────────────────────
# The host does not pass session_id to slash-command handlers (contract is
# fn(raw_args) -> str|None). But pre_llm_call and post_llm_call hooks do
# receive it every turn. This module remembers the last value seen, so that a
# manual /refine command running in the same process can resolve it.

_LAST_SESSION_ID = ""
_LAST_SESSION_LOCK = threading.Lock()


def note_session_id(session_id: str) -> None:
    """Record the session id seen from a host hook. Thread-safe, one value."""
    global _LAST_SESSION_ID
    if not isinstance(session_id, str) or not session_id.strip():
        return
    clean = session_id.strip()
    # Reject anything that scrubbing would alter — it might be content, not an id.
    if scrub_text(clean) != clean or len(clean) > 128:
        return
    with _LAST_SESSION_LOCK:
        _LAST_SESSION_ID = clean


def _noted_session_id() -> str:
    with _LAST_SESSION_LOCK:
        return _LAST_SESSION_ID


def host_session_id() -> str:
    """Best-effort read of the host's current session id via ContextVar/env.

    Available in CLI and cron; returns "" in the gateway (which sets session_key,
    not session_id, into the context). Guarded: any failure → "".
    """
    try:
        from gateway.session_context import get_session_env
        value = get_session_env("HERMES_SESSION_ID", "")
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        return ""


def resolve_session_id(explicit: str = "") -> Tuple[str, str]:
    """Resolve which session to analyse.

    Returns (session_id, how) where how ∈ {explicit, host_env, hook, unknown}.
    When unknown, the caller must refuse rather than guess.
    """
    if explicit and explicit.strip():
        return explicit.strip(), "explicit"
    env_id = host_session_id()
    if env_id:
        return env_id, "host_env"
    hook_id = _noted_session_id()
    if hook_id:
        return hook_id, "hook"
    return "", "unknown"


def scrub_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for recursive shared sanitation."""
    return sanitize(proposal)


# ── trajectory collection ──────────────────────────────────────────────────


def _open_db() -> Optional[sqlite3.Connection]:
    path = config.state_db_path()
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    except Exception as exc:
        logger.warning("Cannot open state.db: %s", exc)
        return None


def _structured_error_status(content: str) -> Optional[bool]:
    """Return a definitive structured status, or None when text is unstructured."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict):
        exit_values = [
            value[key]
            for key in ("exit_code", "returncode", "return_code")
            if key in value
            and isinstance(value[key], (int, float))
            and not isinstance(value[key], bool)
        ]
        if any(code != 0 for code in exit_values):
            return True
        error = value.get("error")
        if error not in (None, "", False, [], {}):
            return True
        if value.get("success") is False or value.get("ok") is False:
            return True
        if exit_values and all(code == 0 for code in exit_values):
            return False
        if value.get("success") is True or value.get("ok") is True:
            return False

    codes = [
        int(match)
        for match in re.findall(
            r"(?i)(?:\bexit[_ ]?code\b|\breturncode\b)\s*[:=]?\s*(-?\d+)",
            content,
        )
    ]
    if any(code != 0 for code in codes):
        return True
    return False if codes else None


def _is_error_content(content: str) -> bool:
    """Classify structured status first, then bounded head/tail error text."""
    if not content:
        return False
    structured = _structured_error_status(content)
    if structured is not None:
        return structured
    sample = (
        content
        if len(content) <= 4000
        else content[:1000] + "\n…\n" + content[-3000:]
    )
    sample = re.sub(r'(?i)["\']?error["\']?\s*:\s*(?:null|""|\'\')', "", sample)
    return bool(
        re.search(
            r"(?i)(?:^|[\s\[{(,:;])(?:traceback|error\b|failed\b|failure\b|timed?\s*out\b|timeout\b)",
            sample,
        )
    )


def _is_correction(content: str) -> bool:
    """Recognize explicit agent corrections, not routine instructions."""
    if len(content.strip()) < 12:
        return False
    text = re.sub(r"\s+", " ", content.strip().lower())
    strong = (
        r"\b(?:that(?:'s| is) (?:wrong|not right)|you (?:are|were) wrong|wrong answer|incorrect)\b",
        r"\b(?:неправильно|це не так|ти помилив|ви помилили)\b",
        r"^(?:no|ні|нет)[,;:]\s+.{0,100}\b(?:wrong|not right|не так|неправильно|instead|замість)\b",
        r"\b(?:you used|ти використав|ви використали)\b.{0,120}\b(?:use|instead|замість)\b",
    )
    return any(re.search(pattern, text) for pattern in strong)


def collect_evidence(session_id: Optional[str] = None, limit: int = 60) -> Dict[str, Any]:
    empty = {
        "messages": [],
        "error_count": 0,
        "tool_errors": [],
        "error_patterns": [],
        "user_corrections": [],
        "session_id": "",
        "session_id_source": "unknown",
    }
    resolved, how = resolve_session_id(session_id or "")
    if not resolved:
        empty["session_id_source"] = how
        return empty
    connection = _open_db()
    if not connection:
        empty["session_id"] = resolved
        empty["session_id_source"] = how
        return empty
    try:
        rows = connection.execute(
            "SELECT role, content, tool_name, timestamp FROM messages "
            "WHERE session_id = ? AND active = 1 ORDER BY timestamp DESC LIMIT ?",
            (resolved, limit),
        ).fetchall()
        messages: List[Dict[str, Any]] = []
        tool_errors: List[Dict[str, Any]] = []
        corrections: List[Dict[str, Any]] = []
        error_items: List[Dict[str, Any]] = []
        for row in reversed(rows):
            role = str(row["role"] or "")
            content = scrub_text(str(row["content"] or ""))
            tool_name = str(row["tool_name"] or "")
            shown = content[:400] + ("…" if len(content) > 400 else "")
            messages.append({"role": role, "content": shown, "tool_name": tool_name})
            if role == "tool" and _is_error_content(content):
                bounded = (
                    content
                    if len(content) <= 4000
                    else content[:1000] + "\n…\n" + content[-3000:]
                )
                tool_errors.append({"tool": tool_name, "snippet": bounded[:300]})
                error_items.append({
                    "tool": tool_name,
                    "content": bounded,
                    "session_id": session_id,
                    "ts": row["timestamp"] or 0,
                })
            if role == "user" and _is_correction(content):
                corrections.append({"snippet": content[:300]})
        return {
            "messages": messages[-limit:],
            "error_count": len(tool_errors),
            "tool_errors": tool_errors[-10:],
            "error_patterns": patterns.extract_patterns(error_items),
            "user_corrections": corrections[-5:],
            "session_id": resolved,
            "session_id_source": how,
        }
    finally:
        connection.close()


def collect_cross_session_patterns(
    days: Optional[int] = None,
    max_rows: Optional[int] = 4000,
    *,
    since_ts: Optional[float] = None,
    max_sessions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not config.cross_session_enabled():
        return []
    connection = _open_db()
    if not connection:
        return []
    since = (
        since_ts
        if since_ts is not None
        else time.time() - ((days or config.cross_session_days()) * 86400)
    )
    sql = (
        "SELECT session_id, tool_name, content, timestamp FROM messages "
        "WHERE role = 'tool' AND active = 1 AND timestamp >= ? ORDER BY timestamp DESC"
    )
    params: List[Any] = [since]
    if max_rows is not None:
        sql += " LIMIT ?"
        params.append(max_rows)
    try:
        cursor = connection.execute(sql, tuple(params))
        session_cap = (
            config.cross_session_max_sessions()
            if max_sessions is None and since_ts is None
            else max_sessions
        )
        seen: set = set()

        def iter_items():
            for row in cursor:
                sid = scrub_text(str(row["session_id"] or ""))
                if sid and sid not in seen:
                    if session_cap is not None and len(seen) >= session_cap:
                        continue
                    seen.add(sid)
                content = scrub_text(str(row["content"] or ""))
                if not _is_error_content(content):
                    continue
                bounded = (
                    content
                    if len(content) <= 4000
                    else content[:1000] + "\n…\n" + content[-3000:]
                )
                yield {
                    "tool": scrub_text(str(row["tool_name"] or "")),
                    "content": bounded,
                    "session_id": sid,
                    "ts": row["timestamp"] or 0,
                }

        full_audit = since_ts is not None and max_rows is None and max_sessions is None
        return patterns.extract_patterns(
            iter_items(), limit=None if full_audit else 10
        )
    except Exception as exc:
        logger.warning("Cross-session query failed: %s", scrub_text(str(exc)))
        return []
    finally:
        connection.close()


# ── host context ───────────────────────────────────────────────────────────


def _skill_items() -> List[Any]:
    """Read the host's one skill listing without opening individual skills."""
    try:
        from tools.skills_tool import skills_list

        raw = skills_list()
        result = raw if not isinstance(raw, str) else json.loads(raw)
        skills = result.get("skills", []) if isinstance(result, dict) else result
        return skills if isinstance(skills, list) else []
    except Exception:
        return []


def list_skill_names() -> List[str]:
    names: List[str] = []
    for item in _skill_items():
        raw_name = item.get("name", "") if isinstance(item, dict) else item
        name = scrub_text(str(raw_name)).strip()
        if name:
            names.append(name)
    return names


def list_skill_entries() -> List[Dict[str, Any]]:
    """Return safe host metadata with a local version when the ledger knows it."""
    try:
        stats = ledger.load_stats()
    except Exception:
        stats = {}
    entries: List[Dict[str, Any]] = []
    for item in _skill_items():
        raw_name = item.get("name", "") if isinstance(item, dict) else item
        name = scrub_text(str(raw_name)).strip()
        if not name:
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "description": scrub_text(str(item.get("description", ""))).strip()
            if isinstance(item, dict)
            else "",
            "category": scrub_text(str(item.get("category", ""))).strip()
            if isinstance(item, dict)
            else "",
        }
        metadata = stats.get(name) if isinstance(stats, dict) else None
        if isinstance(metadata, dict):
            try:
                version = int(metadata.get("version", 0) or 0)
            except (TypeError, ValueError):
                version = 0
            if version >= 1:
                entry["version"] = version
        entries.append(entry)
    return entries


def list_memory_snippets() -> List[str]:
    try:
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        store.load_from_disk()
        return [
            scrub_text(str(entry))[:120]
            for entry in (store.memory_entries + store.user_entries)[-20:]
        ]
    except Exception:
        return []


def _unused_skills_safe() -> List[str]:
    try:
        return ledger.unused_skills()
    except Exception as exc:
        logger.debug("Cannot compute unused skills: %s", exc)
        return []


def _reconcile_pending() -> List[Dict[str, Any]]:
    """Reconcile durable approval states and mirror transitions to the ledger."""
    changed = journal.reconcile()
    for entry in changed:
        try:
            ledger.record_journal_state(entry)
        except Exception as exc:
            logger.warning("Cannot mirror reconciled state in ledger: %s", scrub_text(str(exc)))
    return changed


def auto_cooldown_remaining_minutes() -> float:
    """Minutes left on the automatic-attempt cooldown; ``0.0`` when elapsed.

    Single owner of this arithmetic so the hook gate and the status report can
    never disagree about whether the cooldown has passed.
    """
    last_attempt = journal.last_attempt_ts()
    if last_attempt is None:
        return 0.0
    remaining = config.auto_cooldown_minutes() * 60 - (time.time() - last_attempt)
    return remaining / 60 if remaining > 0 else 0.0


_JOURNAL_DIR_STATE_TEXT = {
    "ok": "usable",
    "missing_creatable": "does not exist yet, will be created on first write",
    "not_a_directory": "path exists but is not a directory",
    "unwritable": "not writable",
    "unknown": "could not be inspected",
}


def _journal_dir_state(directory: Path) -> str:
    """Classify the journal directory without creating or writing anything.

    ``missing_creatable`` walks up to the nearest existing ancestor, because a
    configured path several levels deep is still creatable on first use.
    """
    try:
        if directory.is_dir():
            return "ok" if os.access(str(directory), os.W_OK) else "unwritable"
        if directory.exists():
            return "not_a_directory"
        for ancestor in directory.parents:
            if not ancestor.exists():
                continue
            if not ancestor.is_dir():
                return "not_a_directory"
            return (
                "missing_creatable"
                if os.access(str(ancestor), os.W_OK)
                else "unwritable"
            )
        return "unwritable"
    except Exception:
        return "unknown"


def refine_status() -> Dict[str, Any]:
    """Report why automatic refinement will or will not run.

    Strictly read-only: it creates no directory, writes no journal record,
    consumes no daily budget, and never calls a model. It also does not
    reconcile pending approvals, so an unresolved staged edit still counts
    toward the budget it reports.
    """
    config_readable = config.config_available()
    auto = config.auto_enabled()
    interval = config.auto_turn_interval()
    max_edits = config.max_edits_per_day()
    jdir = config.journal_dir()
    jdir_state = _journal_dir_state(jdir)

    # The effective model belongs in this report. A pinned model that no provider
    # serves turns every pass into an ordinary no_op, and without it here the
    # report would answer "blockers: none" while nothing can possibly succeed.
    try:
        target = config.effective_llm_target()
    except Exception:
        # "unknown", not "host_default": a config key or override file may still
        # pin something, and this report must not claim a resolution it failed to
        # perform.
        target = {
            "provider": "", "model": "", "source": "unknown",
            "issues": ["the effective model could not be resolved"],
        }
    try:
        model_allowed = config.llm_allow_model_override()
        provider_allowed = config.llm_allow_provider_override()
    except Exception:
        model_allowed = provider_allowed = False

    # Read journal-derived numbers only when a journal actually exists, so a
    # mistyped journal_dir is reported rather than silently created.
    journal_present = False
    journal_readable = True
    edits_today = 0
    last_ts: Optional[float] = None
    cooldown_remaining = 0.0
    try:
        journal_path = journal.journal_read_path()
        journal_present = journal_path.is_file()
        if journal_present:
            # An unparseable journal yields no entries rather than an error, so
            # probe readability explicitly: silently reporting "0 edits today"
            # would also report the cooldown as elapsed and let automatic passes
            # run unthrottled against a journal nobody can read.
            journal_path.read_text(encoding="utf-8")
            edits_today = journal.count_today_applied()
            last_ts = journal.last_attempt_ts()
            cooldown_remaining = auto_cooldown_remaining_minutes()
    except Exception as exc:
        journal_readable = False
        logger.warning("Cannot read refine journal for status: %s", scrub_text(str(exc)))

    blockers: List[Dict[str, str]] = []
    if not config_readable:
        blockers.append({
            "code": "config_unreadable",
            "message": (
                "Hermes config could not be read, so automatic refinement stays "
                "off rather than overriding a setting that cannot be confirmed"
            ),
        })
    elif not auto:
        blockers.append({
            "code": "auto_disabled",
            "message": "Automatic refinement is disabled in the config",
        })
    if edits_today >= max_edits:
        blockers.append({
            "code": "budget_exhausted",
            "message": f"Daily edit budget is used up ({edits_today}/{max_edits})",
        })
    cooldown_shown = round(cooldown_remaining, 1)
    if cooldown_remaining > 0:
        blockers.append({
            "code": "cooldown_active",
            # Reuse the rounded value the report prints, so the blocker and the
            # cooldown line can never contradict each other.
            "message": f"Cooldown still active ({cooldown_shown} min left)",
        })
    if jdir_state in ("unwritable", "not_a_directory"):
        blockers.append({
            "code": "journal_dir_unusable",
            "message": (
                "Journal directory is not usable "
                f"({_JOURNAL_DIR_STATE_TEXT.get(jdir_state, jdir_state)})"
            ),
        })
    if not journal_readable:
        blockers.append({
            "code": "journal_unreadable",
            "message": "The journal exists but could not be read",
        })

    warnings: List[Dict[str, str]] = []
    plugin_source_collision = False
    try:
        plugin_source_collision = (jdir / "plugin.yaml").is_file()
    except Exception:
        pass
    if plugin_source_collision:
        warnings.append({
            "code": "journal_dir_is_plugin_source",
            "message": (
                "Journal directory holds the plugin source; "
                "'hermes plugins install --force' would delete runtime data"
            ),
        })
    if not interval:
        warnings.append({
            "code": "turn_trigger_disabled",
            "message": (
                "Turn trigger is off (auto_turn_interval=0); the session-end "
                "fallback still runs"
            ),
        })
    if jdir_state == "unknown":
        warnings.append({
            "code": "journal_dir_unknown",
            "message": (
                "The journal directory could not be inspected, so this report "
                "cannot confirm refinement is able to run"
            ),
        })
    target_issues = [str(item) for item in target.get("issues", []) if item]
    if target_issues:
        # A discarded value must not be visible only in a log line: the file or
        # config key still pins something while this report names another target.
        warnings.append({
            "code": "model_target_issue",
            "message": "; ".join(target_issues),
        })
    if target["source"] == "command":
        warnings.append({
            "code": "model_override_active",
            # Deliberately does not say the override pinned each field: when it
            # sets only one, the other comes from the config and survives
            # '/refine model auto'. Claiming otherwise would describe a state
            # this report did not verify.
            "message": (
                "A '/refine model' override is in force; the effective target is "
                f"{target['model'] or '(host default)'}"
                + (f" on provider {target['provider']}" if target["provider"] else "")
                + ". '/refine model auto' removes the override; any value also set "
                  "in plugins.entries.refine.llm stays in effect after that"
            ),
        })
    # A value the host will refuse is dropped before the call, so it can only be
    # noticed here. Reported per field, because the denied one may be either.
    if target["source"] in ("command", "config"):
        if target["model"] and not model_allowed:
            warnings.append({
                "code": "model_override_trust_denied",
                "message": (
                    f"Model {target['model']} is set but host trust denies model "
                    "overrides, so it is dropped before the call; set "
                    "plugins.entries.refine.llm.allow_model_override to apply it"
                ),
            })
        if target["provider"] and not provider_allowed:
            warnings.append({
                "code": "provider_override_trust_denied",
                "message": (
                    f"Provider {target['provider']} is set but host trust denies "
                    "provider overrides, so it is dropped before the call; set "
                    "plugins.entries.refine.llm.allow_provider_override to apply it"
                ),
            })

    # Session identity — what /refine would analyse if triggered now.
    sid, sid_source = resolve_session_id()
    session_message_count = 0
    if sid:
        try:
            conn = _open_db()
            if conn:
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) n FROM messages WHERE session_id=? AND active=1",
                        (sid,),
                    ).fetchone()
                    session_message_count = row["n"] if row else 0
                finally:
                    conn.close()
        except Exception:
            pass
    if sid_source == "unknown":
        blockers.append({
            "code": "session_unknown",
            "message": (
                "Cannot identify the current session. Neither the host environment "
                "nor a recent hook provided a session id."
            ),
        })

    return {
        "config_readable": config_readable,
        "auto_enabled": auto,
        "auto_turn_interval": interval,
        "turn_trigger_enabled": bool(interval),
        "auto_min_messages": config.auto_min_messages(),
        "auto_cooldown_minutes": config.auto_cooldown_minutes(),
        "last_attempt_ts": last_ts,
        "cooldown_remaining_minutes": cooldown_shown,
        "edits_today": edits_today,
        "max_edits_per_day": max_edits,
        "journal_present": journal_present,
        "journal_readable": journal_readable,
        "journal_dir": str(jdir),
        "journal_dir_state": jdir_state,
        "journal_dir_state_text": _JOURNAL_DIR_STATE_TEXT.get(jdir_state, jdir_state),
        "journal_dir_is_plugin_source": plugin_source_collision,
        "session_id": sid,
        "session_id_source": sid_source,
        "session_message_count": session_message_count,
        "llm_model": target["model"],
        "llm_provider": target["provider"],
        "llm_target_source": target["source"],
        "llm_target_issues": target_issues,
        "llm_model_allowed": model_allowed,
        "llm_provider_allowed": provider_allowed,
        "blockers": blockers,
        "blocker_codes": [b["code"] for b in blockers],
        "warnings": warnings,
        "warning_codes": [w["code"] for w in warnings],
    }


def refine_audit() -> Dict[str, Any]:
    with journal.mutation_lock():
        _reconcile_pending()
    earliest = ledger.earliest_created_ts()
    try:
        current = (
            collect_cross_session_patterns(
                since_ts=earliest,
                max_rows=None,
                max_sessions=None,
            )
            if earliest
            else []
        )
    except Exception:
        current = []
    rows = ledger.audit(current)
    return {"success": True, "rows": rows, "report": ledger.format_audit(rows)}


# ── proposal validation and apply ──────────────────────────────────────────


def _skill_content_error(name: str, content: str) -> Optional[str]:
    if not content.startswith("---"):
        return "Skill content must start with YAML frontmatter"
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.S)
    if not match:
        return "Skill content has incomplete YAML frontmatter"
    frontmatter = match.group(1)
    name_match = re.search(r"(?m)^name\s*:\s*[\"']?([^\n\"']+)", frontmatter)
    if not name_match or name_match.group(1).strip() != name:
        return "Skill frontmatter name must exactly match the target name"
    if not re.search(r"(?m)^description\s*:\s*\S", frontmatter):
        return "Skill frontmatter requires a non-empty description"
    if not content[match.end():].strip():
        return "Skill content requires a Markdown body"
    return None


def _prompt_note_content_error(
    content: str, *, check_rendered_size: bool = True
) -> Optional[str]:
    """Keep globally injected notes narrow, declarative, and renderable as one block."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 2:
        return "Prompt note must contain one or two non-empty policy lines"
    if any(
        line.startswith(("-", "*", "#")) or re.match(r"^\d+[.)]\s", line)
        for line in lines
    ):
        return "Prompt note must be a policy, not a list or procedure"
    first_line = lines[0]
    if not re.match(r"(?i)^when\s+[^,\n]{3,200},\s+\S", first_line):
        return "Prompt note must use 'When <specific condition>, <one action>.'"
    blocked_terms = r"(?i)\b(?:always|never|ignore|system prompt|instruction|any user|all users|every request|first|then|finally)\b"
    if any(re.search(blocked_terms, line) for line in lines):
        return "Prompt note must be a narrow conditional policy, not a global or procedural instruction"
    rendered = "Refine notes:\n- " + content
    if check_rendered_size and len(rendered) > config.prompt_notes_max_chars():
        return (
            f"Prompt note is too large for its rendered context ({len(rendered)} chars; max "
            f"{config.prompt_notes_max_chars()})"
        )
    return None


def _validate_proposal(proposal: Dict[str, Any]) -> Optional[str]:
    action = str(proposal.get("action", "no_op"))
    if action == "no_op":
        return None
    if action not in ("create", "patch"):
        return f"Unsupported action: {action}"
    kind = str(proposal.get("kind", ""))
    if kind not in ("skill", "memory", "prompt"):
        return f"Unsupported kind: {kind}"
    name = str(proposal.get("name", "")).strip()
    content = str(proposal.get("content", ""))
    if not content.strip():
        return f"{action.title()} requires non-empty content"
    if len(content) > _llm.MAX_CONTENT_CHARS:
        return f"Content too large ({len(content)} chars; max {_llm.MAX_CONTENT_CHARS})"
    if kind == "prompt":
        if not config.prompt_notes_enabled():
            return "Prompt notes are disabled"
        if action != "create":
            return "Prompt notes support create only"
        content_error = _prompt_note_content_error(content)
        if content_error:
            return content_error
        scope = proposal.get("scope", "global")
        if scope not in ("global", "session"):
            return "Prompt-note scope must be global or session"
        if scope == "session" and not journal.normalize_prompt_note_session_id(
            proposal.get("session_id", "")
        ):
            return "Session-scoped prompt notes require a verified session ID"
        duplicate = journal.prompt_note_content_exists(content)
        if duplicate is None:
            return "Prompt-note store is unavailable"
        if duplicate:
            return "Identical active prompt note already exists"
    else:
        if not name:
            return "Proposal missing name"
    if kind == "skill":
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            return "Skill name must use lowercase letters, digits, hyphens, or underscores"
        format_error = _skill_content_error(name, content)
        if format_error:
            return format_error
        if name.startswith("hermes-"):
            return f"Skill '{name}' has reserved prefix"
        if action == "create" and name in list_skill_names():
            return f"Skill '{name}' already exists — use patch, not create"
        if action == "patch" and config.only_agent_created():
            try:
                from tools.skill_usage import is_agent_created

                if not is_agent_created(name):
                    return f"Skill '{name}' is bundled/hub-installed (denied by only_agent_created)"
            except ImportError:
                return "Cannot import skill_usage module"
    fingerprint = str(proposal.get("pattern_fingerprint", "") or "")
    if fingerprint and not re.fullmatch(r"[0-9a-f]{12}", fingerprint):
        return "pattern_fingerprint must be the complete 12-character fingerprint"
    if journal.was_applied_recently(proposal, config.dedup_window_days()):
        return f"Identical edit already applied within {config.dedup_window_days()} day(s)"
    return None


def _apply_skill(proposal: Dict[str, Any]) -> Dict[str, Any]:
    from tools.skill_manager_tool import skill_manage

    action = "edit" if proposal["action"] == "patch" else proposal["action"]
    raw = skill_manage(
        action=action,
        name=proposal["name"],
        content=proposal["content"],
        category=proposal.get("category") or None,
    )
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "error": str(raw)}


def _apply_memory(proposal: Dict[str, Any]) -> Dict[str, Any]:
    from tools.memory_tool import MemoryStore

    target = "user" if proposal.get("kind") == "user" else "memory"
    if proposal.get("action") not in ("create", "patch"):
        return {"success": False, "error": f"Unknown memory action: {proposal.get('action')}"}
    store = MemoryStore()
    store.load_from_disk()
    result = store.add(target, proposal["content"])
    if result.get("success") and not result.get("staged"):
        store.save_to_disk(target)
    return result


def _apply_prompt_note(note: Dict[str, str]) -> Dict[str, Any]:
    """Persist a plugin-owned prompt note; no host write or approval is involved."""
    return journal.add_prompt_note(note)


def _skill_baseline_conflict(
    proposal: Dict[str, Any], observed_sha: str = ""
) -> Optional[str]:
    """Return a conflict message when the patch target no longer matches planning.

    Returns None (no conflict) when:
      - baseline is absent or not a dict (legacy proposal without baseline);
      - baseline has invalid structure (logged but not treated as conflict).

    Returns an error string when the current state diverges from the planning
    baseline, meaning the model's replacement was built from stale content.
    """
    import re as _re

    baseline = proposal.get("refine_baseline")
    if not isinstance(baseline, dict):
        return None  # Legacy proposal without baseline — today's behaviour unchanged.
    exists = baseline.get("exists")
    sha = str(baseline.get("sha256", ""))
    if exists is not True or not _re.fullmatch(r"[0-9a-f]{64}", sha):
        logger.warning(
            "Ignoring malformed refine_baseline in proposal for '%s'",
            proposal.get("name", ""),
        )
        return None
    name = str(proposal.get("name", ""))
    if observed_sha:
        # Check B: compare against the sha from prepare_skill_recovery snapshot.
        if observed_sha != sha:
            return (
                f"Skill '{name}': entry changed during refinement planning "
                f"(baseline {sha[:12]}… vs current {observed_sha[:12]}…)"
            )
        return None
    # Check A: read current state from host before backup.
    current = journal.skill_baseline(name)
    if current is None:
        return (
            f"Skill '{name}': entry changed during refinement planning "
            "(cannot confirm target state)"
        )
    if not current.get("exists"):
        return (
            f"Skill '{name}': entry changed during refinement planning "
            "(target was deleted after planning)"
        )
    if current["sha256"] != sha:
        return (
            f"Skill '{name}': entry changed during refinement planning "
            f"(baseline {sha[:12]}… vs current {current['sha256'][:12]}…)"
        )
    return None


def _journal_nonmutation(**kwargs: Any) -> Optional[str]:
    try:
        return journal.log(**kwargs)
    except Exception as exc:
        logger.error("Cannot write refine journal: %s", exc)
        return None


def _reviewer_cooldown_elapsed() -> bool:
    """Keep reviewer calls independently rate-limited across processes."""
    last_review = journal.last_attempt_ts(trigger="reviewer")
    if last_review is None:
        return True
    return time.time() - last_review >= config.reviewer_cooldown_minutes() * 60


def _refine_once(
    llm: PluginLlm,
    *,
    reason: str = "",
    session_id: Optional[str] = None,
    auto: bool = False,
) -> Dict[str, Any]:
    trigger = "auto" if auto else "manual"
    started = time.time()
    safe_reason = scrub_text(reason)

    if journal.daily_limit_reached():
        return {
            "success": False,
            "message": f"Daily edit limit reached ({config.max_edits_per_day()}). "
            f"Applied/pending/prepared today: {journal.count_today_applied()}.",
        }

    evidence_limit = 60
    if config.min_signal_required() and config.reviewer_fallback_enabled():
        evidence_limit = max(evidence_limit, config.reviewer_min_messages())
    evidence = collect_evidence(session_id=session_id, limit=evidence_limit)
    session = evidence.get("session_id", "")
    session_source = evidence.get("session_id_source", "unknown")
    if not session and session_source == "unknown":
        return {
            "success": False,
            "outcome": "session_unknown",
            "message": "Cannot identify the current session; refine did not run.",
            "evidence": evidence,
            "reversible": False,
        }
    if len(evidence.get("messages", [])) < 3:
        return {
            "success": True,
            "message": "Not enough messages in this session to analyze.",
            "evidence": evidence,
        }

    error_patterns = patterns.merge_patterns(
        evidence.get("error_patterns", []), collect_cross_session_patterns()
    )
    evidence["error_patterns"] = error_patterns
    corrections = evidence.get("user_corrections", [])
    lines: List[str] = []
    for message in evidence.get("messages", []):
        tag = f"[{message['role']}]"
        if message.get("tool_name"):
            tag += f"({message['tool_name']})"
        lines.append(f"{tag} {message['content'][:400]}")
    evidence_text = "\n".join(lines)
    proposal_context = safe_reason
    if config.min_signal_required() and not patterns.has_signal(
        error_patterns, corrections, min_count=config.min_pattern_count()
    ):
        should_review = (
            config.reviewer_fallback_enabled()
            and len(evidence.get("messages", [])) >= config.reviewer_min_messages()
            and _reviewer_cooldown_elapsed()
        )
        if should_review:
            reviewer = _llm.review_fallback(llm, evidence_text)
            rationale = scrub_text(str(reviewer.get("rationale", "")))
            decision = "approved" if reviewer.get("should_refine") else "declined"
            reviewer_reason = f"Reviewer {decision}: {rationale}"
            reviewer_entry_id = _journal_nonmutation(
                trigger="reviewer",
                reason=reviewer_reason,
                session_id=session,
                proposal={
                    "action": "no_op",
                    "reason": reviewer_reason,
                    "expected_outcome": "",
                },
                outcome="no_op",
            )
            if not reviewer_entry_id:
                return {
                    "success": False,
                    "message": "Reviewer decision could not be journaled.",
                    "llm_called": True,
                    "reviewer": decision,
                    "evidence": evidence,
                    "reversible": False,
                }
            if not reviewer.get("should_refine"):
                proposal = {
                    "action": "no_op",
                    "reason": reviewer_reason,
                    "expected_outcome": "",
                }
                return {
                    "success": True,
                    "message": f"No actionable improvement found. {reviewer_reason}",
                    "journal_id": reviewer_entry_id,
                    "proposal": proposal,
                    "llm_called": True,
                    "reviewer": "declined",
                    "evidence": evidence,
                    "reversible": False,
                }
            reviewer_instructions = scrub_text(str(reviewer.get("instructions", "")))
            proposal_context = "\n".join(
                part for part in (safe_reason, f"Reviewer-approved instructions: {reviewer_instructions}") if part
            )
        else:
            proposal = {
                "action": "no_op",
                "reason": f"No repeated failure (min {config.min_pattern_count()}x) and no explicit correction.",
                "expected_outcome": "",
            }
            entry_id = _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason or proposal["reason"],
                session_id=session,
                proposal=proposal,
                outcome="no_op",
            )
            if not entry_id:
                return {
                    "success": False,
                    "message": "No edit was needed, but the journal write failed.",
                    "evidence": evidence,
                }
            return {
                "success": True,
                "message": f"No actionable improvement found. {proposal['reason']}",
                "journal_id": entry_id,
                "llm_called": False,
                "evidence": evidence,
                "reversible": False,
            }

    proposal = _llm.propose(
        llm=llm,
        evidence_text=evidence_text,
        existing_skills=list_skill_entries(),
        existing_memories=list_memory_snippets(),
        error_patterns=error_patterns,
        user_corrections=[item.get("snippet", "") for item in corrections],
        unused_skills=_unused_skills_safe(),
        refinement_history=journal.recent_refinements(config.history_max_entries()),
        purpose="refine",
        run_context=proposal_context,
        skill_content_loader=journal.read_skill_content,
    )
    proposal = sanitize(proposal)
    proposal = dict(
        proposal,
        expected_outcome=_llm.normalize_expected_outcome(
            proposal.get("expected_outcome")
        ),
    )
    failure = scrub_text(str(proposal.get("failure", "")).strip())
    if failure:
        failure_messages = {
            "truncated": "The refine proposal was cut off before it completed.",
            "malformed": "The refine proposal was malformed and could not be read.",
            "no_final_text": (
                "The model returned only reasoning and no final refine proposal."
            ),
        }
        failure_message = failure_messages.get(
            failure, "The refine proposal could not be completed."
        )
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or failure_message,
            session_id=session,
            proposal=proposal,
            outcome="llm_incomplete",
            error=failure_message,
        )
        response = {
            "success": False,
            "message": failure_message,
            "llm_called": True,
            "failure": failure,
            "proposal": proposal,
            "evidence": evidence,
            "reversible": False,
        }
        if entry_id:
            response["journal_id"] = entry_id
        return response
    evidence_summary = {
        "session_id": session,
        "messages": len(evidence.get("messages", [])),
        "errors": evidence.get("error_count", 0),
    }

    if proposal.get("action") == "multi":
        transaction = _apply_transaction(
            proposal,
            trigger=trigger,
            safe_reason=safe_reason,
            session=session,
            started=started,
        )
        transaction["evidence"] = evidence_summary
        return transaction

    proposal = _normalize_edit(proposal, session)

    if proposal.get("action") == "no_op":
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or proposal.get("reason", ""),
            session_id=session,
            proposal=proposal,
            outcome="no_op",
        )
        if not entry_id:
            return {
                "success": False,
                "message": "Proposal was no_op, but the journal write failed.",
                "proposal": proposal,
            }
        return {
            "success": True,
            "message": f"No actionable improvement found. {proposal.get('reason', '')}",
            "journal_id": entry_id,
            "proposal": proposal,
            "evidence": evidence,
            "reversible": False,
        }

    response = _apply_edit(
        proposal,
        trigger=trigger,
        safe_reason=safe_reason,
        session=session,
        started=started,
    )
    response["evidence"] = evidence_summary
    return response


def _apply_edit(
    proposal: Dict[str, Any],
    *,
    trigger: str,
    safe_reason: str,
    session: str,
    started: float,
    group: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate, back up, apply, and finalize exactly one edit.

    Guardrails read live host and journal state, so an edit inside a transaction
    is checked against the edits that were already applied before it.
    """
    guardrail_error = _validate_proposal(proposal)
    if guardrail_error:
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason,
            session_id=session,
            proposal=proposal,
            outcome="rejected",
            error=guardrail_error,
            group=group,
        )
        result = {
            "success": False,
            "message": f"Proposal rejected by guardrails: {guardrail_error}",
            "proposal": proposal,
            "reversible": False,
            "edits_applied": 0,
        }
        if entry_id:
            result["record_id"] = entry_id
        return result

    kind = proposal["kind"]
    action = proposal["action"]
    name = proposal.get("name", "")
    backup_path = ""
    snapshot: Optional[Dict[str, Any]] = None
    recovery: Dict[str, Any] = {}
    prompt_note: Optional[Dict[str, str]] = None
    if kind == "skill" and action == "patch":
        # Check A: refuse before backup if planning baseline is stale.
        conflict_a = _skill_baseline_conflict(proposal)
        if conflict_a:
            entry_id = _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason,
                session_id=session,
                proposal=proposal,
                outcome="conflict",
                error=conflict_a,
                group=group,
            )
            result = {
                "success": False,
                "message": conflict_a,
                "proposal": proposal,
                "reversible": False,
                "edits_applied": 0,
            }
            if entry_id:
                result["record_id"] = entry_id
            return result
        captured = journal.prepare_skill_recovery(name)
        if captured is None:
            error = f"Cannot create durable backup for skill '{name}'; mutation aborted"
            _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason,
                session_id=session,
                proposal=proposal,
                outcome="error",
                error=error,
                group=group,
            )
            return {
                "success": False,
                "message": error,
                "proposal": proposal,
                "reversible": False,
                "edits_applied": 0,
            }
        # Check B: verify the backup snapshot came from the planning baseline.
        conflict_b = _skill_baseline_conflict(
            proposal, observed_sha=captured["snapshot"]["before_sha256"]
        )
        if conflict_b:
            # The recovery capture wrote a raw backup before discovering the
            # conflict. A conflict is never reversible, so remove that copy;
            # if cleanup fails, retain its path in the journal for auditability.
            conflict_backup = Path(str(captured["backup_path"]))
            retained_backup_path = ""
            try:
                conflict_backup.unlink(missing_ok=True)
            except OSError as exc:
                retained_backup_path = str(conflict_backup)
                logger.warning(
                    "Cannot remove unused conflict backup for skill '%s': %s",
                    name,
                    scrub_text(str(exc)),
                )
            entry_id = _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason,
                session_id=session,
                proposal=proposal,
                outcome="conflict",
                backup_path=retained_backup_path,
                error=conflict_b,
                group=group,
            )
            result = {
                "success": False,
                "message": conflict_b,
                "proposal": proposal,
                "reversible": False,
                "edits_applied": 0,
            }
            if entry_id:
                result["record_id"] = entry_id
            return result
        backup_path = str(captured["backup_path"])
        snapshot = captured["snapshot"]
        recovery = {"type": "skill_patch", "name": name}
    elif kind == "skill":
        recovery = {"type": "skill_create", "name": name}
    elif kind == "prompt":
        prompt_note = journal.new_prompt_note(
            proposal["content"],
            scope=str(proposal.get("scope", "global")),
            session_id=str(proposal.get("session_id", "")),
        )
        if prompt_note is None:
            error = "Cannot access plugin-owned prompt-note storage; mutation aborted"
            _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason,
                session_id=session,
                proposal=proposal,
                outcome="error",
                error=error,
                group=group,
            )
            return {
                "success": False,
                "message": error,
                "proposal": proposal,
                "reversible": False,
                "edits_applied": 0,
            }
        proposal = dict(proposal, name=prompt_note["id"], note_id=prompt_note["id"])
        name = prompt_note["id"]
        recovery = {"type": "prompt_note", "note_id": prompt_note["id"]}
    else:
        target = "user" if kind == "user" else "memory"
        memory_recovery = journal.memory_recovery(target, proposal["content"])
        if memory_recovery is None:
            error = f"Cannot capture {target} memory recovery state; mutation aborted"
            _journal_nonmutation(
                trigger=trigger,
                reason=safe_reason,
                session_id=session,
                proposal=proposal,
                outcome="error",
                error=error,
                group=group,
            )
            return {
                "success": False,
                "message": error,
                "proposal": proposal,
                "reversible": False,
                "edits_applied": 0,
            }
        recovery = memory_recovery

    try:
        entry_id = journal.prepare(
            trigger=trigger,
            reason=safe_reason,
            session_id=session,
            proposal=proposal,
            backup_path=backup_path,
            recovery=recovery,
            group=group,
            snapshot=snapshot,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"Journal preparation failed; mutation aborted: {scrub_text(str(exc))}",
            "proposal": proposal,
            "reversible": False,
            "edits_applied": 0,
        }

    try:
        if kind == "skill":
            apply_result = _apply_skill(proposal)
        elif kind == "prompt":
            apply_result = _apply_prompt_note(prompt_note or {})
        else:
            apply_result = _apply_memory(proposal)
    except Exception as exc:
        apply_result = {"success": False, "error": scrub_text(str(exc))}
    apply_result = sanitize(apply_result)

    staged = bool(apply_result.get("success") and apply_result.get("staged"))
    pending_id = scrub_text(str(apply_result.get("pending_id", ""))) if staged else ""
    if staged and not pending_id:
        apply_result = {
            "success": False,
            "error": "Host staged the mutation without a pending_id",
        }
        staged = False
    if apply_result.get("success") and not staged:
        prepared_entry = journal.get_entry(entry_id) or {
            "proposal": proposal,
            "recovery": recovery,
            "backup_path": backup_path,
            "snapshot": snapshot or {},
        }
        if not journal.target_matches_applied(prepared_entry):
            apply_result = {
                "success": False,
                "error": "Host reported success but the target state does not match the proposal",
            }
    outcome = (
        "pending_approval"
        if staged
        else ("applied" if apply_result.get("success") else "error")
    )
    try:
        finalized = journal.finalize(
            entry_id,
            outcome,
            error=scrub_text(str(apply_result.get("error", ""))),
            pending_id=pending_id if staged else None,
        )
    except Exception as exc:
        if apply_result.get("success"):
            return {
                "success": False,
                "message": f"Mutation completed but journal finalization failed; recovery id: {entry_id}. Error: {scrub_text(str(exc))}",
                "journal_id": entry_id,
                "proposal": proposal,
                "result": sanitize(apply_result),
                "backup_path": backup_path,
                "reversible": not staged,
                # The mutation landed and its prepared record already consumed
                # budget, so this edit still owns a recovery id even though the
                # run must stop.
                "edits_applied": 1,
            }
        return {
            "success": False,
            "message": f"Apply failed and journal finalization also failed: {scrub_text(str(exc))}",
            "proposal": proposal,
            "result": sanitize(apply_result),
            "reversible": False,
            "edits_applied": 0,
        }

    if outcome in ("applied", "pending_approval"):
        try:
            ledger.record_edit(
                proposal,
                entry_id,
                outcome=outcome,
                pending_id=pending_id,
            )
        except Exception as exc:
            logger.warning("Cannot record edit in ledger: %s", exc)

    message = (
        f"done ({time.time() - started:.1f}s) | action={action} kind={kind} "
        f"name={name} | outcome={outcome}"
    )
    if staged and pending_id:
        message += f" | pending_id={pending_id}"
    if apply_result.get("error"):
        message += f" | error={scrub_text(str(apply_result['error']))[:100]}"

    success = bool(apply_result.get("success"))
    response: Dict[str, Any] = {
        "success": success,
        "message": message,
        "proposal": proposal,
        "result": sanitize(apply_result),
        "backup_path": backup_path,
        "reversible": bool(
            success and outcome == "applied" and journal.is_reversible(finalized)
        ),
        "outcome": outcome,
        # The daily budget counts edits, so a transaction reports each applied or
        # reserved edit rather than one proposal.
        "edits_applied": 1 if success else 0,
    }
    if success:
        response["journal_id"] = entry_id
    else:
        response["record_id"] = entry_id
    return response


def _normalize_edit(proposal: Dict[str, Any], session: str) -> Dict[str, Any]:
    """Apply the boundary normalization every edit needs before guardrails run."""
    normalized = dict(
        proposal,
        expected_outcome=_llm.normalize_expected_outcome(
            proposal.get("expected_outcome")
        ),
    )
    if normalized.get("kind") == "prompt":
        scope = config.prompt_notes_default_scope()
        normalized = dict(
            normalized,
            content=journal.normalize_prompt_note_content(normalized.get("content", "")),
            scope=scope,
            session_id=(
                journal.normalize_prompt_note_session_id(session)
                if scope == "session"
                else ""
            ),
        )
    return normalized


def _apply_transaction(
    proposal: Dict[str, Any],
    *,
    trigger: str,
    safe_reason: str,
    session: str,
    started: float,
) -> Dict[str, Any]:
    """Apply one multi-edit proposal as a sequence of independent durable edits.

    Each edit keeps its own journal record, recovery metadata, and rollback id, so
    the existing single-edit rollback and approval machinery is reused unchanged.
    Edits are applied in order and the run stops at the first failure, leaving a
    journal that states exactly which edits applied and which did not.
    """
    edits = [edit for edit in proposal.get("edits", []) if isinstance(edit, dict)]
    if not edits:
        error = "Transaction contained no usable edit"
        _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or error,
            session_id=session,
            proposal=proposal,
            outcome="rejected",
            error=error,
        )
        return {
            "success": False,
            "outcome": "failed",
            "message": error,
            "proposal": proposal,
            "results": [],
            "recoveries": [],
            "journal_ids": [],
            "reversible": False,
            "edits_applied": 0,
        }
    group_id = uuid.uuid4().hex[:12]
    summary = scrub_text(str(proposal.get("summary", ""))).strip()[
        : _llm.MAX_SUMMARY_CHARS
    ]
    shared_reason = scrub_text(str(proposal.get("reason", "")))
    shared_expected = _llm.normalize_expected_outcome(proposal.get("expected_outcome"))
    shared_fingerprint = str(proposal.get("pattern_fingerprint", "") or "")
    dropped = int(proposal.get("dropped_edits", 0) or 0)

    def edit_proposal(edit: Dict[str, Any]) -> Dict[str, Any]:
        """Give one edit the transaction's shared justification, then normalize it."""
        merged = dict(edit)
        if not str(merged.get("reason", "")).strip():
            merged["reason"] = shared_reason
        if not str(merged.get("expected_outcome", "") or "").strip():
            merged["expected_outcome"] = shared_expected
        if not str(merged.get("pattern_fingerprint", "") or ""):
            merged["pattern_fingerprint"] = shared_fingerprint
        return _normalize_edit(sanitize(merged), session)

    def edit_group(index: int) -> Dict[str, Any]:
        group = {
            "id": group_id,
            "index": index,
            "size": len(edits),
            "summary": summary,
        }
        if dropped:
            group["dropped"] = dropped
        return group

    results: List[Dict[str, Any]] = []
    stop_reason = ""

    # ── Stale-plan preflight: reject the entire transaction if any skill patch
    # was built from content that no longer matches the live host state. This
    # prevents a partial apply where edit #1 succeeds but edit #2 would conflict.
    stale_edits: List[int] = []
    for index, edit in enumerate(edits):
        normalized = edit_proposal(edit)
        if (
            normalized.get("kind") == "skill"
            and normalized.get("action") == "patch"
            and isinstance(normalized.get("refine_baseline"), dict)
        ):
            conflict = _skill_baseline_conflict(normalized)
            if conflict:
                stale_edits.append(index)
    if stale_edits:
        conflict_msg = (
            f"Transaction rejected: entry changed during refinement planning "
            f"(stale edit(s) at index {stale_edits})"
        )
        for index, edit in enumerate(edits):
            normalized = edit_proposal(edit)
            if index in stale_edits:
                _journal_nonmutation(
                    trigger=trigger,
                    reason=safe_reason,
                    session_id=session,
                    proposal=normalized,
                    outcome="conflict",
                    error=conflict_msg,
                    group=edit_group(index),
                )
            else:
                _journal_nonmutation(
                    trigger=trigger,
                    reason=safe_reason,
                    session_id=session,
                    proposal=normalized,
                    outcome="rejected",
                    error=conflict_msg,
                    group=edit_group(index),
                )
        return {
            "success": False,
            "outcome": "failed",
            "message": conflict_msg,
            "proposal": proposal,
            "results": [],
            "recoveries": [],
            "journal_ids": [],
            "reversible": False,
            "edits_applied": 0,
        }

    for index, edit in enumerate(edits):
        # Re-read the durable budget between edits: it counts edits, so a long
        # transaction can legitimately exhaust it part way through.
        if journal.daily_limit_reached():
            stop_reason = (
                f"Daily edit limit reached ({config.max_edits_per_day()}) "
                "before this edit was attempted"
            )
            break
        item = _apply_edit(
            edit_proposal(edit),
            trigger=trigger,
            safe_reason=safe_reason,
            session=session,
            started=started,
            group=edit_group(index),
        )
        results.append(item)
        if not item.get("success"):
            stop_reason = (
                f"An earlier edit of transaction {group_id} did not complete"
            )
            break

    # Every edit of a transaction leaves a durable trace, so a partial
    # application is readable from the journal alone rather than only from a
    # message that automatic runs discard. ``rejected`` consumes no daily budget.
    for index in range(len(results), len(edits)):
        _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason,
            session_id=session,
            proposal=edit_proposal(edits[index]),
            outcome="rejected",
            error=stop_reason or "Edit was not attempted",
            group=edit_group(index),
        )

    # "Recoverable" is deliberately wider than "successful": an edit whose host
    # mutation landed but whose journal finalization then failed still owns a
    # recovery id and must appear in the list the message points the user at.
    recoverable = [item for item in results if int(item.get("edits_applied", 0) or 0)]
    succeeded = [item for item in results if item.get("success")]
    recoveries = _recoveries_for(recoverable)
    skipped = len(edits) - len(results)
    elapsed = time.time() - started

    if len(succeeded) == len(edits) and not dropped:
        success, outcome = True, "completed"
        message = (
            f"transaction {group_id}: {len(succeeded)} edit(s) applied or reserved "
            f"({elapsed:.1f}s)"
        )
    elif recoverable:
        success, outcome = False, "partial_success"
        message = (
            f"PARTIAL SUCCESS: transaction {group_id} applied or reserved "
            f"{len(recoverable)} of {len(edits)} edit(s) and then stopped. "
            "Use the recovery IDs listed below, newest first."
        )
    else:
        success, outcome = False, "failed"
        message = f"transaction {group_id}: no edit was applied"
    if results and not results[-1].get("success"):
        message += f" | stopped: {scrub_text(str(results[-1].get('message', '')))[:160]}"
    elif skipped:
        message += (
            f" | stopped: daily edit limit reached ({config.max_edits_per_day()}); "
            f"{skipped} edit(s) not attempted"
        )
    if dropped:
        message += f" | {dropped} proposed edit(s) discarded before apply"
    if summary:
        message += f" | {summary}"

    return {
        "success": success,
        "outcome": outcome,
        "message": message,
        "proposal": proposal,
        "results": results,
        "recoveries": recoveries,
        "journal_ids": [item["journal_id"] for item in recoveries],
        "reversible": any(item.get("reversible") for item in recoveries),
        "edits_applied": len(recoverable),
    }


def _completed_targets(result: Dict[str, Any]) -> List[str]:
    """Name what a pass already reserved, so the next pass cannot repeat it."""
    items = result.get("results")
    proposals = (
        [
            item.get("proposal", {})
            for item in items
            if isinstance(item, dict) and item.get("success")
        ]
        if isinstance(items, list)
        else [result.get("proposal", {})]
    )
    targets: List[str] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        action = scrub_text(str(proposal.get("action", "")))
        if action in ("", "no_op", "multi"):
            continue
        kind = scrub_text(str(proposal.get("kind", "")))
        name = scrub_text(str(proposal.get("name", "")))
        targets.append(f"{action} {kind} '{name}'")
    return targets


def _recoveries_for(applied: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Describe every durable recovery id an applied or reserved edit left behind.

    Newest first, because that is the only safe rollback order: memory recovery
    is positional, so undoing an earlier append before a later one shifts the
    later entry and its rollback fails closed as a conflict.
    """
    recoveries: List[Dict[str, Any]] = []
    for item in reversed(applied):
        journal_id = item.get("journal_id")
        if not journal_id:
            continue
        durable = journal.get_entry(str(journal_id)) or {}
        recovery: Dict[str, Any] = {
            "journal_id": str(journal_id),
            "outcome": durable.get("outcome", item.get("outcome", "unknown")),
            "reversible": bool(item.get("reversible")),
        }
        if item.get("reversible"):
            recovery["rollback_command"] = f"/refine rollback {journal_id}"
        recoveries.append(recovery)
    return recoveries


def refine_run(
    llm: PluginLlm,
    *,
    reason: str = "",
    session_id: Optional[str] = None,
    auto: bool = False,
) -> Dict[str, Any]:
    """Serialize a run, reconcile approvals, and preserve every recovery id."""
    started = time.time()
    with journal.mutation_lock():
        _reconcile_pending()
        runs: List[Dict[str, Any]] = []
        # ``max_edits_per_run`` bounds proposal passes; ``max_edits_per_proposal``
        # bounds edits inside one transaction; the daily edit budget bounds edits
        # overall and is re-checked before every single edit.
        max_runs = max(1, config.max_edits_per_run())
        run_reason = scrub_text(reason)
        for _ in range(max_runs):
            if journal.daily_limit_reached():
                break
            result = _refine_once(
                llm, reason=run_reason, session_id=session_id, auto=auto
            )
            runs.append(result)
            if not result.get("success") or not int(result.get("edits_applied", 0) or 0):
                break
            targets = _completed_targets(result)
            if not targets:
                break
            note = (
                f"Already completed or reserved {'; '.join(targets)} in this run; "
                "propose a different edit or no_op."
            )
            run_reason = f"{reason}\n{note}".strip() if reason else note
            run_reason = scrub_text(run_reason)

        if not runs:
            return {
                "success": False,
                "message": f"Daily edit limit reached ({config.max_edits_per_day()}).",
                "reversible": False,
            }
        if len(runs) == 1:
            return runs[0]

        recoveries: List[Dict[str, Any]] = []
        for item in runs:
            inner = item.get("recoveries")
            if isinstance(inner, list) and inner:
                recoveries.extend(inner)
                continue
            if item.get("journal_id") and int(item.get("edits_applied", 0) or 0):
                recoveries.extend(_recoveries_for([item]))

        failed_after_success = bool(
            recoveries and any(not item.get("success") for item in runs)
        )
        last = runs[-1]
        if failed_after_success:
            message = (
                f"PARTIAL SUCCESS: {len(recoveries)} earlier edit(s) were applied or reserved, "
                "but a later pass failed. Use the recovery IDs listed below."
            )
            outcome = "partial_success"
            success = False
        else:
            message = (
                f"{len(runs)} pass(es), {len(recoveries)} edit(s) applied or reserved "
                f"({time.time() - started:.1f}s)"
            )
            outcome = "completed"
            success = all(item.get("success") for item in runs)
        response: Dict[str, Any] = {
            "success": success,
            "outcome": outcome,
            "message": message,
            "proposal": last.get("proposal", runs[0].get("proposal", {})),
            "results": runs,
            "recoveries": recoveries,
            "journal_ids": [item["journal_id"] for item in recoveries],
            "evidence": runs[0].get("evidence", {}),
            "reversible": any(item.get("reversible") for item in recoveries),
            "edits_applied": len(recoveries),
        }
        return response


def refine_rollback(entry_id: str) -> Dict[str, Any]:
    with journal.mutation_lock():
        _reconcile_pending()
        entry = journal.get_entry(entry_id)
        if not entry:
            return {"success": False, "error": f"Entry {entry_id} not found"}
        if entry.get("outcome") == "rolled_back":
            return {"success": True, "message": f"Entry {entry_id} is already rolled back"}
        if entry.get("outcome") == "pending_rollback":
            return {
                "success": True,
                "staged": True,
                "pending_id": entry.get("pending_id", ""),
                "message": "Rollback is still pending approval; target is unchanged",
            }
        if not journal.is_reversible(entry):
            return {"success": False, "error": f"Entry {entry_id} is not reversible"}
        kind = entry.get("proposal", {}).get("kind", "skill")
        if kind == "skill":
            result = journal.rollback_skill(entry_id)
        elif kind in ("memory", "user"):
            result = journal.rollback_memory(entry_id)
        elif kind == "prompt":
            result = journal.rollback_prompt_note(entry_id)
        else:
            return {"success": False, "error": f"Unknown kind for rollback: {kind}"}
        latest = journal.get_entry(entry_id)
        if latest:
            try:
                ledger.record_journal_state(latest)
            except Exception as exc:
                logger.warning("Cannot mirror rollback state in ledger: %s", scrub_text(str(exc)))
        return sanitize(result)
