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
_UNTRUSTED_TOOL_TAG = re.compile(
    r"<\s*/?\s*untrusted_tool_result\s*>", re.IGNORECASE
)
_RECORD_SEPARATOR = re.compile(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]+")
_RESOURCE_REFERENCE = re.compile(
    r"""(?ix)
    (?: [a-z]+:// )                         # any URL scheme
    | (?: ~[/\\] | (?<![\w.])[/\\][\w.] )   # absolute or home-relative path
    | (?: [A-Za-z]: (?=\S) )                    # absolute or drive-relative Windows path
    | (?: \$\{?\w+ | %\w+% )                # environment expansion
    | [`|;&><$]                               # shell metacharacters
    """
)
_RESOURCE_NETWORK_OR_SHELL = re.compile(r"(?ix)(?:[a-z]+://|[`|;&><$])")
_HOST_REFERENCE = re.compile(
    r"""(?ix)
    (?:
        (?<![\w.-])(?:[a-z0-9-]+\.)+[a-z]{2,63}\.?(?![\w.-]) # dotted hostname
        | \b(?:localhost|intranet)\b                            # common bare hostnames
        | \b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b # IPv4
        | \[(?:[0-9a-f]{0,4}:){2,7}[0-9a-f:]*\]              # bracketed IPv6
        | (?<![0-9a-f])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f:]*(?![0-9a-f:]) # bare IPv6
    )
    """
)
_LEGACY_IPV4_COMPONENT = r"(?:0x[0-9a-f]{1,8}|0[0-7]{0,11}|[1-9]\d{0,9})"
_LEGACY_IPV4_LITERAL = re.compile(
    rf"""(?ix)
    (?<![\w.])
    (?:
        (?:{_LEGACY_IPV4_COMPONENT})(?:\.(?:{_LEGACY_IPV4_COMPONENT})){{1,3}}
        | 0x[0-9a-f]{{1,8}}
        | 0[0-7]{{1,11}}
        | [1-9]\d{{6,9}}
    )
    \.?(?![\w.])
    """
)
_LEGACY_IPV4_OVERFLOW = re.compile(
    r"""(?ix)
    (?<![\w.])
    (?:
        0x[0-9a-f]{9,}(?![0-9a-f])
        | 0[0-7]{12,}(?![0-7])
        | [1-9]\d{10,}(?!\d)
    )
    (?=\.|[^\w.]|$)
    """
)
_SHORT_DECIMAL_IPV4_LITERAL = re.compile(r"(?<![\w.])(?:0|[1-9]\d{0,5})(?![\w.])")
_HTTP_STATUS_REFERENCE = re.compile(
    r"(?i)\b(?:the\s+)?(?:request|response)\s+returns?\s+([1-5]\d{2})\b"
)
_OVERRIDE_INTENT = re.compile(
    r"(?i)\b(?:ignore|disregard|override|bypass|skip|forget|regardless of|instead of)\b"
)
_HIGHER_PRIORITY_GUIDANCE = re.compile(
    r"(?i)\b(?:developer|system|prompt|instruction|guidance|constraint|policy|rule|guardrail)\b"
)
_PROMPT_NOTE_FORMAT = re.compile(r"(?i)^when\s+[^,\n]{3,200},\s+\S")
_PROMPT_NOTE_CONDITION = re.compile(r"(?i)^when\s+([^,\n]{3,200}),\s+\S")
_PROMPT_NOTE_ACTION = re.compile(r"(?i)^when\s+[^,\n]{3,200},\s+(.+?)\s*$")
_PROMPT_NOTE_SAFE_TARGET = r"""
(?:(?:the|this|its|an?|expected|relevant|exact)\s+)+
(?:endpoint|parameters?|target|response|result|output|value|shape|error|failure|tests?|request|details?)
(?:\s+(?:and|or|the|this|its|expected|relevant|exact|endpoint|parameters?|target|response|result|output|value|shape|error|failure|tests?|request|details?))*
"""
_PROMPT_NOTE_SAFE_ACTION = re.compile(
    rf"""(?ix)
    (?:
        (?:check|confirm|inspect|verify)\s+(?:{_PROMPT_NOTE_SAFE_TARGET})(?:\s+before\s+(?:acting|continuing))?
        | confirm\s+it(?:\s+before\s+acting)?
        | confirm\s+it['’]s\s+clear,\s+concise,\s+and\s+accurate
        | avoid\s+(?:unsupported\s+claims|speculation|unnecessary\s+changes)
        | ask\s+(?:for\s+clarification|(?:a|one)\s+focused\s+question)
        | follow\s+(?:the\s+)?(?:old|current|existing|established)\s+(?:policy|guidance)
        | keep\s+(?:the\s+)?(?:response|result|scope|change|policy)\s+(?:narrow|concise|minimal|focused)
        | log\s+(?:the\s+)?(?:error|failure|outcome)
        | mention\s+(?:the\s+)?(?:limitation|uncertainty|assumption)(?:\s+plainly)?
        | prefer\s+(?:unified|concise|clear|minimal)\s+(?:format|response|summary)
        | redact\s+(?:credentials?|secrets?|sensitive\s+(?:data|values?)|api[_-]?key(?:\s*=\s*["']?\[REDACTED\]["']?)?)
        | reject\s+(?:it|the\s+(?:invalid\s+)?(?:target|request|response|result))
        | retry\s+(?:the\s+|this\s+)?request
        | summarize\s+(?:the\s+)?(?:common\s+cause|error|failure|result|outcome)
        | use\s+the\s+(?:supplied|provided|exact)\s+(?:spelling|name|format)
        | wait\s+for\s+(?:clarification|confirmation|approval|input)
        | (?:always\s+)?(?:include|provide|supply|set|pass)\s+(?:both\s+|all\s+)?(?:the\s+)?(?:required\s+)?[\u2018\u2019](?:path|content)[\u2018\u2019](?:\s*(?:,|and|or)\s*[\u2018\u2019](?:path|content)[\u2018\u2019])*\s+(?:fields?|arguments?|parameters?|values?|keys?)
        | always\s+include\s+both\s+path\s+and\s+content\s+fields?
        | ask\s+before\s+retrying\s+(?:a|the)\s+third\s+time
        | check\s+timing\s+assumptions\s+before\s+rerunning
        | mention\s+which\s+sections\s+were\s+skipped
    )\.?
    """
)

# One canonical action list drives both model guidance and validator anti-drift tests.
PROMPT_NOTE_ACTION_EXAMPLES = _llm.PROMPT_NOTE_ACTION_EXAMPLES



def _one_line(value: Any) -> str:
    """Normalize every Unicode line boundary before rendering one record."""
    return _RECORD_SEPARATOR.sub(" ", str(value)).strip()


def _has_host_reference(text: str) -> bool:
    """Reject names plus conventional and legacy numeric IP address literals."""
    if (
        _HOST_REFERENCE.search(text)
        or _LEGACY_IPV4_LITERAL.search(text)
        or _LEGACY_IPV4_OVERFLOW.search(text)
    ):
        return True
    status_spans = [match.span(1) for match in _HTTP_STATUS_REFERENCE.finditer(text)]
    return any(
        match.span() not in status_spans
        for match in _SHORT_DECIMAL_IPV4_LITERAL.finditer(text)
    )


# ── session identity ───────────────────────────────────────────────────────
# The host does not pass session_id to slash-command handlers (contract is
# fn(raw_args) -> str|None). But pre_llm_call and post_llm_call hooks do
# receive it every turn. This module remembers the last value seen, so that a
# manual /refine command running in the same process can resolve it.

_LAST_SESSION_ID = ""
_LAST_SESSION_LOCK = threading.Lock()
_AUTO_EVENTS: List[Dict[str, Any]] = []
_LAST_AUTO_EVENT_LOCK = threading.Lock()
_AUTO_EVENTS_MAX = 10
_PERSISTENCE_WARNING_BYTES = 100 * 1024 * 1024


def note_auto_event(code: str, message: str) -> None:
    """Remember bounded, scrubbed background events for /refine status."""
    event = {
        "code": _one_line(scrub_text(code))[:64],
        "message": _one_line(scrub_text(message))[:300],
        "ts": time.time(),
    }
    with _LAST_AUTO_EVENT_LOCK:
        _AUTO_EVENTS.append(event)
        del _AUTO_EVENTS[:-_AUTO_EVENTS_MAX]


def recent_auto_events() -> List[Dict[str, Any]]:
    with _LAST_AUTO_EVENT_LOCK:
        return [dict(event) for event in _AUTO_EVENTS]


def last_auto_event() -> Dict[str, Any]:
    events = recent_auto_events()
    return events[-1] if events else {}


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
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    except Exception as exc:
        logger.warning("Cannot open state.db: %s", scrub_text(str(exc)))
        return None


def _get_session_source_status(session_id: str) -> Tuple[str, str]:
    """Return a scrubbed source plus ``ok``, ``missing``, or ``error``."""
    if not session_id:
        return "", "missing"
    connection = _open_db()
    if not connection:
        return "", "error"
    try:
        row = connection.execute(
            "SELECT source FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return "", "missing"
        return scrub_text(str(row["source"] or "")), "ok"
    except Exception as exc:
        logger.warning("Cannot read session source: %s", scrub_text(str(exc)))
        return "", "error"
    finally:
        connection.close()


def _get_session_source(session_id: str) -> str:
    """Compatibility wrapper for status and callers that only need the value."""
    return _get_session_source_status(session_id)[0]


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
            r"(?i)(?:^|[\s\[{(,:;])(?:traceback|error\b|failed\b|failure\b|file\s+not\s+found\b|no\s+such\s+file\b|cannot\s+find\s+the\s+(?:file|path)\b|ENOENT\b|timed?\s*out\b|timeout\b)",
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
        "collection_status": "session_unknown",
        "collection_error": "",
    }
    resolved, how = resolve_session_id(session_id or "")
    if not resolved:
        empty["session_id_source"] = how
        return empty
    db_path = config.state_db_path()
    if not db_path.is_file():
        empty["session_id"] = resolved
        empty["session_id_source"] = how
        empty["collection_status"] = "db_absent"
        return empty
    connection = _open_db()
    if not connection:
        empty["session_id"] = resolved
        empty["session_id_source"] = how
        empty["collection_status"] = "db_unavailable"
        return empty
    try:
        sql = (
            "SELECT m.role, m.content, m.tool_name, m.timestamp FROM messages m "
            "LEFT JOIN sessions s ON s.id = m.session_id "
            "WHERE m.session_id = ? AND m.active = 1"
        )
        params: List[Any] = [resolved]
        skipped_sources = config.skip_session_sources()
        if skipped_sources:
            placeholders = ",".join("?" for _ in skipped_sources)
            sql += f" AND (s.source IS NULL OR LOWER(s.source) NOT IN ({placeholders}))"
            params.extend(skipped_sources)
        sql += " ORDER BY m.timestamp DESC LIMIT ?"
        params.append(limit)
        rows = connection.execute(sql, tuple(params)).fetchall()
        messages: List[Dict[str, Any]] = []
        tool_errors: List[Dict[str, Any]] = []
        corrections: List[Dict[str, Any]] = []
        error_items: List[Dict[str, Any]] = []
        for row in reversed(rows):
            # Every string from SQLite is scrubbed at this single extraction
            # boundary so evidence, journals, and returned tool results inherit it.
            role = _one_line(scrub_text(str(row["role"] or "")))[:32].lower()
            if role not in {"user", "assistant", "tool", "system"}:
                role = "unknown"
            content = scrub_text(str(row["content"] or ""))
            tool_name = _one_line(
                scrub_text(str(row["tool_name"] or ""))
            )[:120]
            shown = content[:400] + ("…" if len(content) > 400 else "")
            messages.append({"role": role, "content": shown, "tool_name": tool_name})
            if role == "tool" and _is_error_content(content):
                bounded = (
                    content
                    if len(content) <= 4000
                    else content[:1000] + "\n…\n" + content[-3000:]
                )
                pattern_content = _UNTRUSTED_TOOL_TAG.sub("", bounded)
                tool_errors.append({"tool": tool_name, "snippet": pattern_content[:300]})
                error_items.append({
                    "tool": tool_name,
                    "content": pattern_content,
                    "session_id": resolved,
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
            "collection_status": "ok",
            "collection_error": "",
        }
    except Exception as exc:
        safe_error = scrub_text(str(exc))
        logger.warning("Current-session evidence query failed: %s", safe_error)
        empty["session_id"] = resolved
        empty["session_id_source"] = how
        empty["collection_status"] = "query_error"
        empty["collection_error"] = safe_error[:300]
        return empty
    finally:
        connection.close()


def collect_cross_session_patterns(
    days: Optional[int] = None,
    max_rows: Optional[int] = -1,
    *,
    since_ts: Optional[float] = None,
    max_sessions: Optional[int] = None,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    if not config.cross_session_enabled():
        if strict:
            raise IOError("Cross-session pattern collection is disabled")
        return []
    connection = _open_db()
    if not connection:
        if strict:
            raise IOError("Cross-session database is unavailable")
        return []
    if max_rows == -1:
        max_rows = config.cross_session_max_rows()
    since = (
        since_ts
        if since_ts is not None
        else time.time() - ((days or config.cross_session_days()) * 86400)
    )
    sql = (
        "SELECT m.session_id, m.tool_name, m.content, m.timestamp FROM messages m "
        "LEFT JOIN sessions s ON s.id = m.session_id "
        "WHERE m.role = 'tool' AND m.active = 1 AND m.timestamp >= ?"
    )
    params: List[Any] = [since]
    skipped_sources = config.skip_session_sources()
    if skipped_sources:
        placeholders = ",".join("?" for _ in skipped_sources)
        sql += f" AND (s.source IS NULL OR LOWER(s.source) NOT IN ({placeholders}))"
        params.extend(skipped_sources)
    sql += " ORDER BY m.timestamp DESC"
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
        rows_seen = 0

        def iter_items():
            nonlocal rows_seen
            for row in cursor:
                rows_seen += 1
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
                    "tool": _one_line(
                        scrub_text(str(row["tool_name"] or ""))
                    )[:120],
                    "content": _UNTRUSTED_TOOL_TAG.sub("", bounded),
                    "session_id": sid,
                    "ts": row["timestamp"] or 0,
                }

        full_audit = since_ts is not None and max_rows is None and max_sessions is None
        result = patterns.extract_patterns(
            iter_items(), limit=None if full_audit else 10
        )
        if max_rows is not None and rows_seen >= max_rows:
            logger.warning(
                "Cross-session row limit reached (%d); interactive evidence may be truncated",
                max_rows,
            )
        return result
    except Exception as exc:
        safe_error = scrub_text(str(exc))
        logger.warning("Cross-session query failed: %s", safe_error)
        if strict:
            raise IOError(f"Cross-session query failed: {safe_error}") from exc
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
    except Exception as exc:
        logger.warning("Cannot retrieve skill items: %s", scrub_text(str(exc)))
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
    except Exception as exc:
        logger.warning("Cannot read memory snippets: %s", scrub_text(str(exc)))
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


def _persistence_snapshot(directory: Path) -> Dict[str, Any]:
    """Inspect plugin runtime storage without creating, mutating, or pruning it."""
    def file_metric(path: Path) -> Dict[str, Any]:
        try:
            if not path.exists():
                return {"state": "absent", "bytes": 0}
            if not path.is_file():
                return {"state": "unknown", "bytes": None}
            return {"state": "ok", "bytes": path.stat().st_size}
        except Exception:
            return {"state": "unknown", "bytes": None}

    journal_metric = file_metric(journal.journal_read_path())
    journal_metric.update({"physical_lines": 0, "logical_entries": 0})
    if journal_metric["state"] == "ok":
        try:
            with journal.journal_read_path().open("r", encoding="utf-8", errors="replace") as handle:
                journal_metric["physical_lines"] = sum(1 for line in handle if line.strip())
            loaded, state = journal._load_entries_safe()
            journal_metric["state"] = state
            journal_metric["logical_entries"] = len(loaded) if state == "ok" else None
        except Exception:
            journal_metric["state"] = "unknown"
            journal_metric["logical_entries"] = None

    backup_metric: Dict[str, Any] = {"state": "absent", "count": 0, "bytes": 0}
    backup_dir = directory / "backups"
    try:
        if backup_dir.exists() and not backup_dir.is_dir():
            backup_metric = {"state": "unknown", "count": None, "bytes": None}
        elif backup_dir.is_dir():
            files = [path for path in backup_dir.iterdir() if path.is_file()]
            backup_metric = {
                "state": "ok",
                "count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
    except Exception:
        backup_metric = {"state": "unknown", "count": None, "bytes": None}

    ledger_metric = file_metric(ledger.stats_read_path())
    if ledger_metric["state"] == "ok":
        try:
            ledger.load_stats()
            ledger_metric["readable"] = True
        except IOError:
            ledger_metric.update({"state": "unreadable", "readable": False})
    else:
        ledger_metric["readable"] = ledger_metric["state"] == "absent"

    prompt_metric = file_metric(journal.prompt_notes_read_path())
    if prompt_metric["state"] == "ok":
        notes = journal.load_prompt_notes()
        prompt_metric["readable"] = notes is not None
        if notes is None:
            prompt_metric.update({"state": "unreadable", "not_injected_count": None})
        else:
            prompt_metric["not_injected_count"] = sum(
                1
                for note in notes
                if _stored_prompt_note_content_error(note["content"])
            )
    else:
        prompt_metric["readable"] = prompt_metric["state"] == "absent"
        prompt_metric["not_injected_count"] = 0 if prompt_metric["readable"] else None
    metrics = (journal_metric, backup_metric, ledger_metric, prompt_metric)
    total_complete = all(isinstance(metric.get("bytes"), int) for metric in metrics)
    byte_values = [
        metric.get("bytes")
        for metric in metrics
        if isinstance(metric.get("bytes"), int)
    ]
    total_bytes = sum(byte_values)
    return {
        "journal": journal_metric,
        "backups": backup_metric,
        "ledger": ledger_metric,
        "prompt_notes": prompt_metric,
        "total_bytes": total_bytes,
        "total_bytes_complete": total_complete,
        "total_bytes_is_lower_bound": not total_complete,
        "warning_threshold_bytes": _PERSISTENCE_WARNING_BYTES,
        "over_warning_threshold": total_bytes >= _PERSISTENCE_WARNING_BYTES,
    }


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
    migration = journal.migration_status()
    persistence = _persistence_snapshot(jdir)

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
            _, state = journal._load_entries_safe()
            if state != "ok":
                raise IOError(f"journal state is {state}")
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
    if not persistence["total_bytes_complete"]:
        warnings.append({
            "code": "persistence_size_unknown",
            "message": (
                "One or more runtime stores could not be sized; the displayed "
                "storage value is only a lower bound"
            ),
        })
    if persistence["over_warning_threshold"]:
        warnings.append({
            "code": "persistence_growth",
            "message": (
                "Refine runtime data uses "
                f"{persistence['total_bytes']} bytes, above the read-only status "
                f"warning threshold of {persistence['warning_threshold_bytes']} bytes"
            ),
        })
    for store_name in ("ledger", "prompt_notes"):
        if persistence[store_name].get("state") == "unreadable":
            warnings.append({
                "code": f"{store_name}_unreadable",
                "message": f"The refine {store_name.replace('_', '-')} store is unreadable",
            })
    invalid_prompt_notes = persistence["prompt_notes"].get("not_injected_count")
    if isinstance(invalid_prompt_notes, int) and invalid_prompt_notes:
        warnings.append({
            "code": "prompt_notes_invalid",
            "message": (
                f"{invalid_prompt_notes} stored prompt note(s) do not meet the current "
                "injection policy and will not be injected"
            ),
        })
    if migration.get("outcome") == "failed":
        warnings.append({
            "code": "journal_migration_failed",
            "message": (
                "Runtime-data migration failed; refine is using the intact legacy "
                f"store at {migration.get('active_dir') or jdir}"
            ),
        })
    if migration.get("rename_warning"):
        warnings.append({
            "code": "journal_migration_rename_failed",
            "message": "Runtime data migrated, but the legacy directory could not be renamed",
        })
    plugin_source_collision = False
    try:
        plugin_source_collision = (jdir / "plugin.yaml").is_file()
    except Exception as exc:
        logger.debug("Cannot inspect plugin source collision: %s", scrub_text(str(exc)))
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
        except Exception as exc:
            logger.warning("Cannot read session message count: %s", scrub_text(str(exc)))
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
        "persistence": persistence,
        "last_auto_event": last_auto_event(),
        "recent_auto_events": recent_auto_events(),
        "migration": migration,
        "migration_outcome": migration.get("outcome", "not_checked"),
        "session_id": sid,
        "session_id_source": sid_source,
        "session_message_count": session_message_count,
        "session_source": _get_session_source(sid) if sid else "",
        "skip_session_sources": config.skip_session_sources(),
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
    try:
        with journal.mutation_lock():
            _reconcile_pending()
            journal_entries = journal.entries()
    except Exception as exc:
        safe_error = scrub_text(str(exc))
        logger.error("Audit journal read failed: %s", safe_error)
        return {
            "success": False,
            "complete": False,
            "rows": [],
            "report": "Audit incomplete: the refine journal is unreadable; no conclusions were drawn.",
        }
    try:
        ledger_earliest = ledger.earliest_created_ts()
    except IOError as exc:
        safe_error = scrub_text(str(exc))
        logger.error("Audit ledger read failed: %s", safe_error)
        return {
            "success": False,
            "complete": False,
            "rows": [],
            "report": "Audit incomplete: the refine ledger is unreadable; no conclusions were drawn.",
        }

    journal_times = [
        float(entry.get("ts", 0))
        for entry in journal_entries
        if entry.get("outcome") == "applied"
        and isinstance(entry.get("proposal"), dict)
        and entry["proposal"].get("action") in ("create", "patch")
        and entry.get("ts")
    ]
    earliest_candidates = [value for value in [ledger_earliest, *journal_times] if value]
    earliest = min(earliest_candidates) if earliest_candidates else None

    complete = True
    current: Optional[List[Dict[str, Any]]] = []
    if earliest:
        try:
            current = collect_cross_session_patterns(
                since_ts=earliest,
                max_rows=None,
                max_sessions=None,
                strict=True,
            )
        except Exception as exc:
            logger.error("Audit pattern collection failed: %s", scrub_text(str(exc)))
            current = None
            complete = False
    rows = ledger.audit(current, journal_entries=journal_entries)
    report = ledger.format_audit(rows)
    if not complete:
        report = (
            "⚠ Audit incomplete: trajectory recurrence could not be measured; "
            "recurrence-dependent verdicts remain unknown.\n\n" + report
        )
    return {"success": True, "complete": complete, "rows": rows, "report": report}


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
    if not journal.prompt_note_content_is_structurally_safe(content):
        return "Prompt note cannot contain markup or context-control characters"
    if not 1 <= len(lines) <= 2:
        return "Prompt note must contain one or two non-empty policy lines"
    if any(
        line.startswith(("-", "*", "#")) or re.match(r"^\d+[.)]\s", line)
        for line in lines
    ):
        return "Prompt note must be a policy, not a list or procedure"
    if not _PROMPT_NOTE_FORMAT.match(lines[0]):
        return "Prompt note must use 'When <specific condition>, <one action>.'"
    if len(lines) > 1 and not _PROMPT_NOTE_FORMAT.match(lines[1]):
        return "Every line of a prompt note must use 'When <specific condition>, <one action>.'"
    if any(
        _RESOURCE_REFERENCE.search(line) or _has_host_reference(line)
        for line in lines
    ):
        if any(_RESOURCE_NETWORK_OR_SHELL.search(line) for line in lines):
            return "Prompt note cannot reference URLs, commands, or shell syntax"
        if any(_has_host_reference(line) for line in lines):
            return "Prompt note cannot reference hosts"
        return "Prompt note cannot reference file paths or environment variables"
    if any(_OVERRIDE_INTENT.search(line) for line in lines):
        return "Prompt note cannot override prior guidance"
    for line in lines:
        condition_match = _PROMPT_NOTE_CONDITION.match(line)
        if not condition_match or _HIGHER_PRIORITY_GUIDANCE.search(condition_match.group(1)):
            return "Prompt note condition cannot refer to higher-priority guidance"
        action_match = _PROMPT_NOTE_ACTION.match(line)
        if not action_match or not _PROMPT_NOTE_SAFE_ACTION.fullmatch(action_match.group(1)):
            return "Prompt note action must match an approved behavioral policy"
    rendered = "Refine notes:\n- " + content
    per_note_limit = max(
        1, config.prompt_notes_max_chars() // config.prompt_notes_max_count()
    )
    if check_rendered_size and len(rendered) > per_note_limit:
        return (
            f"Prompt note is too large for its per-note rendered context budget ({len(rendered)} chars; max "
            f"{per_note_limit})"
        )
    return None


def _stored_prompt_note_content_error(content: Any) -> Optional[str]:
    """Return the semantic injection error for a structurally stored note."""
    safe_content = scrub_text(str(content)).strip()
    if not safe_content:
        return "Prompt note is empty after scrubbing"
    return _prompt_note_content_error(safe_content, check_rendered_size=False)


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

    Returns None (no conflict) only when baseline is a well-formed dict with
    exists=True and a valid 64-hex-char sha256 that matches the current state.

    Returns an error string when:
      - baseline is absent or not a dict (unsafe: patch was built without
        verifying the target content);
      - baseline has invalid structure (exists != True, or sha256 malformed);
      - the current state diverges from the planning baseline.
    """
    import re as _re

    baseline = proposal.get("refine_baseline")
    name = str(proposal.get("name", ""))
    if not isinstance(baseline, dict):
        return (
            f"Skill '{name}': patch requires a locally grounded refine_baseline "
            "(absent or not a dict)"
        )
    exists = baseline.get("exists")
    sha = str(baseline.get("sha256", ""))
    if exists is not True or not _re.fullmatch(r"[0-9a-f]{64}", sha):
        return (
            f"Skill '{name}': patch has malformed refine_baseline "
            f"(exists={exists!r}, sha256 valid={bool(_re.fullmatch(r'[0-9a-f]{{64}}', sha))})"
        )
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
    """Write a non-mutating journal entry. Accepts all journal.log kwargs including llm_meta."""
    try:
        return journal.log(**kwargs)
    except Exception as exc:
        logger.error("Cannot write refine journal: %s", scrub_text(str(exc)))
        return None


def record_evidence_failure(
    session_id: str,
    collection_status: str,
    collection_error: str = "",
    *,
    trigger: str = "auto",
    timeout: float = 30.0,
) -> Optional[str]:
    """Wait off the host callback, then durably record unavailable evidence."""
    safe_status = _one_line(scrub_text(collection_status))[:64] or "unknown"
    safe_error = _one_line(scrub_text(collection_error))[:300]
    message = f"Current-session evidence is unavailable ({safe_status})."
    try:
        with journal.mutation_lock(timeout=timeout):
            _, state = journal._load_entries_safe()
            if state == "unreadable":
                raise IOError("journal is unreadable")
            return journal.log(
                trigger=trigger,
                reason=message,
                session_id=session_id,
                proposal={
                    "action": "no_op",
                    "reason": message,
                    "expected_outcome": "",
                },
                outcome="evidence_unavailable",
                error=safe_error or message,
            )
    except Exception as exc:
        logger.error(
            "Cannot durably record evidence failure: %s", scrub_text(str(exc))
        )
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
    dry_run: bool = False,
) -> Dict[str, Any]:
    trigger = "auto" if auto else "manual"
    started = time.time()
    safe_reason = scrub_text(reason)

    # Fail closed when journal is unreadable: without history the budget, dedup,
    # and context guards are all bypassed. Must be distinguishable from no_op.
    _, journal_state = journal._load_entries_safe()
    if journal_state == "unreadable":
        return {
            "success": False,
            "outcome": "journal_unreadable",
            "message": "Journal could not be read; refine did not run to avoid bypassing budget limits.",
            "reversible": False,
        }

    resolved_session, resolved_source = resolve_session_id(session_id or "")
    if not resolved_session:
        evidence = {
            "messages": [],
            "error_count": 0,
            "tool_errors": [],
            "error_patterns": [],
            "user_corrections": [],
            "session_id": "",
            "session_id_source": resolved_source,
        }
        return {
            "success": False,
            "outcome": "session_unknown",
            "message": "Cannot identify the current session; refine did not run.",
            "evidence": evidence,
            "reversible": False,
        }

    # Resolve machine-generated sources before reading any private trajectory.
    session_db_source, source_lookup_status = _get_session_source_status(
        resolved_session
    )
    skip_sources = config.skip_session_sources()
    if session_db_source and session_db_source.lower() in skip_sources:
        proposal = {
            "action": "no_op",
            "reason": f"Session source '{session_db_source}' is configured to be skipped.",
            "expected_outcome": "",
        }
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or proposal["reason"],
            session_id=resolved_session,
            proposal=proposal,
            outcome="skipped_session_source",
        )
        response = {
            "success": bool(entry_id),
            "outcome": "skipped_session_source",
            "message": (
                f"Session source '{session_db_source}' is in skip_session_sources; "
                "refine did not run."
            ),
            "evidence": {
                "messages": [],
                "session_id": resolved_session,
                "session_id_source": resolved_source,
                "session_source": session_db_source,
                "source_lookup_status": source_lookup_status,
            },
            "reversible": False,
        }
        if entry_id:
            response["journal_id"] = entry_id
        else:
            response["message"] += " The skip decision could not be journaled."
        return response

    if not dry_run and journal.daily_limit_reached():
        return {
            "success": False,
            "message": f"Daily edit limit reached ({config.max_edits_per_day()}). "
            f"Applied/pending/prepared today: {journal.count_today_applied()}.",
            "evidence": {
                "session_id": resolved_session,
                "session_id_source": resolved_source,
                "session_source": session_db_source,
                "source_lookup_status": source_lookup_status,
            },
        }

    # Resolve the LLM target once per pass so every call within it uses the same
    # model. This makes the choice deterministic and attributable in the journal.
    try:
        _effective = config.effective_llm_target()
        _run_target: Dict[str, str] = {}
        if _effective.get("provider") and config.llm_allow_provider_override():
            _run_target["provider"] = _effective["provider"]
        if _effective.get("model") and config.llm_allow_model_override():
            _run_target["model"] = _effective["model"]
        _run_target_source = _effective.get("source", "host_default")
        _run_target_issues = [str(i) for i in _effective.get("issues", []) if i]
    except Exception:
        _run_target = {}
        _run_target_source = "unknown"
        _run_target_issues = ["the effective model could not be resolved"]
    _run_target_unusable = bool(
        _run_target_issues
        and not _run_target
        and _run_target_source in ("unknown", "host_default")
    )

    _min_signal_required = config.min_signal_required()
    _min_pattern_count = config.min_pattern_count()
    evidence_limit = 60
    if _min_signal_required and config.reviewer_fallback_enabled():
        evidence_limit = max(evidence_limit, config.reviewer_min_messages())
    evidence = collect_evidence(session_id=resolved_session, limit=evidence_limit)
    evidence["session_id_source"] = resolved_source
    evidence["session_source"] = session_db_source
    evidence["source_lookup_status"] = source_lookup_status
    session = resolved_session
    session_source = resolved_source
    collection_status = str(evidence.get("collection_status", "ok"))
    if collection_status != "ok":
        failure_message = (
            "Current-session evidence is unavailable "
            f"({collection_status}); refine did not infer an empty session."
        )
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or failure_message,
            session_id=session,
            proposal={
                "action": "no_op",
                "reason": failure_message,
                "expected_outcome": "",
            },
            outcome="evidence_unavailable",
            error=failure_message,
        )
        response = {
            "success": False,
            "outcome": "evidence_unavailable",
            "failure": collection_status,
            "message": failure_message,
            "evidence": evidence,
            "reversible": False,
        }
        if entry_id:
            response["journal_id"] = entry_id
        else:
            response["message"] += " The failure could not be journaled."
        return response
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
        role = _one_line(message["role"])[:32].lower()
        if role not in {"user", "assistant", "tool", "system"}:
            role = "unknown"
        content = _one_line(str(message["content"])[:400])
        if role == "tool":
            # Tool metadata is untrusted too: keep the complete physical record
            # inside one plugin-owned boundary after removing forged variants.
            tool_name = _one_line(message.get("tool_name", ""))[:120]
            record = f"tool={tool_name or '?'} | {content}"
            safe_record = _UNTRUSTED_TOOL_TAG.sub("", record)
            lines.append(
                f"[tool] <untrusted_tool_result>{safe_record}</untrusted_tool_result>"
            )
        else:
            lines.append(f"[{role}] {content}")
    evidence_text = "\n".join(lines)
    proposal_context = safe_reason
    reviewer_context = ""
    _signal_path = "gate_disabled"
    if _min_signal_required and not patterns.has_signal(
        error_patterns, corrections, min_count=_min_pattern_count
    ):
        _signal_path = "no_signal"
        should_review = (
            config.reviewer_fallback_enabled()
            and len(evidence.get("messages", [])) >= config.reviewer_min_messages()
            and _reviewer_cooldown_elapsed()
        )
        if should_review:
            reviewer = _llm.review_fallback(llm, evidence_text, target=_run_target)
            reviewer_call_meta = _llm.last_call_meta()
            reviewer_llm_meta = {
                "requested_provider": _run_target.get("provider", ""),
                "requested_model": _run_target.get("model", ""),
                "target_source": _run_target_source,
                **{k: v for k, v in reviewer_call_meta.items() if k in (
                    "reported_model", "latency_ms", "output_tokens", "output_mode"
                )},
            }
            if _run_target_issues:
                reviewer_llm_meta["target_issues"] = _run_target_issues
            rationale = scrub_text(str(reviewer.get("rationale", "")))
            decision = "approved" if reviewer.get("should_refine") else "declined"
            reviewer_reason = f"Reviewer {decision}: {rationale}"
            reviewer_failure = scrub_text(str(reviewer.get("failure", "")).strip())
            reviewer_target_issue = bool(
                not reviewer_failure
                and _run_target_unusable
                and not reviewer.get("should_refine")
            )
            reviewer_outcome = (
                (
                    "llm_incomplete"
                    if reviewer_failure in {"malformed", "truncated", "no_final_text"}
                    else "llm_error"
                )
                if reviewer_failure
                else ("target_issue" if reviewer_target_issue else "no_op")
            )
            reviewer_error = (
                (
                    "The reviewer returned an incomplete or malformed verdict."
                    if reviewer_failure in {"malformed", "truncated", "no_final_text"}
                    else (
                        "The host trust policy denied the reviewer model call."
                        if reviewer_failure == "llm_trust_denied"
                        else "The reviewer model call failed."
                    )
                )
                if reviewer_failure
                else (
                    "The configured refine model target is unusable."
                    if reviewer_target_issue
                    else ""
                )
            )
            reviewer_entry_id = _journal_nonmutation(
                trigger="reviewer",
                reason=reviewer_reason,
                session_id=session,
                proposal={
                    "action": "no_op",
                    "reason": reviewer_reason,
                    "expected_outcome": "",
                },
                outcome=reviewer_outcome,
                error=reviewer_error,
                llm_meta=reviewer_llm_meta,
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
            if reviewer_failure:
                return {
                    "success": False,
                    "outcome": reviewer_outcome,
                    "failure": reviewer_failure,
                    "message": reviewer_error,
                    "journal_id": reviewer_entry_id,
                    "llm_called": True,
                    "reviewer": "failed",
                    "evidence": evidence,
                    "llm_meta": reviewer_llm_meta,
                    "reversible": False,
                }
            if not reviewer.get("should_refine"):
                proposal = {
                    "action": "no_op",
                    "reason": reviewer_reason,
                    "expected_outcome": "",
                }
                if reviewer_target_issue:
                    return {
                        "success": False,
                        "outcome": "target_issue",
                        "failure": "target_configuration",
                        "message": "The configured refine model target is unusable.",
                        "journal_id": reviewer_entry_id,
                        "proposal": proposal,
                        "llm_called": True,
                        "reviewer": "declined",
                        "evidence": evidence,
                        "llm_meta": reviewer_llm_meta,
                        "reversible": False,
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
            reviewer_context = reviewer_instructions
            _signal_path = "reviewer_approved"
        else:
            proposal = {
                "action": "no_op",
                "reason": f"No repeated failure (min {_min_pattern_count}x) and no explicit correction.",
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

    if _signal_path == "gate_disabled" and _min_signal_required:
        _signal_path = "gate_opened"

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
        reviewer_context=reviewer_context,
        skill_content_loader=journal.read_skill_content,
        target=_run_target,
    )
    # Capture metadata from the LLM call that produced this proposal.
    llm_meta = _llm.last_call_meta()
    _run_llm_meta = {
        "requested_provider": _run_target.get("provider", ""),
        "requested_model": _run_target.get("model", ""),
        "target_source": _run_target_source,
        "signal_path": _signal_path,
        **{k: v for k, v in llm_meta.items() if k in (
            "reported_model", "latency_ms", "output_tokens", "output_mode"
        )},
    }
    if _run_target_issues:
        _run_llm_meta["target_issues"] = _run_target_issues
    proposal = sanitize(proposal)
    proposal = dict(
        proposal,
        expected_outcome=_llm.normalize_expected_outcome(
            proposal.get("expected_outcome")
        ),
    )
    _offered_fps = {
        str(pattern.get("fingerprint", ""))
        for pattern in error_patterns[:patterns.FORMAT_PATTERNS_LIMIT]
        if pattern.get("fingerprint")
    }
    _proposal_fp = str(proposal.get("pattern_fingerprint", "") or "")
    _run_llm_meta["fingerprint_offered"] = len(_offered_fps)
    _run_llm_meta["grounded"] = bool(
        _proposal_fp and _proposal_fp in _offered_fps
    )
    failure = scrub_text(str(proposal.get("failure", "")).strip())
    if failure:
        failure_messages = {
            "truncated": "The refine proposal was cut off before it completed.",
            "malformed": "The refine proposal was malformed and could not be read.",
            "no_final_text": (
                "The model returned only reasoning and no final refine proposal."
            ),
            "llm_call_error": "The refine model call failed.",
            "llm_trust_denied": "The host trust policy denied the refine model call.",
            "local_safety": scrub_text(str(proposal.get("reason", "")))
            or "The refine proposal could not be completed safely.",
        }
        failure_message = failure_messages.get(
            failure, "The refine proposal could not be completed."
        )
        if failure in ("llm_call_error", "llm_trust_denied"):
            failure_outcome = "llm_error"
        elif failure == "local_safety":
            failure_outcome = "safety_blocked"
        else:
            failure_outcome = "llm_incomplete"
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or failure_message,
            session_id=session,
            proposal=proposal,
            outcome=failure_outcome,
            error=failure_message,
            llm_meta=_run_llm_meta,
        )
        response = {
            "success": False,
            "outcome": failure_outcome,
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
        "session_id_source": session_source,
        "session_source": session_db_source,
        "source_lookup_status": source_lookup_status,
        "messages": len(evidence.get("messages", [])),
        "errors": evidence.get("error_count", 0),
        "fingerprint_offered": _run_llm_meta["fingerprint_offered"],
        "grounded": _run_llm_meta["grounded"],
    }

    if _run_target_unusable and proposal.get("action") == "no_op":
        failure_message = "The configured refine model target is unusable."
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or failure_message,
            session_id=session,
            proposal=proposal,
            outcome="target_issue",
            error=failure_message,
            llm_meta=_run_llm_meta,
        )
        response = {
            "success": False,
            "outcome": "target_issue",
            "failure": "target_configuration",
            "message": failure_message,
            "proposal": proposal,
            "evidence": evidence_summary,
            "llm_called": True,
            "llm_meta": _run_llm_meta,
            "reversible": False,
        }
        if entry_id:
            response["journal_id"] = entry_id
        return response

    # ── Dry-run exit: show what would happen, apply nothing ────────────────
    if dry_run:
        import difflib as _difflib

        dry_proposal = proposal
        if proposal.get("action") == "multi":
            # Normalize each edit so the user sees the final form.
            edits = [
                _normalize_edit(sanitize(edit), session)
                for edit in proposal.get("edits", [])
                if isinstance(edit, dict)
            ]
            dry_proposal = dict(proposal, edits=edits)
        else:
            dry_proposal = _normalize_edit(proposal, session)

        # Build a diff for patch proposals.
        diff_text = ""
        max_diff_chars = _llm.MAX_CONTENT_CHARS
        truncated = False

        def _build_diff(name: str, new_content: str) -> str:
            old_content = journal.read_skill_content(name) or ""
            old_lines = old_content.splitlines()
            new_lines = new_content.splitlines()
            diff_lines = list(_difflib.unified_diff(
                old_lines, new_lines, fromfile=f"a/{name}", tofile=f"b/{name}", lineterm=""
            ))
            return "\n".join(diff_lines)

        if dry_proposal.get("action") == "patch" and dry_proposal.get("kind") == "skill":
            name = str(dry_proposal.get("name", ""))
            content = str(dry_proposal.get("content", ""))
            if name and content:
                raw_diff = _build_diff(name, content)
                if len(raw_diff) > max_diff_chars:
                    diff_text = scrub_text(raw_diff[:max_diff_chars]) + "\n… [truncated]"
                    truncated = True
                else:
                    diff_text = scrub_text(raw_diff)
        elif dry_proposal.get("action") == "multi":
            diff_parts = []
            for edit in dry_proposal.get("edits", []):
                if edit.get("action") == "patch" and edit.get("kind") == "skill":
                    name = str(edit.get("name", ""))
                    content = str(edit.get("content", ""))
                    if name and content:
                        diff_parts.append(_build_diff(name, content))
            if diff_parts:
                combined = "\n".join(diff_parts)
                if len(combined) > max_diff_chars:
                    diff_text = scrub_text(combined[:max_diff_chars]) + "\n… [truncated]"
                    truncated = True
                else:
                    diff_text = scrub_text(combined)

        # Journal the dry run so /refine audit shows it was considered.
        dry_run_entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or "dry-run",
            session_id=session,
            proposal=dry_proposal,
            outcome="dry_run",
            llm_meta=_run_llm_meta,
        )
        if not dry_run_entry_id:
            return {
                "success": False,
                "outcome": "journal_error",
                "message": "Dry-run proposal was generated, but its journal write failed.",
                "proposal": dry_proposal,
                "evidence": evidence_summary,
                "llm_called": True,
                "llm_meta": _run_llm_meta,
                "reversible": False,
                "edits_applied": 0,
            }

        return {
            "success": True,
            "outcome": "dry_run",
            "message": "Dry run: proposal shown, nothing applied.",
            "journal_id": dry_run_entry_id,
            "proposal": dry_proposal,
            "diff": diff_text,
            "diff_truncated": truncated,
            "evidence": evidence_summary,
            "llm_called": True,
            "llm_meta": _run_llm_meta,
            "reversible": False,
            "edits_applied": 0,
        }

    if proposal.get("action") == "multi":
        transaction = _apply_transaction(
            proposal,
            trigger=trigger,
            safe_reason=safe_reason,
            session=session,
            started=started,
            llm_meta=_run_llm_meta,
        )
        transaction["evidence"] = evidence_summary
        transaction["llm_meta"] = _run_llm_meta
        return transaction

    proposal = _normalize_edit(proposal, session)

    if proposal.get("action") == "no_op":
        entry_id = _journal_nonmutation(
            trigger=trigger,
            reason=safe_reason or proposal.get("reason", ""),
            session_id=session,
            proposal=proposal,
            outcome="no_op",
            llm_meta=_run_llm_meta,
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
            "evidence": evidence_summary,
            "reversible": False,
            "llm_meta": _run_llm_meta,
        }

    response = _apply_edit(
        proposal,
        trigger=trigger,
        safe_reason=safe_reason,
        session=session,
        started=started,
        llm_meta=_run_llm_meta,
    )
    response["evidence"] = evidence_summary
    response["llm_meta"] = _run_llm_meta
    return response


def _apply_edit(
    proposal: Dict[str, Any],
    *,
    trigger: str,
    safe_reason: str,
    session: str,
    started: float,
    group: Optional[Dict[str, Any]] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
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
            llm_meta=llm_meta,
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
                llm_meta=llm_meta,
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
                llm_meta=llm_meta,
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
                llm_meta=llm_meta,
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
                llm_meta=llm_meta,
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
                llm_meta=llm_meta,
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
            llm_meta=llm_meta,
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
                llm_meta=llm_meta,
            )
        except Exception as exc:
            logger.warning(
                "Ledger unreadable; edit was applied but attribution was skipped: %s",
                scrub_text(str(exc)),
            )

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
    llm_meta: Optional[Dict[str, Any]] = None,
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
            llm_meta=llm_meta,
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
    # Also rejects patches with missing or malformed baselines (fail closed).
    stale_edits: List[int] = []
    for index, edit in enumerate(edits):
        normalized = edit_proposal(edit)
        if (
            normalized.get("kind") == "skill"
            and normalized.get("action") == "patch"
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
                    llm_meta=llm_meta,
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
                    llm_meta=llm_meta,
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
            llm_meta=llm_meta,
        )
        results.append(item)
        if not item.get("success"):
            stop_reason = (
                f"An earlier edit of transaction {group_id} did not complete"
            )
            break
        if item.get("outcome") == "pending_approval":
            stop_reason = (
                "An earlier edit is pending host approval; remaining edits "
                "cannot proceed until the approval is resolved"
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
            llm_meta=llm_meta,
        )

    # "Recoverable" is deliberately wider than "successful": an edit whose host
    # mutation landed but whose journal finalization then failed still owns a
    # recovery id and must appear in the list the message points the user at.
    recoverable = [item for item in results if int(item.get("edits_applied", 0) or 0)]
    succeeded = [item for item in results if item.get("success")]
    recoveries = _recoveries_for(recoverable)
    skipped = len(edits) - len(results)
    elapsed = time.time() - started

    has_pending = any(item.get("outcome") == "pending_approval" for item in results)
    if len(succeeded) == len(edits) and not dropped and not has_pending:
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
        rendered_stop = scrub_text(stop_reason)[:160] or "edits were not attempted"
        if rendered_stop.startswith("Daily "):
            rendered_stop = "daily " + rendered_stop[6:]
        message += f" | stopped: {rendered_stop}; {skipped} edit(s) not attempted"
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
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Serialize a run, reconcile approvals, and preserve every recovery id."""
    started = time.time()
    with journal.mutation_lock():
        try:
            _reconcile_pending()
        except IOError:
            return {
                "success": False,
                "outcome": "journal_unreadable",
                "message": "Journal could not be read; refine did not run to avoid bypassing budget limits.",
                "reversible": False,
            }

        if dry_run:
            # Dry-run: one proposal pass, no apply, no budget consumed.
            return _refine_once(
                llm, reason=scrub_text(reason), session_id=session_id,
                auto=auto, dry_run=True,
            )

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
