"""Error fingerprinting and aggregation.

The point of this module: "the same failure happened again" is a question about
*shapes*, not strings. Two errors that differ only by a request id, a row count
or a temp path are the same failure. Normalizing those away and hashing what
remains turns a flat list of error text into countable patterns — which is what
makes "this recurs" a fact the plugin can assert instead of a guess it delegates
to the model.

Pure functions only: no DB, no config, no LLM. Everything here is unit-testable
without a Hermes host.
"""

import hashlib
import re
from typing import Any, Dict, Iterable, List

# Order matters: timestamps and paths must be replaced before bare integers,
# otherwise the digit rule eats the parts that make them recognizable.
_NORMALIZERS = [
    # ISO-ish timestamps and clock times
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "T"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"), "T"),
    # Quoted literals — usually the volatile part of a message
    (re.compile(r"'[^']*'"), "S"),
    (re.compile(r'"[^"]*"'), "S"),
    # URLs before paths (a URL contains slashes)
    (re.compile(r"https?://\S+"), "URL"),
    # Filesystem paths, POSIX and Windows
    (re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w.\-]+[\\/]){1,}[\w.\-]*"), "PATH"),
    # UUIDs, then any long hex run (ids, hashes, object addresses)
    (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "X"),
    (re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b"), "X"),
    # Whatever integers survive
    (re.compile(r"\b\d+\b"), "N"),
]

_TRACEBACK_MARKERS = ("Traceback (most recent call last)", 'File "', "  at ")


def normalize_error(content: str) -> str:
    """Reduce an error message to its invariant shape.

    ``HTTP 429 for /users/8821`` and ``HTTP 429 for /users/9134`` both normalize
    to ``http N for PATH`` — one pattern, not two.
    """
    if not content:
        return ""

    text = content.strip()

    # For a traceback, the only stable part is the final exception line; the
    # frames above it are noise that changes with every refactor.
    if any(marker in text for marker in _TRACEBACK_MARKERS):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[-1]

    for pattern, repl in _NORMALIZERS:
        text = pattern.sub(repl, text)

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:200]


def fingerprint(tool_name: str, content: str) -> str:
    """Stable short id for an error shape, scoped by the tool that produced it."""
    key = f"{tool_name or ''}|{normalize_error(content)}"
    return hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:12]


def extract_patterns(items: Iterable[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Group error occurrences into counted patterns.

    Args:
        items: dicts with ``tool``, ``content``, and optionally ``session_id``
            and ``ts``. Content is expected to be already scrubbed.
        limit: max patterns returned.

    Returns:
        Patterns sorted by (distinct sessions, total count) descending.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for item in items:
        content = str(item.get("content") or "")
        if not content:
            continue
        tool = str(item.get("tool") or "")
        fp = fingerprint(tool, content)
        ts = item.get("ts") or 0
        sid = str(item.get("session_id") or "")

        entry = grouped.get(fp)
        if entry is None:
            grouped[fp] = {
                "fingerprint": fp,
                "tool": tool,
                "sample": content[:300],
                "shape": normalize_error(content),
                "count": 1,
                "_sessions": {sid} if sid else set(),
                "first_ts": ts,
                "last_ts": ts,
            }
            continue

        entry["count"] += 1
        if sid:
            entry["_sessions"].add(sid)
        if ts:
            entry["first_ts"] = min(entry["first_ts"] or ts, ts)
            entry["last_ts"] = max(entry["last_ts"] or ts, ts)

    out: List[Dict[str, Any]] = []
    for entry in grouped.values():
        sessions = entry.pop("_sessions")
        entry["sessions_seen"] = max(1, len(sessions))
        out.append(entry)

    out.sort(key=lambda e: (e["sessions_seen"], e["count"]), reverse=True)
    return out[:limit]


def merge_patterns(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge pattern lists from different windows, taking the max of counters."""
    merged: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        for entry in group or []:
            fp = entry.get("fingerprint", "")
            if not fp:
                continue
            current = merged.get(fp)
            if current is None:
                merged[fp] = dict(entry)
                continue
            current["count"] = max(current.get("count", 0), entry.get("count", 0))
            current["sessions_seen"] = max(
                current.get("sessions_seen", 1), entry.get("sessions_seen", 1)
            )
            current["first_ts"] = min(
                current.get("first_ts") or entry.get("first_ts") or 0,
                entry.get("first_ts") or current.get("first_ts") or 0,
            )
            current["last_ts"] = max(
                current.get("last_ts") or 0, entry.get("last_ts") or 0
            )

    out = list(merged.values())
    out.sort(key=lambda e: (e.get("sessions_seen", 1), e.get("count", 0)), reverse=True)
    return out


def has_signal(
    patterns: List[Dict[str, Any]],
    corrections: List[Any],
    min_count: int = 2,
) -> bool:
    """Is there anything worth spending a model call on?

    True when a failure actually repeated (within a session or across sessions),
    or when the user explicitly corrected the agent. Everything else is a single
    one-off error, which teaches nothing generalizable.
    """
    if corrections:
        return True
    for entry in patterns or []:
        if entry.get("count", 0) >= min_count or entry.get("sessions_seen", 1) >= 2:
            return True
    return False


def format_patterns(patterns: List[Dict[str, Any]], limit: int = 8) -> str:
    """Render patterns as a compact block for the proposal prompt."""
    if not patterns:
        return "  (none)"
    lines = []
    for entry in patterns[:limit]:
        lines.append(
            "  [{count}x across {sessions} session(s)] {tool} — {sample} (fp:{fp})".format(
                count=entry.get("count", 1),
                sessions=entry.get("sessions_seen", 1),
                tool=entry.get("tool") or "?",
                sample=(entry.get("sample") or "").replace("\n", " ")[:160],
                fp=entry.get("fingerprint", "")[:8],
            )
        )
    return "\n".join(lines)
