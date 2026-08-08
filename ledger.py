"""Timestamp-aware usefulness ledger for refine-created entries."""

import json
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import journal
    from .config import journal_dir, state_db_path
    from .sanitization import scrub_text
except ImportError:
    import journal  # type: ignore
    from config import journal_dir, state_db_path  # noqa: F811
    from sanitization import scrub_text  # type: ignore

logger = logging.getLogger(__name__)
_STATS_FILE_NAME = "skill_stats.json"


def stats_path() -> Path:
    path = journal_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path / _STATS_FILE_NAME


def stats_read_path() -> Path:
    """Ledger location for readers, without creating anything."""
    return journal_dir() / _STATS_FILE_NAME


def load_stats() -> Dict[str, Any]:
    """Load the skill stats ledger, distinguishing absence from corruption.

    Returns {} only when the file is genuinely absent. On read/parse error the
    function raises IOError so callers do not overwrite a corrupted or locked
    file with a single entry.
    """
    path = stats_read_path()
    if not path.is_file():
        return {}
    try:
        raw = journal._retry_on_contention(
            lambda: path.read_text(encoding="utf-8"),
            journal._READ_RETRY_BUDGET_SECONDS,
        )
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("ledger root must be an object")
        return data
    except Exception as exc:
        logger.error("Cannot read skill stats: %s", exc)
        raise IOError(f"Ledger unreadable: {scrub_text(str(exc))}") from exc


def _save_stats(stats: Dict[str, Any]) -> None:
    path = stats_path()
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        journal._replace_with_retry(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def record_edit(
    proposal: Dict[str, Any],
    journal_id: str,
    *,
    outcome: str = "applied",
    pending_id: str = "",
    llm_meta: Optional[Dict[str, Any]] = None,
) -> None:
    name = str(proposal.get("name", "")).strip()
    if not name:
        return
    kind = str(proposal.get("kind", "skill") or "skill")
    # Skills keep their bare name as the key so existing statistics, the audit
    # table, and the prompt overview keep resolving. Other kinds are namespaced,
    # because one transaction can legitimately create a skill and a same-named
    # memory entry, and a shared key would hide one of them from the audit.
    key = name if kind == "skill" else f"{kind}:{name}"
    with journal.mutation_lock():
        stats = load_stats()
        # Before kind-qualified keys existed, every entry used its bare name.
        # Move a legacy non-skill row out of the skill namespace before any
        # write, including a same-named skill write, so history and versions
        # survive the upgrade instead of becoming a duplicate audit row.
        legacy = stats.get(name)
        if isinstance(legacy, dict):
            legacy_kind = str(legacy.get("kind", "skill") or "skill")
            if legacy_kind != "skill":
                legacy_key = f"{legacy_kind}:{name}"
                stats.setdefault(legacy_key, legacy)
                del stats[name]
        previous = stats.get(key, {})
        now = time.time()
        same_edit = previous.get("journal_id") == journal_id
        created_ts = previous.get("created_ts", now) if same_edit else now
        previous_version = previous.get("version", 1 if previous else 0)
        version = previous_version if same_edit else previous_version + 1
        stats[key] = {
            "created_ts": created_ts,
            "updated_ts": now,
            "version": version,
            "journal_id": journal_id,
            "name": name,
            "kind": kind,
            "action": proposal.get("action", ""),
            "pattern_fingerprint": proposal.get("pattern_fingerprint", ""),
            "expected_outcome": (
                scrub_text(proposal["expected_outcome"]).strip()
                if isinstance(proposal.get("expected_outcome"), str)
                else ""
            ),
            "outcome": outcome,
            "pending_id": pending_id,
        }
        # LLM attribution for the audit display is additive and survives later
        # reconciliation calls that carry no model metadata.
        reported_model = ""
        if isinstance(llm_meta, dict) and llm_meta.get("reported_model"):
            reported_model = scrub_text(str(llm_meta["reported_model"]))[:60]
        elif isinstance(previous, dict) and previous.get("reported_model"):
            reported_model = scrub_text(str(previous["reported_model"]))[:60]
        if reported_model:
            stats[key]["reported_model"] = reported_model
        _save_stats(stats)


def record_journal_state(entry: Dict[str, Any]) -> None:
    """Mirror a reconciled journal state without resetting its creation time."""
    proposal = entry.get("proposal", {})
    record_edit(
        proposal,
        str(entry.get("id", "")),
        outcome=str(entry.get("outcome", "")),
        pending_id=str(entry.get("pending_id", "")),
        llm_meta=entry.get("llm_meta") if isinstance(entry.get("llm_meta"), dict) else None,
    )


def earliest_created_ts() -> Optional[float]:
    values = [
        float(meta.get("created_ts", 0))
        for meta in load_stats().values()
        if isinstance(meta, dict)
        and meta.get("created_ts")
        and meta.get("outcome", "applied") == "applied"
    ]
    return min(values) if values else None


# ── usage counting ─────────────────────────────────────────────────────────


def _count_uses_with_scope(name: str, since_ts: float) -> Tuple[Optional[int], str]:
    """Return (count, scope): since_exact, all_time, since_approx, unavailable."""
    try:
        from tools import skill_usage as usage

        for function_name in ("get_usage_count", "usage_count", "get_use_count"):
            function = getattr(usage, function_name, None)
            if not callable(function):
                continue
            try:
                return int(function(name, since_ts=since_ts)), "since_exact"
            except TypeError:
                try:
                    return int(function(name)), "all_time"
                except Exception:
                    continue
            except Exception:
                continue
    except ImportError:
        pass

    try:
        path = state_db_path()
        if not path.is_file():
            return None, "unavailable"
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            escaped_name = (
                name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            row = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE active = 1 "
                "AND timestamp > ? AND (tool_name = ? OR content LIKE ? ESCAPE '\\')",
                (since_ts, name, f"%/{escaped_name}%"),
            ).fetchone()
            return (int(row[0]) if row else 0), "since_approx"
        finally:
            connection.close()
    except Exception as exc:
        logger.debug("Usage fallback failed for %s: %s", name, exc)
        return None, "unavailable"


def count_uses(name: str, since_ts: float) -> Optional[int]:
    """Compatibility API returning the best available count."""
    return _count_uses_with_scope(name, since_ts)[0]


def unused_skills(min_age_days: int = 14) -> List[str]:
    cutoff = time.time() - (min_age_days * 86400)
    result: List[str] = []
    for name, meta in load_stats().items():
        if (
            meta.get("kind") != "skill"
            or meta.get("created_ts", 0) > cutoff
            or meta.get("outcome", "applied") != "applied"
        ):
            continue
        uses, scope = _count_uses_with_scope(name, meta.get("created_ts", 0))
        # since_approx is a heuristic (LIKE '%/name%' in message content); absence
        # of detected uses is not absence of use.  Only since_exact (the host's
        # authoritative usage counter) can prove a skill is genuinely unused.
        # Round 6 allowed since_approx here; Round 8 reverted that after review.
        if uses == 0 and scope == "since_exact":
            result.append(name)
    return result[:10]


# ── audit ──────────────────────────────────────────────────────────────────


def _merge_journal_stats(
    stats: Dict[str, Any], journal_entries: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Fill ledger attribution gaps from the journal lifecycle authority."""
    merged = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in stats.items()
    }
    tracked_outcomes = {
        "prepared", "applied", "pending_approval", "rollback_prepared",
        "pending_rollback", "rolled_back", "rejected",
    }
    ordered = sorted(
        (entry for entry in (journal_entries or []) if isinstance(entry, dict)),
        key=lambda entry: float(entry.get("ts", 0) or 0),
    )
    for entry in ordered:
        if entry.get("outcome") not in tracked_outcomes:
            continue
        proposal = entry.get("proposal")
        if not isinstance(proposal, dict) or proposal.get("action") not in ("create", "patch"):
            continue
        name = str(proposal.get("name", "")).strip()
        if not name:
            continue
        kind = str(proposal.get("kind", "skill") or "skill")
        key = name if kind == "skill" else f"{kind}:{name}"
        entry_id = str(entry.get("id", ""))
        timestamp = float(entry.get("ts", 0) or 0)
        existing = merged.get(key)
        if isinstance(existing, dict) and existing.get("journal_id") == entry_id:
            existing["outcome"] = entry.get("outcome", existing.get("outcome", ""))
            existing["pending_id"] = entry.get("pending_id", existing.get("pending_id", ""))
            continue
        if isinstance(existing, dict):
            try:
                existing_ts = float(existing.get("updated_ts", existing.get("created_ts", 0)) or 0)
            except (TypeError, ValueError):
                existing_ts = 0
            if existing_ts > timestamp:
                continue
            try:
                version = max(1, int(existing.get("version", 1) or 1)) + 1
            except (TypeError, ValueError):
                version = 2
        else:
            version = 1
        meta = {
            "created_ts": timestamp,
            "updated_ts": float(entry.get("finalized_ts", timestamp) or timestamp),
            "version": version,
            "journal_id": entry_id,
            "name": name,
            "kind": kind,
            "action": proposal.get("action", ""),
            "pattern_fingerprint": proposal.get("pattern_fingerprint", ""),
            "expected_outcome": (
                scrub_text(proposal["expected_outcome"]).strip()
                if isinstance(proposal.get("expected_outcome"), str)
                else ""
            ),
            "outcome": entry.get("outcome", ""),
            "pending_id": entry.get("pending_id", ""),
        }
        llm_meta = entry.get("llm_meta")
        if isinstance(llm_meta, dict) and llm_meta.get("reported_model"):
            meta["reported_model"] = scrub_text(str(llm_meta["reported_model"]))[:60]
        elif isinstance(existing, dict) and existing.get("reported_model"):
            meta["reported_model"] = existing["reported_model"]
        merged[key] = meta
    return merged


def audit(
    current_patterns: Optional[List[Dict[str, Any]]] = None,
    *,
    journal_entries: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    patterns_available = current_patterns is not None
    by_fingerprint = {
        str(item.get("fingerprint", "")): item for item in (current_patterns or [])
    }
    now = time.time()
    rows: List[Dict[str, Any]] = []
    merged_stats = _merge_journal_stats(load_stats(), journal_entries)
    for key, meta in sorted(merged_stats.items()):
        # Legacy rows have no explicit name; their key is the name.
        name = str(meta.get("name") or key)
        created = meta.get("created_ts", 0) or now
        age_days = max(0, int((now - created) // 86400))
        try:
            version = max(1, int(meta.get("version", 1) or 1))
        except (TypeError, ValueError):
            version = 1
        try:
            updated_ts = float(meta.get("updated_ts", created) or created)
        except (TypeError, ValueError):
            updated_ts = created
        outcome = meta.get("outcome", "applied")
        fingerprint = str(meta.get("pattern_fingerprint", "") or "")
        recurred: Optional[bool] = None

        if outcome == "prepared":
            uses, usage_scope = None, "unavailable"
            verdict = "recovery needed"
        elif outcome == "pending_approval":
            uses, usage_scope = None, "unavailable"
            verdict = "pending approval"
        elif outcome in ("rollback_prepared", "pending_rollback"):
            uses, usage_scope = None, "unavailable"
            verdict = "rollback pending"
        elif outcome == "rolled_back":
            uses, usage_scope = None, "unavailable"
            verdict = "rolled back"
        elif outcome == "rejected":
            uses, usage_scope = None, "unavailable"
            verdict = "rejected"
        else:
            uses, usage_scope = _count_uses_with_scope(name, created)
            if fingerprint and patterns_available:
                hit = by_fingerprint.get(fingerprint)
                # Pattern exists but has no last_ts -> still active (recurred)
                if hit is not None and not (hit.get("last_ts") or 0) > created:
                    recurred = True if hit.get("last_ts") is None else False
                else:
                    recurred = bool(hit and (hit.get("last_ts") or 0) > created)

            if recurred is True:
                verdict = "did not help"
            elif uses == 0 and age_days >= 14 and usage_scope == "since_exact":
                # since_approx cannot prove unused or working: the DB fallback
                # detects usage by pattern-matching message content, which can
                # miss real uses and over-count incidental mentions.  Reporting
                # "unused" on an approximate zero would mislead the operator into
                # deleting a skill that is actually in use.
                verdict = "unused"
            elif (
                uses
                and uses > 0
                and recurred is False
                and usage_scope == "since_exact"
            ):
                verdict = "working"
            else:
                verdict = "too early" if age_days < 14 else "unclear"

        if version >= 3 and verdict == "unclear":
            verdict = "churning"

        rows.append({
            "name": name,
            "kind": meta.get("kind", "skill"),
            "age_days": age_days,
            "version": version,
            "updated_ts": updated_ts,
            "uses": uses,
            "usage_scope": usage_scope,
            "pattern_recurred": recurred,
            "verdict": verdict,
            "journal_id": meta.get("journal_id", ""),
            "outcome": outcome,
            "reported_model": (
                scrub_text(str(meta.get("reported_model", "")))[:60]
                if meta.get("reported_model")
                else ""
            ),
            "expected_outcome": (
                scrub_text(meta["expected_outcome"]).strip()
                if isinstance(meta.get("expected_outcome"), str)
                else ""
            ),
        })
    return rows


def format_audit(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No refine-created skills recorded yet."
    lines = [f"Refine-created entries ({len(rows)}):", ""]
    lines.append(
        f"  {'name':<28} {'age':>5}  {'ver':>3}  {'uses':>7}  {'recurred':>8}  verdict"
    )
    for row in rows:
        scope = row.get("usage_scope")
        if row["uses"] is None:
            uses = "?"
        elif scope == "all_time":
            uses = f"all:{row['uses']}"
        elif scope == "since_approx":
            uses = f"~{row['uses']}"
        else:
            uses = str(row["uses"])
        recurred = {True: "yes", False: "no", None: "—"}[row["pattern_recurred"]]
        kind = str(row.get("kind", "skill") or "skill")
        name = str(row.get("name", ""))
        display_name = name if kind == "skill" else f"{kind}:{name}"
        lines.append(
            f"  {display_name[:28]:<28} {str(row['age_days']) + 'd':>5}  "
            f"{'v' + str(row.get('version', 1)):>3}  {uses:>7}  "
            f"{recurred:>8}  {row['verdict']}"
        )
        expected_outcome = str(row.get("expected_outcome", "") or "—")
        lines.append(f"      expects: {expected_outcome[:57]}")
        reported_model = str(row.get("reported_model", "") or "")
        if reported_model:
            lines.append(f"      model: {reported_model[:40]}")

    candidates = [row for row in rows if row["verdict"] in ("unused", "did not help")]
    if candidates:
        lines.extend(["", "Candidates for removal:"])
        for row in candidates:
            kind = str(row.get("kind", "skill") or "skill")
            name = str(row.get("name", ""))
            display_name = name if kind == "skill" else f"{kind}:{name}"
            lines.append(
                f"  {display_name} — /refine rollback {row['journal_id']}"
            )
        lines.extend(["", "Nothing was deleted. Run the command yourself if you agree."])
    lines.extend([
        "",
        "Use labels: plain = timestamped host count, ~ = trajectory estimate, all: = host all-time aggregate.",
        "All-time aggregates are not used to claim post-edit usage.",
    ])
    return "\n".join(lines)
