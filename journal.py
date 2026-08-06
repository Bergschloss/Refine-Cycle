"""Durable append-only journal, mutation lock, approvals, and rollback."""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from .config import journal_dir, max_edits_per_day
    from .sanitization import sanitize, scrub_text
except ImportError:
    from config import journal_dir, max_edits_per_day  # noqa: F811
    from sanitization import sanitize, scrub_text  # noqa: F811

logger = logging.getLogger(__name__)

_BACKUPS_DIR_NAME = "backups"
_JOURNAL_FILE_NAME = "refine_journal.jsonl"
_PROMPT_NOTES_FILE_NAME = "prompt_notes.json"
_LOCK_FILE_NAME = ".mutation.lock"
_LOCK_STALE_SECONDS = 300
_THREAD_LOCK = threading.RLock()
_LOCK_STATE = threading.local()


def ensure_dirs() -> Path:
    directory = journal_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _BACKUPS_DIR_NAME).mkdir(exist_ok=True)
    return directory


def journal_path() -> Path:
    return ensure_dirs() / _JOURNAL_FILE_NAME


def backups_dir() -> Path:
    return ensure_dirs() / _BACKUPS_DIR_NAME


def prompt_notes_path() -> Path:
    """Return the plugin-owned prompt-note store; never a host memory path."""
    return ensure_dirs() / _PROMPT_NOTES_FILE_NAME


def normalize_prompt_note_session_id(session_id: Any) -> str:
    """Accept only a stable, already-safe hook/session identifier."""
    raw = str(session_id).strip()
    safe = scrub_text(raw).strip()
    return safe if raw and raw == safe and len(safe) <= 64 else ""


def _normalize_prompt_note(note: Any) -> Optional[Dict[str, str]]:
    """Validate one plugin-owned note and canonicalize legacy notes as global."""
    if not isinstance(note, dict):
        return None
    note_id = note.get("id")
    content = note.get("content")
    scope = note.get("scope", "global")
    if (
        not isinstance(note_id, str)
        or len(note_id) != 12
        or any(char not in "0123456789abcdef" for char in note_id)
        or not isinstance(content, str)
        or not content.strip()
        or scrub_text(content) != content
        or scope not in ("global", "session")
    ):
        return None
    normalized = {"id": note_id, "content": content, "scope": scope}
    if scope == "session":
        session_id = normalize_prompt_note_session_id(note.get("session_id", ""))
        if not session_id:
            return None
        normalized["session_id"] = session_id
    return normalized


def _load_prompt_notes() -> Optional[List[Dict[str, str]]]:
    """Return validated prompt notes, [] when absent, or None when unavailable."""
    path = prompt_notes_path()
    if not path.exists():
        return []
    if not path.is_file():
        logger.warning("Prompt-note store is not a regular file")
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        raw_notes = document.get("notes") if isinstance(document, dict) else None
        if not isinstance(raw_notes, list):
            raise ValueError("notes must be a list")
        notes: List[Dict[str, str]] = []
        seen_ids = set()
        for raw_note in raw_notes:
            note = _normalize_prompt_note(raw_note)
            if note is None or note["id"] in seen_ids:
                raise ValueError("unsafe prompt note")
            seen_ids.add(note["id"])
            notes.append(note)
        return notes
    except Exception as exc:
        logger.warning("Cannot read prompt-note store: %s", scrub_text(str(exc)))
        return None


def load_prompt_notes() -> Optional[List[Dict[str, str]]]:
    """Read safe prompt notes. Callers hold a mutation lock when consistency matters."""
    return _load_prompt_notes()


def _write_prompt_notes(notes: List[Dict[str, str]]) -> None:
    """Atomically persist only validated, already-scrubbed note objects."""
    safe_notes = []
    seen_ids = set()
    for raw_note in notes:
        note = _normalize_prompt_note(raw_note)
        if note is None or note["id"] in seen_ids:
            raise ValueError("Refusing to write an unsafe prompt note")
        seen_ids.add(note["id"])
        safe_notes.append(note)
    _atomic_write_text(
        prompt_notes_path(),
        json.dumps({"notes": safe_notes}, ensure_ascii=False, separators=(",", ":")),
    )


def prompt_note_content_exists(content: str) -> Optional[bool]:
    """Return None for unavailable storage so callers fail closed before mutation."""
    with mutation_lock():
        notes = _load_prompt_notes()
        if notes is None:
            return None
        return any(note["content"] == content for note in notes)


def normalize_prompt_note_content(content: str) -> str:
    """Canonicalize a note once so journal proof and storage always agree."""
    return scrub_text(str(content)).strip()


def new_prompt_note(
    content: str, *, scope: str = "global", session_id: str = ""
) -> Optional[Dict[str, str]]:
    """Preflight storage and allocate a stable ID without mutating the store."""
    candidate: Dict[str, str] = {
        "id": uuid.uuid4().hex[:12],
        "content": normalize_prompt_note_content(content),
        "scope": scope,
    }
    if scope == "session":
        candidate["session_id"] = session_id
    note = _normalize_prompt_note(candidate)
    if note is None:
        return None
    with mutation_lock():
        if _load_prompt_notes() is None:
            return None
        return note


def add_prompt_note(note: Dict[str, str]) -> Dict[str, Any]:
    """Persist one note atomically; this is plugin-owned and needs no host approval."""
    safe_note = _normalize_prompt_note(note)
    if safe_note is None:
        return {"success": False, "error": "Prompt note is invalid"}
    with mutation_lock():
        notes = _load_prompt_notes()
        if notes is None:
            return {"success": False, "error": "Prompt-note store is unavailable"}
        if any(
            item["id"] == safe_note["id"] or item["content"] == safe_note["content"]
            for item in notes
        ):
            return {"success": False, "error": "Prompt note already exists"}
        try:
            _write_prompt_notes(notes + [safe_note])
        except Exception as exc:
            return {
                "success": False,
                "error": f"Cannot persist prompt note: {scrub_text(str(exc))}",
            }
        return {"success": True, "note_id": safe_note["id"]}


def clear_session_prompt_notes(
    session_id: str, *, timeout: float = 30.0
) -> Optional[int]:
    """Remove all notes scoped to one ended/reset session; None means no mutation occurred.

    Host callbacks pass a short ``timeout`` so a running refine pass cannot stall
    the user's session-end or session-reset path behind the mutation lock.
    """
    safe_session_id = normalize_prompt_note_session_id(session_id)
    if not safe_session_id:
        return None
    with mutation_lock(timeout=timeout):
        notes = _load_prompt_notes()
        if notes is None:
            return None
        remaining = [
            note
            for note in notes
            if not (
                note.get("scope") == "session"
                and note.get("session_id") == safe_session_id
            )
        ]
        removed = len(notes) - len(remaining)
        if not removed:
            return 0
        try:
            _write_prompt_notes(remaining)
        except Exception as exc:
            logger.warning("Cannot clear session prompt notes: %s", scrub_text(str(exc)))
            return None
        return removed


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _try_clear_stale_lock(path: Path) -> None:
    """Clear only locks old enough to be stale, including malformed locks.

    A creator may have made the file but not yet written its JSON. The mtime is
    therefore authoritative for malformed/uninitialized locks and also guards
    against deleting a recently replaced valid lock with an old timestamp.
    """
    try:
        modified = path.stat().st_mtime
    except FileNotFoundError:
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
        created = float(data.get("created", 0))
    except Exception:
        pid, created = 0, 0
    try:
        modified = max(modified, path.stat().st_mtime)
    except FileNotFoundError:
        return
    freshness = max(modified, created) if created > 0 else modified
    if time.time() - freshness < _LOCK_STALE_SECONDS or _pid_is_alive(pid):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def _acquire_mutation_lock(*, wait: bool, timeout: float = 0.0) -> Iterator[bool]:
    """Acquire the re-entrant thread/process lock, optionally without waiting.

    ``timeout`` bounds the whole acquisition, in-process contention included.
    Bounding only the lock file would let a caller on a host callback thread wait
    forever behind another thread of the same process.
    """
    deadline = time.monotonic() + timeout
    if wait:
        acquired_thread = _THREAD_LOCK.acquire(timeout=timeout if timeout > 0 else -1)
    else:
        acquired_thread = _THREAD_LOCK.acquire(blocking=False)
    if not acquired_thread:
        yield False
        return
    try:
        depth = getattr(_LOCK_STATE, "depth", 0)
        if depth:
            _LOCK_STATE.depth = depth + 1
            try:
                yield True
            finally:
                _LOCK_STATE.depth -= 1
            return

        lock_path = ensure_dirs() / _LOCK_FILE_NAME
        token = uuid.uuid4().hex
        payload = json.dumps({"pid": os.getpid(), "created": time.time(), "token": token})
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, payload.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                _try_clear_stale_lock(lock_path)
                if not wait:
                    try:
                        fd = os.open(
                            str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                        )
                    except FileExistsError:
                        yield False
                        return
                    try:
                        os.write(fd, payload.encode("utf-8"))
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for refine mutation lock: {lock_path}")
                time.sleep(0.05)

        _LOCK_STATE.depth = 1
        try:
            yield True
        finally:
            _LOCK_STATE.depth = 0
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
                if current.get("token") == token:
                    lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not release refine mutation lock: %s", scrub_text(str(exc)))
    finally:
        _THREAD_LOCK.release()


@contextmanager
def mutation_lock(timeout: float = 30.0) -> Iterator[None]:
    """Serialize refine mutations across threads and processes."""
    with _acquire_mutation_lock(wait=True, timeout=timeout) as acquired:
        if not acquired:  # Another thread of this process still owns the lock.
            raise TimeoutError("Timed out waiting for refine mutation lock")
        yield


@contextmanager
def try_mutation_lock() -> Iterator[bool]:
    """Attempt mutation serialization once, without queueing behind another owner."""
    with _acquire_mutation_lock(wait=False) as acquired:
        yield acquired


# ── durable file I/O ───────────────────────────────────────────────────────


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace backup/stat files; journals use append-only writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _load_entries() -> List[Dict[str, Any]]:
    """Stream journal state, skipping corrupt lines and collapsing updates by id."""
    path = journal_path()
    if not path.is_file():
        return []
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt journal line")
                    continue
                entry_id = str(entry.get("id", ""))
                if not entry_id:
                    continue
                if entry_id not in latest:
                    order.append(entry_id)
                latest[entry_id] = entry
    except Exception as exc:
        logger.warning("Failed to read journal: %s", scrub_text(str(exc)))
        return []
    return [latest[entry_id] for entry_id in order]


def entries() -> List[Dict[str, Any]]:
    """Return the latest durable state of each logical journal record."""
    return _load_entries()


def recent_refinements(limit: int) -> List[Dict[str, Any]]:
    """Return capped create/patch outcomes in chronological order for model feedback."""
    try:
        capped_limit = max(0, int(limit))
    except (TypeError, ValueError):
        return []
    if not capped_limit:
        return []
    included_outcomes = {
        "applied", "pending_approval", "error", "rejected", "rolled_back"
    }
    refinements: List[Dict[str, Any]] = []
    for entry in entries():
        proposal = entry.get("proposal", {})
        if not isinstance(proposal, dict):
            continue
        if (
            proposal.get("action") in ("create", "patch")
            and entry.get("outcome") in included_outcomes
        ):
            refinements.append(entry)
    return refinements[-capped_limit:]


def last_attempt_ts(trigger: Optional[str] = None) -> Optional[float]:
    """Return the most recent durable attempt timestamp, optionally by trigger."""
    latest: Optional[float] = None
    for entry in _load_entries():
        if trigger is not None and entry.get("trigger") != trigger:
            continue
        try:
            timestamp = float(entry.get("ts"))
        except (TypeError, ValueError):
            continue
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def _append_entry(entry: Dict[str, Any]) -> None:
    """Append one fsynced JSON line without rewriting journal history."""
    safe_entry = sanitize(entry)
    record = json.dumps(safe_entry, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded = record.encode("utf-8")
    with mutation_lock():
        path = journal_path()
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            separator = b""
            if size:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    # Isolate a corrupt/partial prior tail so this valid record
                    # remains independently loadable.
                    separator = b"\n"
            handle.seek(0, os.SEEK_END)
            handle.write(separator + encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _new_entry(
    *,
    trigger: str,
    reason: str,
    session_id: str,
    proposal: Dict[str, Any],
    outcome: str,
    backup_path: str = "",
    error: str = "",
    recovery: Optional[Dict[str, Any]] = None,
    group: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "trigger": trigger,
        "reason": reason,
        "session_id": str(session_id)[:64],
        "proposal": proposal,
        "outcome": outcome,
        "backup_path": backup_path,
        "recovery": recovery or {},
        "error": error,
    }
    # A multi-edit transaction stays one durable record per edit, so rollback,
    # reconciliation, dedup, and the daily edit budget keep working unchanged.
    # ``group`` only reports which edits belonged together.
    if group:
        entry["group"] = group
    return entry


def log(
    *,
    trigger: str,
    reason: str,
    session_id: str,
    proposal: Dict[str, Any],
    outcome: str,
    backup_path: str = "",
    error: str = "",
    recovery: Optional[Dict[str, Any]] = None,
    group: Optional[Dict[str, Any]] = None,
) -> str:
    entry = _new_entry(
        trigger=trigger,
        reason=reason,
        session_id=session_id,
        proposal=proposal,
        outcome=outcome,
        backup_path=backup_path,
        error=error,
        recovery=recovery,
        group=group,
    )
    _append_entry(entry)
    return entry["id"]


def prepare(
    *,
    trigger: str,
    reason: str,
    session_id: str,
    proposal: Dict[str, Any],
    backup_path: str = "",
    recovery: Optional[Dict[str, Any]] = None,
    group: Optional[Dict[str, Any]] = None,
) -> str:
    return log(
        trigger=trigger,
        reason=reason,
        session_id=session_id,
        proposal=proposal,
        outcome="prepared",
        backup_path=backup_path,
        recovery=recovery,
        group=group,
    )


def finalize(
    entry_id: str,
    outcome: str,
    *,
    error: str = "",
    pending_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a new durable state for a logical record."""
    entry = get_entry(entry_id)
    if not entry:
        raise KeyError(f"Prepared journal entry {entry_id} not found")
    updated = dict(entry)
    updated["outcome"] = outcome
    updated["error"] = scrub_text(error)
    if pending_id is not None:
        updated["pending_id"] = scrub_text(str(pending_id))
        recovery = dict(updated.get("recovery", {}))
        recovery["pending_id"] = updated["pending_id"]
        updated["recovery"] = recovery
    updated["finalized_ts"] = time.time()
    _append_entry(updated)
    return sanitize(updated)


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    for entry in _load_entries():
        if entry.get("id") == entry_id:
            return entry
    return None


def is_reversible(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry or entry.get("outcome") not in ("applied", "prepared", "rollback_prepared"):
        return False
    proposal = entry.get("proposal", {})
    kind = proposal.get("kind")
    action = proposal.get("action")
    if kind == "skill" and action == "create":
        return bool(proposal.get("name") and proposal.get("content"))
    if kind == "skill" and action == "patch":
        return bool(entry.get("backup_path"))
    if kind in ("memory", "user"):
        return bool(entry.get("recovery"))
    if kind == "prompt":
        recovery = entry.get("recovery", {})
        return bool(
            action == "create"
            and recovery.get("type") == "prompt_note"
            and recovery.get("note_id")
            and proposal.get("content")
        )
    return False


def count_today_applied() -> int:
    """Count today's edits that are applied, reserved, or rollback-in-flight."""
    today = datetime.now(timezone.utc).date()
    consumed = {
        "applied", "pending_approval", "prepared", "rollback_prepared", "pending_rollback"
    }
    count = 0
    for entry in _load_entries():
        if entry.get("outcome") not in consumed:
            continue
        try:
            if datetime.fromtimestamp(entry.get("ts", 0), tz=timezone.utc).date() == today:
                count += 1
        except (OSError, OverflowError, ValueError, TypeError):
            continue
    return count


def daily_limit_reached() -> bool:
    return count_today_applied() >= max_edits_per_day()


def proposal_hash(proposal: Dict[str, Any]) -> str:
    key = "|".join([
        str(proposal.get("kind", "")),
        str(proposal.get("name", "")),
        hashlib.sha1(str(proposal.get("content", "")).encode("utf-8", "replace")).hexdigest(),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def was_applied_recently(proposal: Dict[str, Any], within_days: int) -> bool:
    target = proposal_hash(proposal)
    cutoff = time.time() - (within_days * 86400)
    consumed = {
        "applied", "pending_approval", "prepared", "rollback_prepared", "pending_rollback"
    }
    for entry in _load_entries():
        if entry.get("outcome") not in consumed:
            continue
        if (entry.get("ts") or 0) >= cutoff and proposal_hash(entry.get("proposal", {})) == target:
            return True
    return False


# ── recovery metadata and target-state proof ───────────────────────────────


def _read_skill_state(name: str) -> tuple:
    """Return (known, content); absence is known only from an explicit not-found."""
    from tools.skills_tool import skill_view

    try:
        raw = skill_view(name)
        result = raw if isinstance(raw, dict) else json.loads(raw)
    except Exception as exc:
        logger.warning("Cannot view skill '%s': %s", name, scrub_text(str(exc)))
        return False, None
    if not isinstance(result, dict):
        return False, None
    if not result.get("success"):
        error = str(result.get("error", "")).lower()
        return (True, None) if "not found" in error else (False, None)
    direct = result.get("content")
    if isinstance(direct, str):
        return True, direct
    skill_path = result.get("skill_dir", "") or result.get("path", "")
    if not skill_path:
        return False, None
    path = Path(skill_path)
    if path.is_dir():
        path = path / "SKILL.md"
    try:
        return True, path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return True, None
    except Exception:
        return False, None


def _skill_view_result(name: str) -> Optional[Dict[str, Any]]:
    """Compatibility view used by callers that require an existing skill."""
    known, content = _read_skill_state(name)
    return {"success": True, "content": content} if known and content is not None else None


def read_skill_content(name: str) -> Optional[str]:
    known, content = _read_skill_state(name)
    return content if known else None


def backup_skill(name: str) -> Optional[Path]:
    content = read_skill_content(name)
    if content is None:
        return None
    backup = backups_dir() / f"{int(time.time() * 1000)}_skill_{name}.bak"
    try:
        _atomic_write_text(backup, content)
    except Exception as exc:
        logger.warning("Cannot back up skill '%s': %s", name, scrub_text(str(exc)))
        return None
    return backup


def _memory_entries(target: str) -> Optional[List[str]]:
    from tools.memory_tool import MemoryStore

    try:
        store = MemoryStore()
        store.load_from_disk()
        return list(store._entries_for(target))  # noqa: SLF001 - host has no public reader
    except Exception as exc:
        logger.warning("Cannot read %s memory: %s", target, scrub_text(str(exc)))
        return None


def backup_memory(target: str) -> Optional[str]:
    entries_value = _memory_entries(target)
    if entries_value is None:
        return None
    return "\n\n---\n\n".join(entries_value)


def memory_recovery(target: str, content: str) -> Optional[Dict[str, Any]]:
    entries_value = _memory_entries(target)
    if entries_value is None:
        return None
    digest = hashlib.sha256(
        json.dumps(entries_value, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "type": "memory_append",
        "target": target,
        "index": len(entries_value),
        "prefix_digest": digest,
        "content": content,
    }


def _memory_prefix_matches(recovery: Dict[str, Any], values: List[str]) -> bool:
    index = recovery.get("index")
    if not isinstance(index, int) or index < 0 or index > len(values):
        return False
    digest = hashlib.sha256(
        json.dumps(list(values[:index]), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest == recovery.get("prefix_digest")


def target_matches_applied(entry: Dict[str, Any]) -> Optional[bool]:
    """Prove the proposal target; return None when target state is unavailable."""
    proposal = entry.get("proposal", {})
    kind = proposal.get("kind")
    if kind == "skill":
        known, content = _read_skill_state(str(proposal.get("name", "")))
        return (content == str(proposal.get("content", ""))) if known else None
    if kind in ("memory", "user"):
        recovery = entry.get("recovery", {})
        values = _memory_entries(str(recovery.get("target", "memory")))
        if values is None:
            return None
        index = recovery.get("index")
        return bool(
            _memory_prefix_matches(recovery, values)
            and isinstance(index, int)
            and index < len(values)
            and values[index] == recovery.get("content")
        )
    if kind == "prompt":
        recovery = entry.get("recovery", {})
        if recovery.get("type") != "prompt_note":
            return False
        notes = _load_prompt_notes()
        if notes is None:
            return None
        return any(
            note["id"] == recovery.get("note_id")
            and note["content"] == proposal.get("content", "")
            for note in notes
        )
    return False


def rollback_target_matches(entry: Dict[str, Any]) -> Optional[bool]:
    """Prove rollback state; return None when target state is unavailable."""
    proposal = entry.get("proposal", {})
    kind = proposal.get("kind")
    if kind == "skill":
        name = str(proposal.get("name", ""))
        if proposal.get("action") == "create":
            known, content = _read_skill_state(name)
            return (content is None) if known else None
        backup_path = Path(str(entry.get("backup_path", "")))
        if not backup_path.is_file():
            return False
        try:
            expected = backup_path.read_text(encoding="utf-8")
        except Exception:
            return False
        known, current = _read_skill_state(name)
        return (current == expected) if known else None
    if kind in ("memory", "user"):
        recovery = entry.get("recovery", {})
        values = _memory_entries(str(recovery.get("target", "memory")))
        if values is None:
            return None
        index = recovery.get("index")
        return bool(
            _memory_prefix_matches(recovery, values)
            and isinstance(index, int)
            and (index >= len(values) or values[index] != recovery.get("content"))
        )
    if kind == "prompt":
        recovery = entry.get("recovery", {})
        if recovery.get("type") != "prompt_note":
            return False
        notes = _load_prompt_notes()
        if notes is None:
            return None
        return not any(note["id"] == recovery.get("note_id") for note in notes)
    return False


def _pending_exists(subsystem: str, pending_id: str) -> Optional[bool]:
    """Return True/False for known approval state, None when host lookup failed."""
    if not pending_id:
        return False
    try:
        from tools.write_approval import get_pending

        raw = get_pending(subsystem, pending_id)
        result = json.loads(raw) if isinstance(raw, str) else raw
        return bool(result)
    except Exception as exc:
        logger.warning("Cannot query pending approval %s: %s", pending_id, scrub_text(str(exc)))
        return None


def reconcile() -> List[Dict[str, Any]]:
    """Lazily reconcile forward and rollback approvals from host and target state."""
    changed: List[Dict[str, Any]] = []
    for snapshot in _load_entries():
        entry_id = str(snapshot.get("id", ""))
        outcome = snapshot.get("outcome")
        proposal = snapshot.get("proposal", {})
        subsystem = "skills" if proposal.get("kind") == "skill" else "memory"
        try:
            if outcome == "prepared":
                applied_state = target_matches_applied(snapshot)
                if applied_state is True:
                    changed.append(finalize(entry_id, "applied"))
                continue
            if outcome == "pending_approval":
                pending = _pending_exists(subsystem, str(snapshot.get("pending_id", "")))
                if pending is not False:
                    # While the host record still exists (or its state cannot be
                    # queried), an already-matching target is not proof that this
                    # particular request was approved.
                    continue
                applied_state = target_matches_applied(snapshot)
                if applied_state is True:
                    changed.append(finalize(entry_id, "applied"))
                elif applied_state is False:
                    changed.append(finalize(entry_id, "rejected", error="Approval rejected"))
                continue
            if outcome == "rollback_prepared":
                if rollback_target_matches(snapshot) is True:
                    changed.append(finalize(entry_id, "rolled_back"))
                continue
            if outcome == "pending_rollback":
                pending = _pending_exists(subsystem, str(snapshot.get("pending_id", "")))
                if pending is not False:
                    continue
                rollback_state = rollback_target_matches(snapshot)
                if rollback_state is True:
                    changed.append(finalize(entry_id, "rolled_back"))
                elif rollback_state is False:
                    changed.append(
                        finalize(entry_id, "applied", error="Rollback approval rejected")
                    )
        except Exception as exc:
            logger.warning("Cannot reconcile journal entry %s: %s", entry_id, scrub_text(str(exc)))
    return changed


# ── rollback side effects ──────────────────────────────────────────────────


def _restore_applied(entry_id: str, error: str) -> None:
    try:
        finalize(entry_id, "applied", error=error)
    except Exception as exc:
        logger.warning("Cannot restore applied state for %s: %s", entry_id, scrub_text(str(exc)))


def rollback_skill(entry_id: str) -> Dict[str, Any]:
    entry = get_entry(entry_id)
    if not is_reversible(entry):
        return {"success": False, "error": f"Journal entry {entry_id} is not a reversible skill edit"}
    proposal = entry.get("proposal", {})
    if proposal.get("kind") != "skill":
        return {"success": False, "error": "Journal entry is not a skill edit"}
    name = str(proposal.get("name", ""))
    action = proposal.get("action")

    current = read_skill_content(name)
    expected = str(proposal.get("content", ""))
    if current != expected:
        return {"success": False, "error": f"Rollback conflict: skill '{name}' changed after refine applied it"}
    backup_content = ""
    if action != "create":
        backup_path = Path(str(entry.get("backup_path", "")))
        if not backup_path.is_file():
            return {"success": False, "error": f"Backup file not found: {backup_path}"}
        backup_content = backup_path.read_text(encoding="utf-8")

    try:
        if entry.get("outcome") != "rollback_prepared":
            entry = finalize(entry_id, "rollback_prepared")
    except Exception as exc:
        return {"success": False, "error": f"Cannot journal rollback intent: {scrub_text(str(exc))}"}

    try:
        from tools.skill_manager_tool import skill_manage

        raw = (
            skill_manage(action="delete", name=name)
            if action == "create"
            else skill_manage(action="edit", name=name, content=backup_content)
        )
        result = raw if isinstance(raw, dict) else json.loads(raw)
    except Exception as exc:
        error = f"Rollback failed: {scrub_text(str(exc))}"
        _restore_applied(entry_id, error)
        return {"success": False, "error": error}

    if not result.get("success"):
        error = scrub_text(str(result.get("error", "Rollback host operation failed")))
        _restore_applied(entry_id, error)
        return sanitize(result)

    if result.get("staged"):
        pending_id = str(result.get("pending_id", ""))
        if not pending_id:
            error = "Rollback was staged without a pending_id"
            _restore_applied(entry_id, error)
            return {"success": False, "error": error}
        try:
            finalize(entry_id, "pending_rollback", pending_id=pending_id)
        except Exception as exc:
            return {
                "success": False,
                "staged": True,
                "pending_id": pending_id,
                "error": (
                    "Rollback was reserved but pending state finalization failed; "
                    f"recovery id: {entry_id}. {scrub_text(str(exc))}"
                ),
            }
        result["message"] = "Rollback is pending approval; target has not been marked rolled back"
        return sanitize(result)

    current_entry = get_entry(entry_id) or entry
    if not rollback_target_matches(current_entry):
        error = "Rollback host reported success but the target state did not change"
        _restore_applied(entry_id, error)
        return {"success": False, "error": error}
    try:
        finalize(entry_id, "rolled_back")
    except Exception as exc:
        return {
            "success": False,
            "error": (
                "Rollback changed the target but journal finalization failed; "
                f"recovery id: {entry_id}. {scrub_text(str(exc))}"
            ),
        }
    result["message"] = result.get("message", f"Skill '{name}' rolled back")
    return sanitize(result)


def rollback_memory(entry_id: str) -> Dict[str, Any]:
    entry = get_entry(entry_id)
    if not is_reversible(entry):
        return {"success": False, "error": f"Journal entry {entry_id} is not a reversible memory edit"}
    recovery = entry.get("recovery", {})
    if recovery.get("type") != "memory_append":
        return {"success": False, "error": "Memory recovery metadata is missing"}
    target = str(recovery.get("target", "memory"))
    expected = recovery.get("content", "")
    index = recovery.get("index")

    try:
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        store.load_from_disk()
        values = store._entries_for(target)  # noqa: SLF001
        if not isinstance(index, int) or index < 0 or index >= len(values):
            return {"success": False, "error": "Memory rollback conflict: appended entry position changed"}
        if not _memory_prefix_matches(recovery, list(values)) or values[index] != expected:
            return {"success": False, "error": "Memory rollback conflict: target entry or earlier memory changed"}
        if entry.get("outcome") != "rollback_prepared":
            entry = finalize(entry_id, "rollback_prepared")
        del values[index]
        store.save_to_disk(target)
    except Exception as exc:
        latest = get_entry(entry_id) or entry
        if not rollback_target_matches(latest):
            _restore_applied(entry_id, scrub_text(str(exc)))
        return {"success": False, "error": f"Memory rollback failed: {scrub_text(str(exc))}"}

    latest = get_entry(entry_id) or entry
    if not rollback_target_matches(latest):
        error = "Memory rollback target state did not change"
        _restore_applied(entry_id, error)
        return {"success": False, "error": error}
    try:
        finalize(entry_id, "rolled_back")
    except Exception as exc:
        return {
            "success": False,
            "error": (
                "Memory rollback changed the target but journal finalization failed; "
                f"recovery id: {entry_id}. {scrub_text(str(exc))}"
            ),
        }
    return {"success": True, "message": f"Removed the exact appended {target} memory entry"}


def rollback_prompt_note(entry_id: str) -> Dict[str, Any]:
    """Remove only the unchanged plugin-owned note identified by this journal entry."""
    with mutation_lock():
        entry = get_entry(entry_id)
        if not is_reversible(entry):
            return {"success": False, "error": f"Journal entry {entry_id} is not a reversible prompt note"}
        proposal = entry.get("proposal", {})
        recovery = entry.get("recovery", {})
        if proposal.get("kind") != "prompt" or recovery.get("type") != "prompt_note":
            return {"success": False, "error": "Prompt-note recovery metadata is missing"}
        note_id = recovery.get("note_id")
        expected = proposal.get("content", "")
        notes = _load_prompt_notes()
        if notes is None:
            return {"success": False, "error": "Prompt-note store is unavailable"}
        index = next((i for i, note in enumerate(notes) if note["id"] == note_id), None)
        if index is None:
            return {"success": False, "error": "Prompt-note rollback conflict: note is missing"}
        if notes[index]["content"] != expected:
            return {"success": False, "error": "Prompt-note rollback conflict: note changed after refine applied it"}
        try:
            if entry.get("outcome") != "rollback_prepared":
                entry = finalize(entry_id, "rollback_prepared")
            _write_prompt_notes(notes[:index] + notes[index + 1:])
        except Exception as exc:
            _restore_applied(entry_id, scrub_text(str(exc)))
            return {"success": False, "error": f"Prompt-note rollback failed: {scrub_text(str(exc))}"}

        latest = get_entry(entry_id) or entry
        if not rollback_target_matches(latest):
            error = "Prompt-note rollback target state did not change"
            _restore_applied(entry_id, error)
            return {"success": False, "error": error}
        try:
            finalize(entry_id, "rolled_back")
        except Exception as exc:
            return {
                "success": False,
                "error": (
                    "Prompt-note rollback changed the target but journal finalization failed; "
                    f"recovery id: {entry_id}. {scrub_text(str(exc))}"
                ),
            }
        return {"success": True, "message": f"Removed prompt note {note_id}"}
