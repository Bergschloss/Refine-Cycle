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
    """Acquire the re-entrant thread/process lock, optionally without waiting."""
    if not _THREAD_LOCK.acquire(blocking=wait):
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
        deadline = time.monotonic() + timeout
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
        if not acquired:  # Defensive: blocking acquisition always either succeeds or raises.
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
) -> Dict[str, Any]:
    return {
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
) -> str:
    return log(
        trigger=trigger,
        reason=reason,
        session_id=session_id,
        proposal=proposal,
        outcome="prepared",
        backup_path=backup_path,
        recovery=recovery,
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
