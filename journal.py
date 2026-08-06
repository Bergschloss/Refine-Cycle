"""Append-only JSONL journal + rollback for the refine plugin.

Every refine action (applied, no_op, rejected, error) is logged
to ``refine_journal.jsonl``. Before applying a skill/memory edit we
back up the current state so it can be rolled back.
"""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .config import journal_dir, max_edits_per_day
except ImportError:
    from config import journal_dir, max_edits_per_day  # noqa: F811 — standalone test

logger = logging.getLogger(__name__)

_BACKUPS_DIR_NAME = "backups"
_JOURNAL_FILE_NAME = "refine_journal.jsonl"


def ensure_dirs() -> Path:
    """Ensure journal and backup directories exist. Returns journal_dir Path."""
    d = journal_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / _BACKUPS_DIR_NAME).mkdir(exist_ok=True)
    return d


def journal_path() -> Path:
    return ensure_dirs() / _JOURNAL_FILE_NAME


def backups_dir() -> Path:
    return ensure_dirs() / _BACKUPS_DIR_NAME


# ── journal I/O ─────────────────────────────────────────────────────────────


def _load_entries() -> List[Dict[str, Any]]:
    """Read all journal entries (append-safe, corrupted lines skipped)."""
    jp = journal_path()
    entries: List[Dict[str, Any]] = []
    if not jp.is_file():
        return entries
    try:
        for line in jp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping corrupt journal line")
    except Exception as exc:
        logger.warning("Failed to read journal: %s", exc)
    return entries


def _append_entry(entry: Dict[str, Any]) -> None:
    jp = journal_path()
    try:
        with open(jp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Failed to write journal entry: %s", exc)


def count_today_applied() -> int:
    """Count applied edits for today (UTC)."""
    today = datetime.now(timezone.utc).date()
    count = 0
    for entry in _load_entries():
        if entry.get("outcome") != "applied":
            continue
        ts = entry.get("ts", 0)
        if not ts:
            continue
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except (OSError, OverflowError, ValueError):
            continue
        if dt == today:
            count += 1
    return count


def daily_limit_reached() -> bool:
    return count_today_applied() >= max_edits_per_day()


def log(
    *,
    trigger: str,
    reason: str,
    session_id: str,
    proposal: Dict[str, Any],
    outcome: str,
    backup_path: str = "",
    error: str = "",
) -> str:
    """Append a journal entry. Returns the entry id (UUID)."""
    entry_id = uuid.uuid4().hex[:12]
    entry: Dict[str, Any] = {
        "id": entry_id,
        "ts": time.time(),
        "trigger": trigger,
        "reason": reason,
        "session_id": str(session_id)[:64],
        "proposal": proposal,
        "outcome": outcome,
        "backup_path": backup_path,
        "error": error,
    }
    _append_entry(entry)
    return entry_id


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    """Find a journal entry by id."""
    for entry in _load_entries():
        if entry.get("id") == entry_id:
            return entry
    return None


# ── backups ─────────────────────────────────────────────────────────────────


def backup_skill(name: str) -> Optional[Path]:
    """Copy current SKILL.md for *name* into the backups dir.

    Returns the backup path, or None if the skill doesn't exist.
    """
    from tools.skills_tool import skill_view as _skill_view

    # skill_view returns a JSON string — parse it (dict input also handled).
    try:
        raw = _skill_view(name)
        if isinstance(raw, dict):
            result = raw
        else:
            result = json.loads(raw)
    except Exception:
        logger.warning("Cannot view skill '%s' for backup", name)
        return None

    if not isinstance(result, dict) or not result.get("success"):
        return None

    skill_dir = result.get("skill_dir", "") or result.get("path", "")
    if not skill_dir:
        return None
    skill_dir = Path(skill_dir)

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None

    ts = int(time.time() * 1000)
    backup = backups_dir() / f"{ts}_skill_{name}.bak"
    shutil.copy2(skill_md, backup)
    logger.info("Backed up skill '%s' → %s", name, backup)
    return backup


def backup_memory(target: str) -> Optional[str]:
    """Extract current memory/user entry text as a backup string.

    Returns the full current content, or "" when the store is empty (still a
    valid backup — "memory was empty"), or None on read failure.
    """
    from tools.memory_tool import MemoryStore

    try:
        store = MemoryStore()
        store.load_from_disk()
        entries = store._entries_for(target)  # noqa: SLF001 — internal but safe
        if not entries:
            return ""
        return "\n\n---\n\n".join(entries)
    except Exception as exc:
        logger.warning("Cannot back up memory: %s", exc)
        return None


def rollback_skill(entry_id: str) -> Dict[str, Any]:
    """Restore a skill from its backup. Returns a result dict."""
    entry = get_entry(entry_id)
    if not entry:
        return {"success": False, "error": f"Journal entry {entry_id} not found"}

    backup_path = entry.get("backup_path", "")
    if not backup_path or not Path(backup_path).is_file():
        return {"success": False, "error": f"Backup file not found: {backup_path}"}

    proposal = entry.get("proposal", {})
    name = proposal.get("name", "")
    if not name:
        return {"success": False, "error": "Journal entry has no skill name"}

    backup_content = Path(backup_path).read_text(encoding="utf-8")
    try:
        from tools.skill_manager_tool import skill_manage
        result_str = skill_manage(action="edit", name=name, content=backup_content)
        result = json.loads(result_str)
    except Exception as exc:
        return {"success": False, "error": f"Rollback failed: {exc}"}

    if result.get("success"):
        log(
            trigger=f"rollback:{entry_id}",
            reason=f"Rollback of skill '{name}'",
            session_id=entry.get("session_id", ""),
            proposal=proposal,
            outcome="rolled_back",
            backup_path=backup_path,
        )
    return result


def rollback_memory(entry_id: str) -> Dict[str, Any]:
    """Restore memory from a backup string. Returns a result dict."""
    entry = get_entry(entry_id)
    if not entry:
        return {"success": False, "error": f"Journal entry {entry_id} not found"}

    backup_path = entry.get("backup_path", "")
    if not backup_path or not Path(backup_path).is_file():
        return {"success": False, "error": f"Memory backup file not found: {backup_path}"}

    proposal = entry.get("proposal", {})
    # The proposal schema only knows "skill"|"memory"; "user" memory entries
    # are not produced by refine, but keep the mapping defensive anyway.
    kind = proposal.get("kind", "memory")
    target = "user" if kind == "user" else "memory"
    old_content = Path(backup_path).read_text(encoding="utf-8")

    try:
        from tools.memory_tool import MemoryStore
        store = MemoryStore()
        store.load_from_disk()

        # Remove all current and restore from backup
        entries = [e.strip() for e in old_content.split("\n\n---\n\n") if e.strip()]
        # Just overwrite the target file — crude but works with MemoryStore
        from tools.memory_tool import get_memory_dir as _get_mem_dir
        mem_dir = _get_mem_dir()
        filename = "USER.md" if target == "user" else "MEMORY.md"
        Path(mem_dir, filename).write_text(old_content, encoding="utf-8")

        log(
            trigger=f"rollback:{entry_id}",
            reason=f"Rollback of {target} memory",
            session_id=entry.get("session_id", ""),
            proposal=proposal,
            outcome="rolled_back",
            backup_path=backup_path,
        )
        return {"success": True, "message": f"Memory '{target}' restored"}
    except Exception as exc:
        return {"success": False, "error": f"Memory rollback failed: {exc}"}
