"""Durable append-only journal, mutation lock, approvals, and rollback."""

import hashlib
import json
import logging
import os
import re
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
_MODEL_OVERRIDE_FILE_NAME = "model_override.json"
_LOCK_FILE_NAME = ".mutation.lock"
_LOCK_STALE_SECONDS = 300
_THREAD_LOCK = threading.RLock()
_LOCK_STATE = threading.local()

# Owned here rather than in the command layer, because this module owns the
# store: the same rule then applies to the command, to a hand-edited file, and
# to any future writer, instead of once per call site.
#
# Two shapes, not one. In ``/refine model <provider>/<model>`` the slash is the
# separator, so each half must not contain one. A model *id*, however, is very
# often namespaced — ``deepseek/deepseek-chat``, ``anthropic/claude-3.5-sonnet``
# — so the id rule has to allow it. Applying the token rule to a configured
# ``llm.model`` would silently discard exactly the ids people pin.
_MODEL_TOKEN_CHARS = r"[A-Za-z0-9._:\-]{1,120}"
_MODEL_TOKEN = re.compile(rf"^{_MODEL_TOKEN_CHARS}$")
_MODEL_ID = re.compile(rf"^{_MODEL_TOKEN_CHARS}(?:/{_MODEL_TOKEN_CHARS})*$")


def valid_model_identifier(value: str) -> bool:
    """Whether text is one slash-free provider/model token, not free-form prose.

    Requires at least one alphanumeric character: ``.`` and ``---`` match the
    character class but name no model, and pinning one would turn every later
    pass into a host-side model error.
    """
    return bool(_MODEL_TOKEN.fullmatch(value)) and any(
        char.isalnum() for char in value
    )


def valid_model_id(value: str) -> bool:
    """Whether text is a model id, allowing the common namespaced form."""
    return bool(_MODEL_ID.fullmatch(value)) and any(
        char.isalnum() for char in value
    )


def model_override_field_problem(value: str, *, allow_namespace: bool = False) -> str:
    """Say why a provider/model value is unusable, or "" when it is fine.

    Set ``allow_namespace`` for a model id, which may be ``vendor/name``; leave it
    off for a value that came from one side of the command's ``provider/model``
    split, where a slash cannot appear.

    Returns a reason rather than a bool so the refusal can name which rule failed.
    "Refusing an unsafe value" reads as a credential accusation for a name that
    merely had the wrong shape, and vice versa — and the caller must be able to
    explain the refusal without echoing the value, which may be a pasted secret.

    Public because the Hermes config writes the same field. If the rule lived on
    the command path only, the identical value would be refused from
    ``/refine model`` and accepted from ``config.yaml``.
    """
    accepted = valid_model_id(value) if allow_namespace else valid_model_identifier(value)
    if not accepted:
        return (
            "it is not a model identifier (letters, digits, '.', '_', ':', '-'"
            + (", '/'" if allow_namespace else "")
            + ", at least one alphanumeric)"
        )
    # The check that matters most. ``ghp_`` + 36 characters is a valid identifier,
    # so shape alone would persist a pasted token verbatim while the command echo
    # reported it redacted — telling the user a value was protected when it
    # was not. It can also fire on a legitimate name such as
    # ``my-token-model:latest``; refusing that is the safe side of the trade, and
    # the message says which rule rejected it.
    if scrub_text(value) != value:
        return "it matches a credential pattern, so it is refused rather than stored"
    return ""


def model_override_field_is_safe(value: str, *, allow_namespace: bool = False) -> bool:
    """Whether a provider/model value may be stored and sent to the host."""
    return not model_override_field_problem(value, allow_namespace=allow_namespace)


def ensure_dirs() -> Path:
    directory = journal_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _BACKUPS_DIR_NAME).mkdir(exist_ok=True)
    return directory


def journal_path() -> Path:
    return ensure_dirs() / _JOURNAL_FILE_NAME


def journal_read_path() -> Path:
    """Journal location for readers, without creating anything.

    Readers must not materialize a mistyped ``journal_dir``: a diagnostic that
    creates the directory it is meant to inspect destroys its own evidence.
    """
    return journal_dir() / _JOURNAL_FILE_NAME


def backups_dir() -> Path:
    return ensure_dirs() / _BACKUPS_DIR_NAME


def prompt_notes_path() -> Path:
    """Return the plugin-owned prompt-note store; never a host memory path."""
    return ensure_dirs() / _PROMPT_NOTES_FILE_NAME


def prompt_notes_read_path() -> Path:
    """Prompt-note store for readers, without creating anything."""
    return journal_dir() / _PROMPT_NOTES_FILE_NAME


def model_override_read_path() -> Path:
    """Model override store for readers, without creating anything."""
    return journal_dir() / _MODEL_OVERRIDE_FILE_NAME


def _read_model_override_bytes() -> Optional[bytes]:
    """Return the raw store, or None when there is genuinely no override.

    A missing file is the common case and must stay on the fast path, so it is
    reported rather than raised: only a *failed* open is worth retrying.

    ``FileNotFoundError`` is absence too, not a failure. It is caught explicitly
    because the file can vanish between the check and the open — a concurrent
    ``/refine model auto`` does exactly that — and retrying it would spend the
    whole budget and then report "could not be read" about a file the user had
    just deliberately deleted.
    """
    path = model_override_read_path()
    try:
        if not path.is_file():
            return None
        return path.read_bytes()
    except FileNotFoundError:
        return None


def read_model_override_state() -> "tuple[Optional[Dict[str, str]], str]":
    """Return the override and why it is, or is not, in force.

    States: ``absent``, ``ok``, ``rejected`` (present but unusable) and
    ``unreadable`` (present but could not be read on this attempt — on Windows a
    concurrent write can deny a read for a moment).

    The last two are kept apart from ``absent`` deliberately. Collapsing them
    would leave a file that pins one model while every diagnostic reports a
    different one, with a log line as the only trace — the invisible failure this
    project treats as worse than an error.

    Never raises: this runs while a proposal call is being assembled, and an
    exception here would surface as a generic LLM failure rather than as a
    problem with the override store.
    """
    # Bytes, not text: decoding here would raise UnicodeDecodeError, which is not
    # an OSError, so a store hand-edited in a non-UTF-8 codepage would escape this
    # function into the proposal call and end every pass as an ordinary no_op.
    # Decoding below keeps that input in the "rejected" state it belongs to.
    try:
        # Only OSError is retried — a sharing denial is the transient case, and it
        # is measured, not assumed: under a concurrent atomic replace this open is
        # denied for a moment on Windows a few times per few hundred reads. A
        # reader that gave up would report "no override" and send the call to a
        # different model than the one the user pinned.
        raw = _retry_on_contention(
            _read_model_override_bytes, _READ_RETRY_BUDGET_SECONDS, OSError
        )
    except Exception as exc:
        # Broad on purpose. "Never raises" has to hold for every exception type,
        # not just the one expected here: anything escaping would be assembled
        # into a proposal call and journaled as an ordinary no_op with
        # success=true, which is the outcome this function exists to prevent.
        logger.warning("Cannot read the model override: %s", scrub_text(str(exc)))
        return None, "unreadable"
    if raw is None:
        return None, "absent"
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, "rejected"
    if not isinstance(data, dict):
        return None, "rejected"
    model = str(data.get("model", "") or "").strip()
    provider = str(data.get("provider", "") or "").strip()
    if not model and not provider:
        return None, "rejected"
    # Validate on the way out as well as on the way in. A hand-edited, partially
    # written or externally produced store must not inject arbitrary strings into
    # the host's LLM call arguments.
    if (provider and not model_override_field_is_safe(provider)) or (
        model and not model_override_field_is_safe(model, allow_namespace=True)
    ):
        logger.warning("Ignoring an unusable model override on disk")
        return None, "rejected"
    return {"model": model, "provider": provider}, "ok"


def read_model_override() -> Optional[Dict[str, str]]:
    """Return the user's command-set model override, or None when not in force."""
    return read_model_override_state()[0]


def write_model_override(provider: str, model: str) -> None:
    """Persist the model override atomically, refusing anything unsafe.

    Mirrors ``_write_prompt_notes``: the store refuses unsafe content instead of
    trusting its caller to have checked. Raises ``ValueError`` so the command
    layer reports a refusal rather than reporting success over a rejected write.
    """
    for name, value, namespaced in (
        ("provider", provider, False),
        ("model", model, True),
    ):
        problem = (
            model_override_field_problem(value, allow_namespace=namespaced)
            if value
            else ""
        )
        if problem:
            # Names the rule but never the value: the value may be a pasted
            # credential, and this message is echoed back into the conversation.
            raise ValueError(f"Refusing to store that {name} because {problem}")
    if not provider and not model:
        raise ValueError("Refusing to store an empty model override")
    payload = json.dumps(
        {"provider": provider, "model": model, "set_ts": time.time()},
        ensure_ascii=False,
    )
    _atomic_write_text(ensure_dirs() / _MODEL_OVERRIDE_FILE_NAME, payload)


def clear_model_override() -> str:
    """Remove the model override file, reporting what actually happened.

    Returns ``removed``, ``absent`` when there was nothing to remove, or
    ``failed`` when the file survives — on Windows an unlink can fail with
    ``PermissionError`` while another session thread holds it open, so it gets the
    same bounded retry as the read and the write. Confirming "removed" in either
    of the last two cases would report an action that did not take place.

    The answer comes from the unlink itself rather than from a preceding
    existence check: with a check first, a ``/refine model x`` from another
    channel landing in between would be deleted and still reported as "no
    override was set".

    The store is deliberately not locked: the mutation lock is held for a whole
    refine pass, and making a settings command queue behind one would be worse
    than the race it removes. Last writer wins, which for a single-user setting
    command is the expected outcome.
    """
    path = model_override_read_path()
    try:
        _retry_on_contention(path.unlink, _UNLINK_RETRY_BUDGET_SECONDS)
        return "removed"
    except FileNotFoundError:
        return "absent"
    except Exception as exc:
        logger.warning("Cannot remove the model override: %s", scrub_text(str(exc)))
        try:
            return "failed" if path.exists() else "removed"
        except OSError:
            return "failed"


# ── legacy journal directory migration ─────────────────────────────────────

_MIGRATION_MARKER = ".migrated_from"
_MIGRATION_FILES = [
    _JOURNAL_FILE_NAME,
    "skill_stats.json",
    _PROMPT_NOTES_FILE_NAME,
    _MODEL_OVERRIDE_FILE_NAME,
]
_MIGRATION_DIRS = [_BACKUPS_DIR_NAME]


def migrate_legacy_journal_dir(
    *,
    _new_dir: "Optional[Path]" = None,
    _legacy_dir: "Optional[Path]" = None,
) -> str:
    """One-time, idempotent migration of runtime data to the new default location.

    Returns one of:
      user_configured — the user explicitly set journal_dir, so we do nothing.
      not_needed      — nothing to migrate (new install or already migrated).
      migrated        — files copied, old dir renamed.
      failed          — something went wrong; refine continues from wherever data
                        actually exists.

    Never deletes data. The old directory is renamed, not removed.
    Must be called under the mutation lock (taken internally) so two processes
    cannot race and double-copy.
    """
    try:
        from . import config as _cfg
    except ImportError:
        import config as _cfg  # type: ignore

    # If the user set journal_dir explicitly, their choice takes priority.
    entry = _cfg._get_refine_entry()
    if entry.get("journal_dir"):
        return "user_configured"

    new_dir = _new_dir if _new_dir is not None else _cfg.journal_dir()
    legacy = _legacy_dir if _legacy_dir is not None else _cfg.legacy_journal_dir()

    # If the legacy dir doesn't exist or has no runtime files, nothing to move.
    if not legacy.is_dir():
        return "not_needed"

    has_data = any(
        (legacy / name).exists()
        for name in _MIGRATION_FILES + _MIGRATION_DIRS
    )
    if not has_data:
        return "not_needed"

    # If the new dir already has a journal, the migration already happened or
    # someone manually moved things. Don't overwrite.
    if (new_dir / _JOURNAL_FILE_NAME).is_file():
        return "not_needed"

    # If the marker says we already renamed, don't do it again.
    if (new_dir / _MIGRATION_MARKER).is_file():
        return "not_needed"

    with _THREAD_LOCK:
        # Re-check after lock acquisition — another thread may have migrated.
        if (new_dir / _JOURNAL_FILE_NAME).is_file():
            return "not_needed"
        if (new_dir / _MIGRATION_MARKER).is_file():
            return "not_needed"

        import shutil as _shutil

        try:
            new_dir.mkdir(parents=True, exist_ok=True)

            # Copy files.
            for name in _MIGRATION_FILES:
                src = legacy / name
                if src.is_file():
                    _shutil.copy2(str(src), str(new_dir / name))

            # Copy backup directory.
            src_backups = legacy / _BACKUPS_DIR_NAME
            if src_backups.is_dir():
                dst_backups = new_dir / _BACKUPS_DIR_NAME
                dst_backups.mkdir(exist_ok=True)
                for item in src_backups.iterdir():
                    if item.is_file():
                        _shutil.copy2(str(item), str(dst_backups / item.name))

            # Write marker with the source path and timestamp.
            marker = new_dir / _MIGRATION_MARKER
            marker.write_text(
                json.dumps({
                    "from": str(legacy),
                    "ts": time.time(),
                }),
                encoding="utf-8",
            )

            # Rename (not delete) the old directory.
            ts_suffix = time.strftime("%Y%m%d-%H%M%S")
            renamed = legacy.parent / f"refine.migrated-{ts_suffix}"
            try:
                legacy.rename(renamed)
            except OSError as exc:
                # Rename failed — not fatal. The data is already in the new
                # location and the marker prevents re-migration.
                logger.warning(
                    "Legacy journal dir could not be renamed: %s",
                    scrub_text(str(exc)),
                )

            return "migrated"
        except Exception as exc:
            logger.warning(
                "Journal directory migration failed: %s", scrub_text(str(exc))
            )
            return "failed"


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
    # A reader must not create journal_dir: this runs on the pre_llm_call path
    # of every turn, and materializing a mistyped path there would erase the
    # very evidence /refine status exists to report.
    path = prompt_notes_read_path()
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


# One base owns every store-contention budget, so they cannot drift apart. The
# write waits longest because it races a *reader*, which holds the file for a
# whole read; a read and an unlink race a single atomic replace, which is orders
# of magnitude shorter. The write budget is not smaller for a measured reason: a
# 0.1s write budget lost a two-thread stress race 10 times out of 12, this one
# lost none in 12. Every budget is spent only under actual contention, and their
# sum stays well inside the timeout host callbacks use for the mutation lock.
_CONTENTION_BUDGET_SECONDS = 0.5
_WRITE_RETRY_BUDGET_SECONDS = _CONTENTION_BUDGET_SECONDS
_READ_RETRY_BUDGET_SECONDS = _CONTENTION_BUDGET_SECONDS / 5
_UNLINK_RETRY_BUDGET_SECONDS = _CONTENTION_BUDGET_SECONDS / 5
_RETRY_MAX_DELAY = 0.05


def _retry_on_contention(operation, budget: float, errors=PermissionError):
    """Run a store operation, tolerating a momentary Windows sharing denial.

    On Windows a file operation fails while another handle on the target is open,
    and every store here is read lock-free from the per-turn hook path of a
    gateway that serves several channels at once. POSIX does not fail this way, so
    this is inert there.

    This does not paper over the race: nothing about the operation's atomicity
    changes, only its scheduling, and once the budget is spent the original error
    is raised unchanged rather than swallowed.
    """
    deadline = time.monotonic() + budget
    delay = 0.005
    while True:
        try:
            return operation()
        except errors:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, _RETRY_MAX_DELAY))
            delay *= 2


def _replace_with_retry(temp_name: str, path: Path) -> None:
    """Atomically replace the target, tolerating a concurrent reader on Windows.

    This sits under ``_atomic_write_text``, so it covers every store written that
    way — prompt notes and skill snapshots as well as the model override. That is
    deliberate: all of them are read lock-free from hook paths, so all of them can
    lose the same race, and fixing it once at the writer is better than three
    times at the call sites.
    """
    _retry_on_contention(
        lambda: os.replace(temp_name, path), _WRITE_RETRY_BUDGET_SECONDS
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace backup/stat files; journals use append-only writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _load_entries() -> List[Dict[str, Any]]:
    """Stream journal state, skipping corrupt lines and collapsing updates by id."""
    path = journal_read_path()
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
    snapshot: Optional[Dict[str, Any]] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
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
    if snapshot:
        entry["snapshot"] = snapshot
    if group:
        entry["group"] = group
    # LLM attribution — additive, old entries without it read fine.
    if llm_meta and isinstance(llm_meta, dict):
        entry["llm_meta"] = llm_meta
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
    snapshot: Optional[Dict[str, Any]] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
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
        snapshot=snapshot,
        llm_meta=llm_meta,
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
    snapshot: Optional[Dict[str, Any]] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
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
        snapshot=snapshot,
        llm_meta=llm_meta,
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
        # Ask the restore path itself rather than checking for a path string, so
        # "reversible" cannot promise more than rollback can actually deliver.
        return snapshot_before_content(entry) is not None
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
        # Baselines and backups must use literal SKILL.md bytes. The host's
        # default preprocessing can render inline shell directives, producing
        # changing content and executing commands on every guard read.
        raw = skill_view(name, preprocess=False)
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


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()


def content_digest(content: str) -> str:
    """Public wrapper over the internal digest used by planning baseline capture."""
    return _content_digest(content)


def skill_baseline(name: str) -> Optional[Dict[str, Any]]:
    """Return the current skill identity for planning-baseline comparison.

    Returns:
        None — host state is unknown (read error); cannot confirm or deny.
        {"exists": False, "sha256": ""} — skill definitively does not exist.
        {"exists": True, "sha256": "<hex>"} — skill exists with this content digest.
    """
    known, content = _read_skill_state(name)
    if not known:
        return None
    if content is None:
        return {"exists": False, "sha256": ""}
    return {"exists": True, "sha256": _content_digest(content)}


def prepare_skill_recovery(name: str) -> Optional[Dict[str, Any]]:
    """Capture a skill's pre-edit state as a journal snapshot and a backup file.

    One host read serves both, so the two records cannot disagree. The snapshot
    makes rollback independent of a file surviving on disk; the ``.bak`` file
    stays because it is human-readable and because entries journaled before
    snapshots existed still restore from it.

    ``before_sha256`` is taken from the real content. The journal scrubs
    credentials out of everything it writes, so a snapshot of a skill that
    contained a secret would come back altered — the digest makes that
    detectable instead of restoring redacted text over the user's skill.
    """
    known, before = _read_skill_state(name)
    if not known or before is None:
        return None
    backup = backups_dir() / f"{int(time.time() * 1000)}_skill_{name}.bak"
    try:
        _atomic_write_text(backup, before)
    except Exception as exc:
        logger.warning("Cannot back up skill '%s': %s", name, scrub_text(str(exc)))
        return None
    return {
        "backup_path": str(backup),
        "snapshot": {
            "kind": "skill",
            "name": name,
            "before": before,
            "before_sha256": _content_digest(before),
        },
    }


def _snapshot_of(entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = entry.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def snapshot_before_content(entry: Dict[str, Any]) -> Optional[str]:
    """Return the verified pre-edit content, preferring the journal snapshot.

    Falls back to the backup file both for entries written before snapshots
    existed and for a snapshot whose digest no longer matches its content.
    """
    snapshot = _snapshot_of(entry)
    before = snapshot.get("before")
    digest = str(snapshot.get("before_sha256", ""))
    if isinstance(before, str) and digest:
        if _content_digest(before) == digest:
            return before
        logger.warning(
            "Refine journal snapshot for '%s' does not match its digest; "
            "falling back to the backup file",
            scrub_text(str(snapshot.get("name", ""))),
        )
    backup_path = Path(str(entry.get("backup_path", "")))
    if not backup_path.is_file():
        return None
    try:
        return backup_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(
            "Cannot read refine backup %s: %s", backup_path, scrub_text(str(exc))
        )
        return None


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
        expected = snapshot_before_content(entry)
        if expected is None:
            # Without a restore source there is nothing to compare against, so
            # the rollback state is unknown rather than proven false. Returning
            # False here would let ``reconcile`` declare an approved staged
            # rollback rejected and push the record back to ``applied``.
            return None
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
        restored = snapshot_before_content(entry)
        if restored is None:
            return {
                "success": False,
                "error": (
                    f"Cannot restore skill '{name}': the journal carries no verified "
                    "snapshot and its backup file is missing or unreadable"
                ),
            }
        backup_content = restored

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
