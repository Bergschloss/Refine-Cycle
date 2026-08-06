"""Refine plugin registration and command handlers."""

import json
import logging
import re
import threading
import time
from typing import Any, Optional

from agent.plugin_llm import PluginLlm

try:
    from . import config, core, journal
except ImportError:
    import config, core, journal  # noqa: F811

logger = logging.getLogger(__name__)
_ROLLBACK_COMMAND = re.compile(r"^rollback\s+([0-9a-fA-F]{12})$")


_AUTO_THREAD_GUARD = threading.Lock()
_AUTO_PENDING_SESSION_ENDS: set[str] = set()
_AUTO_PENDING_LOCK = threading.Lock()
_REGISTERED_LLM: Optional[PluginLlm] = None

# Assistant-message count observed when each session last started an attempt.
# One host turn can append several assistant messages, so the trigger compares a
# delta instead of an exact multiple; an exact multiple is silently skipped
# whenever a tool-using turn steps over it.
_AUTO_TURN_MARKS: dict[str, int] = {}
_AUTO_TURN_MARKS_LOCK = threading.Lock()
_AUTO_TURN_MARKS_MAX = 64
_HOST_PATH_LOCK_TIMEOUT = 2.0


def _defer_session_end(session_id: str) -> None:
    """Coalesce session-end fallbacks without creating blocked worker threads."""
    with _AUTO_PENDING_LOCK:
        _AUTO_PENDING_SESSION_ENDS.add(session_id)


def _finish_auto_worker() -> None:
    """Release one worker slot, then start one coalesced session-end fallback."""
    _AUTO_THREAD_GUARD.release()
    with _AUTO_PENDING_LOCK:
        session_id = next(iter(_AUTO_PENDING_SESSION_ENDS), None)
        if session_id is not None:
            _AUTO_PENDING_SESSION_ENDS.remove(session_id)
    if session_id is not None:
        _on_session_end(session_id=session_id)


def _get_llm(ctx) -> PluginLlm:
    try:
        llm = ctx.llm
        if llm is not None:
            return llm
    except Exception:
        pass
    return PluginLlm(plugin_id="refine")


def _assistant_turn_count(conversation_history: Any) -> int:
    """Count assistant messages in host callback history without assuming its shape.

    One host turn can contribute several assistant messages, so this is a
    monotonic progress measure, not a count of user-visible turns.
    """
    if not isinstance(conversation_history, (list, tuple)):
        return 0
    count = 0
    for message in conversation_history:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "assistant":
            count += 1
    return count


def _cooldown_elapsed() -> bool:
    last_attempt = journal.last_attempt_ts()
    if last_attempt is None:
        return True
    return time.time() - last_attempt >= config.auto_cooldown_minutes() * 60


def _turn_interval_reached(session_id: str, assistant_turns: int) -> bool:
    """Compare assistant messages added since this session's last attempt."""
    interval = config.auto_turn_interval()
    if interval <= 0:
        return False
    with _AUTO_TURN_MARKS_LOCK:
        return assistant_turns - _AUTO_TURN_MARKS.get(session_id, 0) >= interval


def _mark_turn_attempt(session_id: str, assistant_turns: int) -> None:
    """Record the attempt point, keeping the per-session marks bounded."""
    with _AUTO_TURN_MARKS_LOCK:
        if (
            session_id not in _AUTO_TURN_MARKS
            and len(_AUTO_TURN_MARKS) >= _AUTO_TURN_MARKS_MAX
        ):
            _AUTO_TURN_MARKS.pop(next(iter(_AUTO_TURN_MARKS)), None)
        _AUTO_TURN_MARKS[session_id] = assistant_turns


def _forget_turn_marks(session_id: str) -> None:
    with _AUTO_TURN_MARKS_LOCK:
        _AUTO_TURN_MARKS.pop(session_id, None)


def _auto_refine_allowed() -> bool:
    """Return whether an automatic attempt may start without mutating state."""
    return config.auto_enabled() and _cooldown_elapsed()


def _run_auto_refine(session_id: str, *, cleanup_session_notes: bool = False) -> None:
    """Run one guarded automatic pass after its worker thread has started."""
    try:
        if not _auto_refine_allowed():
            if cleanup_session_notes:
                _clear_session_prompt_notes(session_id)
            return
        with journal.try_mutation_lock() as acquired:
            if acquired and _cooldown_elapsed():
                core.refine_run(
                    llm=_REGISTERED_LLM or PluginLlm(plugin_id="refine"),
                    session_id=session_id,
                    auto=True,
                )
            if cleanup_session_notes:
                _clear_session_prompt_notes(session_id)
    except Exception:
        logger.exception("refine auto hook failed")
    finally:
        _finish_auto_worker()


def _start_auto_refine(session_id: str, assistant_turns: int) -> None:
    """Start at most one non-queued automatic attempt when its gates permit it."""
    if (
        not _auto_refine_allowed()
        or not _turn_interval_reached(session_id, assistant_turns)
        or not _AUTO_THREAD_GUARD.acquire(blocking=False)
    ):
        return
    # Charge the attempt to this turn point before the worker starts, so a
    # skipped or failed attempt cannot retry on every following turn.
    _mark_turn_attempt(session_id, assistant_turns)
    try:
        threading.Thread(
            target=_run_auto_refine,
            args=(session_id,),
            daemon=True,
            name="refine-auto",
        ).start()
    except Exception:
        _finish_auto_worker()
        logger.exception("refine auto thread could not start")


def _on_pre_llm_call(**kwargs) -> Optional[dict]:
    """Inject bounded plugin-owned notes without reading or changing the base prompt."""
    try:
        if not config.prompt_notes_enabled():
            return None
        session_id = journal.normalize_prompt_note_session_id(kwargs.get("session_id", ""))
        # Prefer reading under the lock, but never drop notes for a whole turn
        # just because a refine pass owns it: the store is only ever replaced
        # atomically, so a lock-free read still sees one complete generation.
        with journal.try_mutation_lock():
            notes = journal.load_prompt_notes()
        if not notes:
            return None
        selected = []
        for note in notes:
            scope = note.get("scope", "global")
            if scope == "session" and note.get("session_id") != session_id:
                continue
            content = core.scrub_text(note["content"]).strip()
            if not content or core._prompt_note_content_error(content, check_rendered_size=False):
                continue
            selected.append({"id": note["id"], "content": content})
        if not selected:
            return None
        selected = selected[-config.prompt_notes_max_count():]
        while selected:
            rendered = "Refine notes:\n" + "\n".join(
                f"- {note['content']}" for note in selected
            )
            safe_rendered = core.scrub_text(rendered)
            if len(safe_rendered) <= config.prompt_notes_max_chars():
                return {"context": safe_rendered}
            selected = selected[1:]
    except Exception:
        logger.debug("refine prompt-note hook failed", exc_info=True)
    return None


def _clear_session_prompt_notes(
    session_id: str, *, timeout: Optional[float] = None
) -> None:
    """Clear plugin-owned session notes without stalling a host callback."""
    try:
        if timeout is None:
            journal.clear_session_prompt_notes(session_id)
        else:
            journal.clear_session_prompt_notes(session_id, timeout=timeout)
    except Exception:
        logger.debug("refine session prompt-note cleanup failed", exc_info=True)


def _on_session_reset(session_id: str = "", **kwargs) -> None:
    """Expire only notes owned by the session Hermes reset."""
    _forget_turn_marks(session_id)
    _clear_session_prompt_notes(session_id, timeout=_HOST_PATH_LOCK_TIMEOUT)


def _on_post_llm_call(
    session_id: str = "", conversation_history: Any = None, **kwargs
) -> None:
    _start_auto_refine(session_id, _assistant_turn_count(conversation_history))


def _handle_refine_command(raw_args: str) -> Optional[str]:
    """Handle exact audit/rollback subcommands; all other text is a reason."""
    args = raw_args.strip()
    if args == "audit":
        try:
            return core.refine_audit().get("report", "No data.")
        except Exception as exc:
            logger.exception("refine audit failed")
            return f"❌ Audit failed: {exc}"

    if args == "status":
        try:
            from .sanitization import scrub_text as _scrub
        except ImportError:
            from sanitization import scrub_text as _scrub
        try:
            status = core.refine_status()
        except Exception as exc:
            logger.exception("refine status failed")
            return f"❌ Status failed: {exc}"
        lines = [
            f"auto: {'on' if status['auto_enabled'] else 'off'}",
            f"turn interval: {status['auto_turn_interval']}",
            f"min messages: {status['auto_min_messages']}",
            f"cooldown: {status['auto_cooldown_minutes']} min",
            f"edits today: {status['edits_today']}/{status['max_edits_per_day']}",
            f"journal: {status['journal_dir']}",
        ]
        if status.get("cooldown_remaining_minutes"):
            lines.append(f"cooldown remaining: {status['cooldown_remaining_minutes']} min")
        if status["blockers"]:
            lines.append("blockers:")
            for b in status["blockers"]:
                lines.append(f"  • {b}")
        else:
            lines.append("blockers: none — auto-refine is active")
        if status["warnings"]:
            lines.append("warnings:")
            for w in status["warnings"]:
                lines.append(f"  ⚠ {w}")
        return _scrub("\n".join(lines))

    if args == "rollback":
        return (
            "Usage: /refine rollback <12-character journal_id>\n"
            "Find ids in <HERMES_HOME>/plugins/refine/refine_journal.jsonl"
        )
    rollback_match = _ROLLBACK_COMMAND.fullmatch(args)
    if rollback_match:
        entry_id = rollback_match.group(1).lower()
        result = core.refine_rollback(entry_id)
        if result.get("success"):
            return f"✅ Rollback {entry_id}: {result.get('message', 'done')}"
        return f"❌ Rollback failed: {result.get('error', 'unknown error')}"

    try:
        result = core.refine_run(
            llm=PluginLlm(plugin_id="refine"), reason=args, auto=False
        )
    except Exception as exc:
        logger.exception("refine command failed")
        return f"❌ Refine failed: {exc}"

    if not result.get("success") and result.get("outcome") != "partial_success":
        return f"❌ {result.get('message', 'Unknown error')}"

    summary = result.get("message", "done")
    if result.get("outcome") == "partial_success":
        summary = "⚠️ " + summary
    proposal = result.get("proposal", {})
    edits = proposal.get("edits")
    if isinstance(edits, list) and edits:
        if proposal.get("summary"):
            summary += f"\n📝 {proposal['summary']}"
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            summary += (
                f"\n   • {edit.get('action', '?')} {edit.get('kind', '?')} "
                f"\"{edit.get('name', '?')}\""
            )
    elif proposal.get("action") not in (None, "no_op"):
        summary += (
            f"\n📝 {proposal.get('action', '?')} {proposal.get('kind', '?')} "
            f"\"{proposal.get('name', '?')}\""
        )
    recoveries = result.get("recoveries", [])
    if recoveries:
        summary += "\nRecovery / rollback IDs:"
        for recovery in recoveries:
            journal_id = recovery.get("journal_id", "?")
            line = f"\n🔖 {journal_id} ({recovery.get('outcome', 'unknown')})"
            if recovery.get("rollback_command"):
                line += f" — {recovery['rollback_command']}"
            summary += line
    else:
        journal_id = result.get("journal_id")
        if journal_id:
            summary += f"\n🔖 {journal_id}"
            if result.get("reversible"):
                summary += f" (rollback: /refine rollback {journal_id})"
    return summary


def _handle_refine_run(args: dict, **kw) -> str:
    reason = args.get("reason", "") if isinstance(args, dict) else ""
    try:
        result = core.refine_run(
            llm=PluginLlm(plugin_id="refine"), reason=reason, auto=False
        )
    except Exception as exc:
        logger.exception("refine_run tool failed")
        return json.dumps({"success": False, "error": str(exc)})
    return json.dumps(result, ensure_ascii=False)


def _on_session_end(
    session_id: str = "",
    turn_id: str = "",
    completed: bool = False,
    interrupted: bool = False,
    **kwargs,
) -> None:
    """Run the session-end fallback without blocking or dropping it behind a turn run."""
    _forget_turn_marks(session_id)
    if not config.auto_enabled() or interrupted:
        _clear_session_prompt_notes(session_id, timeout=_HOST_PATH_LOCK_TIMEOUT)
        return
    if not _AUTO_THREAD_GUARD.acquire(blocking=False):
        _defer_session_end(session_id)
        return

    def _collect_and_run() -> None:
        handed_off = False
        try:
            evidence = core.collect_evidence(session_id=session_id, limit=30)
            messages = evidence.get("messages", [])
            if len(messages) < config.auto_min_messages():
                logger.debug("refine auto: not enough messages (%d)", len(messages))
                return
            handed_off = True
            _run_auto_refine(session_id, cleanup_session_notes=True)
        except Exception:
            logger.exception("refine auto session-end hook failed")
        finally:
            if not handed_off:
                _clear_session_prompt_notes(session_id)
                _finish_auto_worker()

    try:
        threading.Thread(
            target=_collect_and_run, daemon=True, name="refine-auto"
        ).start()
    except Exception:
        _finish_auto_worker()
        logger.exception("refine auto session-end thread could not start")


REFINE_RUN_SCHEMA = {
    "name": "refine_run",
    "description": (
        "Trigger a self-improvement pass over recent repeated failures or explicit corrections. "
        "Mutations are serialized, journaled, and reversible when applied."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional issue or area to focus on; passed to the proposal model.",
            }
        },
        "required": [],
    },
}


def register(ctx) -> None:
    global _REGISTERED_LLM
    _REGISTERED_LLM = _get_llm(ctx)
    ctx.register_command(
        "refine",
        _handle_refine_command,
        description="Self-improve skills/memory. Usage: /refine [reason|audit|status|rollback <id>]",
        args_hint="[reason | audit | status | rollback <id>]",
    )
    ctx.register_tool(
        "refine_run",
        "refine",
        REFINE_RUN_SCHEMA,
        _handle_refine_run,
        description="Run one self-improvement pass over trajectory",
        emoji="🧠",
    )
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_reset", _on_session_reset)
    _warn_on_register()


_REGISTER_WARNED = False


def _warn_on_register() -> None:
    """Log once-per-process warnings about configuration issues."""
    global _REGISTER_WARNED
    if _REGISTER_WARNED:
        return
    _REGISTER_WARNED = True
    jdir = config.journal_dir()
    try:
        if (jdir / "plugin.yaml").is_file():
            logger.warning(
                "Refine journal_dir (%s) contains plugin source code. "
                "A forced reinstall may delete runtime data (journal, backups, ledger). "
                "Set plugins.entries.refine.journal_dir to a separate path.",
                jdir,
            )
    except Exception:
        pass
