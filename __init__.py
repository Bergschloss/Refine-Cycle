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
    """Count assistant turns from host callback history without assuming its shape."""
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


def _auto_refine_allowed(assistant_turns: int, *, require_turn_interval: bool) -> bool:
    """Return whether an automatic attempt may start without mutating state."""
    if not config.auto_enabled() or not _cooldown_elapsed():
        return False
    if not require_turn_interval:
        return True
    interval = config.auto_turn_interval()
    return (
        interval > 0
        and assistant_turns >= interval
        and assistant_turns % interval == 0
    )


def _run_auto_refine(
    session_id: str, assistant_turns: int, *, require_turn_interval: bool
) -> None:
    """Run one guarded automatic pass after its worker thread has started."""
    try:
        if not _auto_refine_allowed(
            assistant_turns, require_turn_interval=require_turn_interval
        ):
            return
        with journal.try_mutation_lock() as acquired:
            if not acquired or not _cooldown_elapsed():
                return
            core.refine_run(
                llm=_REGISTERED_LLM or PluginLlm(plugin_id="refine"),
                session_id=session_id,
                auto=True,
            )
    except Exception:
        logger.exception("refine auto hook failed")
    finally:
        _finish_auto_worker()

def _start_auto_refine(
    session_id: str, assistant_turns: int, *, require_turn_interval: bool = True
) -> None:
    """Start at most one non-queued automatic attempt when its gates permit it."""
    if not _auto_refine_allowed(
        assistant_turns, require_turn_interval=require_turn_interval
    ) or not _AUTO_THREAD_GUARD.acquire(blocking=False):
        return
    try:
        threading.Thread(
            target=_run_auto_refine,
            args=(session_id, assistant_turns),
            kwargs={"require_turn_interval": require_turn_interval},
            daemon=True,
            name="refine-auto",
        ).start()
    except Exception:
        _finish_auto_worker()
        logger.exception("refine auto thread could not start")


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
    if proposal.get("action") not in (None, "no_op"):
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
    if not config.auto_enabled() or interrupted:
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
            _run_auto_refine(
                session_id,
                _assistant_turn_count(messages),
                require_turn_interval=False,
            )
        except Exception:
            logger.exception("refine auto session-end hook failed")
        finally:
            if not handed_off:
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
        description="Self-improve skills/memory. Usage: /refine [reason|audit|rollback <id>]",
        args_hint="[reason | audit | rollback <id>]",
    )
    ctx.register_tool(
        "refine_run",
        "refine",
        REFINE_RUN_SCHEMA,
        _handle_refine_run,
        description="Run one self-improvement pass over trajectory",
        emoji="🧠",
    )
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
