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


def _session_llm() -> PluginLlm:
    """Return the host-provided client, falling back to a fresh one.

    Reusing the host-provided facade keeps every path on one client instead of
    building a new one per invocation. It does not by itself decide the model:
    ``ctx.llm`` is a ``PluginLlm(plugin_id=...)`` and Hermes resolves the model
    inside its own ``call_llm``, preferring the live main model. Pin
    ``plugins.entries.refine.llm.model`` when a deterministic target is needed.
    """
    return _REGISTERED_LLM if _REGISTERED_LLM is not None else PluginLlm(
        plugin_id="refine"
    )


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
    # One owner for the arithmetic, so the gate and /refine status cannot drift.
    return core.auto_cooldown_remaining_minutes() <= 0


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
                    llm=_session_llm(),
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
        # Record the session id before anything else — this must not be lost
        # because a later error in prompt-note loading skips the rest.
        core.note_session_id(kwargs.get("session_id", ""))
        if not config.prompt_notes_enabled():
            return None
        session_id = journal.normalize_prompt_note_session_id(kwargs.get("session_id", ""))
        # With no store there is nothing to inject, and taking the lock would
        # create journal_dir on a path that may simply be mistyped. This hook
        # runs every turn, so it must stay strictly read-only.
        if not journal.prompt_notes_read_path().is_file():
            return None
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
    core.note_session_id(session_id)
    _start_auto_refine(session_id, _assistant_turn_count(conversation_history))


_MODEL_SUBCOMMAND = "model"


def _is_model_target(remainder: str) -> bool:
    """Whether text is exactly ``<model>`` or ``<provider>/<model>``.

    Only the first slash separates provider from model; any further ones belong to
    the model id, which is commonly namespaced (``openrouter/deepseek/deepseek-chat``).
    Rejecting those made the command start a real refine pass instead — spending a
    daily edit on a mistyped setting.

    Anything else — prose, an empty half such as ``a/`` or ``/b`` — is a free-form
    reason and must reach the proposal path untouched. The rule itself lives in
    ``journal``, which owns the store.
    """
    head, separator, tail = remainder.partition("/")
    if not journal.valid_model_identifier(head):
        return False
    if not separator:
        return True
    return bool(tail) and journal.valid_model_id(tail)


def _handle_model_subcommand(remainder: str) -> str:
    """Handle /refine model [auto | <provider/model> | <model>]."""
    trust_model = config.llm_allow_model_override()
    trust_prov = config.llm_allow_provider_override()

    if not remainder:
        # Show current effective target
        effective = config.effective_llm_target()
        source = effective["source"]
        set_model = effective.get("model", "")
        set_provider = effective.get("provider", "")
        lines = [
            f"model: {set_model or '(host default)'}",
            f"provider: {set_provider or '(host default)'}",
            f"source: {source}",
            f"trust: model={'allowed' if trust_model else 'denied'}, "
            f"provider={'allowed' if trust_prov else 'denied'}",
        ]
        for issue in effective.get("issues", ()):
            lines.append(f"⚠ {issue}")
        # Only warn where trust actually changes the outcome. On ``live`` the
        # user set nothing and the host uses that model regardless, so warning
        # there would report a problem that does not exist.
        if source in ("command", "config"):
            if set_model and not trust_model:
                lines.append(
                    "⚠ Model is set but host trust denies overrides. "
                    "Enable plugins.entries.refine.llm.allow_model_override to apply it."
                )
            if set_provider and not trust_prov:
                lines.append(
                    "⚠ Provider is set but host trust denies overrides. Enable "
                    "plugins.entries.refine.llm.allow_provider_override to apply it."
                )
        return core.scrub_text("\n".join(lines))

    if remainder == "auto":
        outcome = journal.clear_model_override()
        effective = config.effective_llm_target()
        prefix = {
            "removed": "Override removed.",
            "absent": "No override was set.",
            # Does not claim the override is "still in force": the file surviving
            # and the file being usable are different things, and the effective
            # target printed next is the accurate answer either way.
            "failed": "⚠ Could not remove the override file.",
        }[outcome]
        return core.scrub_text(
            f"{prefix} Effective model: {effective.get('model') or '(host default)'} "
            f"(source: {effective['source']})"
        )

    # Parse provider/model or bare model. The store validates and refuses; doing
    # it again here would put the same rule in two places that could drift.
    provider = ""
    model = remainder
    if "/" in remainder:
        provider, model = remainder.split("/", 1)

    journal.write_model_override(provider, model)
    lines = [f"Override set: model={model}" + (f" provider={provider}" if provider else "")]
    if not trust_model:
        lines.append(
            "⚠ Host trust denies model overrides. The value is saved but will not "
            "be sent until plugins.entries.refine.llm.allow_model_override is true."
        )
    if provider and not trust_prov:
        lines.append(
            "⚠ Host trust denies provider overrides. The provider value is saved but "
            "will not be sent until plugins.entries.refine.llm.allow_provider_override is true."
        )
    return core.scrub_text("\n".join(lines))


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
            status = core.refine_status()
        except Exception as exc:
            logger.exception("refine status failed")
            return f"❌ Status failed: {core.scrub_text(str(exc))}"
        lines = [
            f"auto: {'on' if status['auto_enabled'] else 'off'}",
            f"turn interval: {status['auto_turn_interval']}"
            + ("" if status["turn_trigger_enabled"] else " (turn trigger off)"),
            f"min messages: {status['auto_min_messages']}",
            f"cooldown: {status['auto_cooldown_minutes']} min",
            f"edits today: {status['edits_today']}/{status['max_edits_per_day']}",
            f"session: {status['session_id'] or '(unknown)'}"
            + f" (source: {status['session_id_source']}"
            + f", messages: {status['session_message_count']})",
            f"session db source: {status['session_source'] or '(empty/unknown)'}",
            "skipped session sources: "
            + (", ".join(status["skip_session_sources"]) or "(none)"),
            f"model: {status['llm_model'] or '(host default)'}"
            + (f" @ {status['llm_provider']}" if status["llm_provider"] else "")
            + f" (source: {status['llm_target_source']})",
            f"journal: {status['journal_dir']} ({status['journal_dir_state_text']})",
            f"migration: {status['migration_outcome']}"
            + (
                f" (active: {status['migration'].get('active_dir')})"
                if status["migration"].get("active_dir")
                else ""
            ),
        ]
        if status["cooldown_remaining_minutes"] > 0:
            lines.append(
                f"cooldown remaining: {status['cooldown_remaining_minutes']} min"
            )
        if status["blockers"]:
            lines.append("blockers:")
            lines.extend(f"  • {item['message']}" for item in status["blockers"])
        else:
            lines.append("blockers: none — automatic refinement is active")
        if status["warnings"]:
            lines.append("warnings:")
            lines.extend(f"  ⚠ {item['message']}" for item in status["warnings"])
        return core.scrub_text("\n".join(lines))

    if args == "dry-run" or args.startswith("dry-run "):
        dry_reason = args[7:].strip()  # len("dry-run") == 7
        try:
            result = core.refine_run(
                llm=_session_llm(), reason=dry_reason, auto=False, dry_run=True
            )
        except Exception as exc:
            logger.exception("refine dry-run failed")
            return f"❌ Dry-run failed: {core.scrub_text(str(exc))}"
        if result.get("outcome") == "dry_run":
            proposal = result.get("proposal", {})
            lines = ["🔍 Dry run — nothing applied."]
            evidence = result.get("evidence", {})
            if evidence.get("session_id"):
                lines.append(
                    f"session: {evidence['session_id']} "
                    f"(source: {evidence.get('session_id_source', 'unknown')})"
                )
            if proposal.get("action") and proposal["action"] != "no_op":
                lines.append(
                    f"action: {proposal.get('action')} | kind: {proposal.get('kind', '')} "
                    f"| name: {proposal.get('name', '')}"
                )
                if proposal.get("summary"):
                    lines.append(f"summary: {proposal['summary']}")
                if proposal.get("expected_outcome"):
                    lines.append(f"expected: {proposal['expected_outcome']}")
            else:
                lines.append(f"action: no_op | reason: {proposal.get('reason', '')}")
            diff = result.get("diff", "")
            if diff:
                lines.append("\n```diff")
                lines.append(diff)
                lines.append("```")
                if result.get("diff_truncated"):
                    lines.append("(diff truncated)")
            return core.scrub_text("\n".join(lines))
        # Non-dry-run outcome (session_unknown, skipped, etc.)
        if not result.get("success"):
            return f"❌ {result.get('message', 'Unknown error')}"
        return result.get("message", "No proposal.")

    if args == _MODEL_SUBCOMMAND or args.startswith(_MODEL_SUBCOMMAND + " "):
        remainder = args[len(_MODEL_SUBCOMMAND):].strip()
        # Only treat as a subcommand when:
        # - no remainder (show current)
        # - remainder is "auto"
        # - remainder is exactly a model or provider/model token
        # Anything else ("model of gmail failures") is a free-form reason.
        if not remainder or remainder == "auto" or _is_model_target(remainder):
            try:
                return _handle_model_subcommand(remainder)
            except Exception as exc:
                # The other subcommands report failures; this one used to escape
                # as a traceback on exactly the unwritable journal_dir that
                # /refine status exists to diagnose.
                logger.exception("refine model command failed")
                return f"❌ Model command failed: {core.scrub_text(str(exc))}"
        # Fall through to the proposal path below

    if args == "rollback":
        return (
            "Usage: /refine rollback <12-character journal_id>\n"
            f"Find ids in {journal.journal_read_path()}"
        )
    rollback_match = _ROLLBACK_COMMAND.fullmatch(args)
    if rollback_match:
        entry_id = rollback_match.group(1).lower()
        result = core.refine_rollback(entry_id)
        if result.get("success"):
            return f"✅ Rollback {entry_id}: {result.get('message', 'done')}"
        return f"❌ Rollback failed: {result.get('error', 'unknown error')}"

    try:
        result = core.refine_run(llm=_session_llm(), reason=args, auto=False)
    except Exception as exc:
        logger.exception("refine command failed")
        return f"❌ Refine failed: {exc}"

    if not result.get("success") and result.get("outcome") != "partial_success":
        return f"❌ {result.get('message', 'Unknown error')}"

    summary = result.get("message", "done")
    evidence = result.get("evidence", {})
    if evidence.get("session_id"):
        summary += (
            f"\nsession: {evidence['session_id']} "
            f"(source: {evidence.get('session_id_source', 'unknown')})"
        )
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
        result = core.refine_run(llm=_session_llm(), reason=reason, auto=False)
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
    core.note_session_id(session_id)
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
            session_source, _ = core._get_session_source_status(session_id)
            if (
                session_source
                and session_source.lower() in config.skip_session_sources()
            ):
                # Let the normal source gate create the non-budget journal record,
                # but do not fetch even one trajectory row in this preflight.
                handed_off = True
                _run_auto_refine(session_id, cleanup_session_notes=True)
                return
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
    # One-time migration of runtime data out of the plugin install directory.
    # Must not fail registration — a broken migration just leaves data in place.
    try:
        journal.migrate_legacy_journal_dir()
    except Exception:
        logger.debug("refine journal migration failed", exc_info=True)
    ctx.register_command(
        "refine",
        _handle_refine_command,
        description=(
            "Self-improve skills/memory. "
            "Usage: /refine [reason|audit|status|dry-run|model [target|auto]|rollback <id>]"
        ),
        args_hint="[reason | audit | status | dry-run | model [target|auto] | rollback <id>]",
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
    """Warn once per process that runtime data sits in the plugin directory."""
    global _REGISTER_WARNED
    if _REGISTER_WARNED:
        return
    try:
        jdir = config.journal_dir()
        if not (jdir / "plugin.yaml").is_file():
            return
    except Exception:
        return
    # Set only after the condition held, so a warning is never skipped because
    # an earlier call returned before there was anything to report.
    _REGISTER_WARNED = True
    logger.warning(
        "Refine journal_dir (%s) contains plugin source code. "
        "A forced reinstall may delete runtime data (journal, backups, ledger). "
        "Set plugins.entries.refine.journal_dir to a separate path.",
        jdir,
    )
