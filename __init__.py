"""Refine plugin registration and command handlers."""

import json
import logging
import re
import threading
from typing import Optional

from agent.plugin_llm import PluginLlm

try:
    from . import config, core
except ImportError:
    import config, core  # noqa: F811

logger = logging.getLogger(__name__)
_ROLLBACK_COMMAND = re.compile(r"^rollback\s+([0-9a-fA-F]{12})$")


def _get_llm(ctx) -> PluginLlm:
    try:
        return ctx.llm
    except Exception:
        return PluginLlm(plugin_id="refine")


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
    if not config.auto_enabled() or interrupted:
        return

    def _run() -> None:
        try:
            evidence = core.collect_evidence(session_id=session_id, limit=30)
            count = len(evidence.get("messages", []))
            if count < config.auto_min_messages():
                logger.debug("refine auto: not enough messages (%d)", count)
                return
            core.refine_run(
                llm=PluginLlm(plugin_id="refine"),
                session_id=session_id,
                auto=True,
            )
        except Exception:
            logger.exception("refine auto hook failed")

    threading.Thread(target=_run, daemon=True, name="refine-auto").start()


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
    ctx.register_hook("on_session_end", _on_session_end)
