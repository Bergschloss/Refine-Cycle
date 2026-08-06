"""refine plugin — self-improvement loop for Hermes Agent.

Registers:
  - /refine  slash command (manual trigger)
  - refine_run tool (agent-invocable)
  - on_session_end hook (auto trigger, config-gated)
"""

import logging
import threading
from typing import Optional

from agent.plugin_llm import PluginLlm

try:
    from . import config, core
except ImportError:
    import config, core  # noqa: F811 — standalone test

logger = logging.getLogger(__name__)


def _get_llm(ctx) -> PluginLlm:
    """Get a PluginLlm, preferring ctx.llm, falling back to standalone."""
    try:
        return ctx.llm
    except Exception:
        return PluginLlm(plugin_id="refine")


# ── /refine slash command handler ───────────────────────────────────────────


def _handle_refine_command(raw_args: str) -> Optional[str]:
    """Handler for /refine [reason|rollback <id>].

    Called inside a session; raw_args is everything after " /refine".
    Returns a string the user sees.
    """
    args = raw_args.strip()

    # Audit path — read-only, never deletes anything
    if args.startswith("audit"):
        try:
            result = core.refine_audit()
        except Exception as exc:
            logger.exception("refine audit failed")
            return f"❌ Audit failed: {exc}"
        return result.get("report", "No data.")

    # Rollback path
    if args.startswith("rollback"):
        entry_id = args.replace("rollback", "").strip()
        if not entry_id:
            return (
                "Usage: /refine rollback <journal_id>\n"
                "Find ids in ~/.hermes/plugins/refine/refine_journal.jsonl"
            )
        result = core.refine_rollback(entry_id)
        if result.get("success"):
            return f"✅ Rollback {entry_id}: {result.get('message', 'done')}"
        return f"❌ Rollback failed: {result.get('error', 'unknown error')}"

    # Normal refine run
    try:
        llm = PluginLlm(plugin_id="refine")
        result = core.refine_run(llm=llm, reason=args, auto=False)
    except Exception as exc:
        logger.exception("refine command failed")
        return f"❌ Refine failed: {exc}"

    if result.get("success") and result.get("message"):
        jid = result.get("journal_id", "?")
        summary = result["message"]
        proposal = result.get("proposal", {})
        if proposal.get("action") not in (None, "no_op"):
            summary += (
                f"\n📝 {proposal.get('action', '?')} {proposal.get('kind', '?')} "
                f"\"{proposal.get('name', '?')}\""
            )
        summary += f"\n🔖 {jid} (rollback: /refine rollback {jid})"
        return summary

    return f"❌ {result.get('message', 'Unknown error')}"


# ── refine_run tool handler ─────────────────────────────────────────────────


def _handle_refine_run(args: dict, **kw) -> str:
    """Tool handler for ``refine_run``."""
    import json

    reason = args.get("reason", "") if isinstance(args, dict) else ""

    try:
        llm = PluginLlm(plugin_id="refine")
        result = core.refine_run(llm=llm, reason=reason, auto=False)
    except Exception as exc:
        logger.exception("refine_run tool failed")
        return json.dumps({"success": False, "error": str(exc)})

    return json.dumps(result, ensure_ascii=False)


# ── on_session_end hook ────────────────────────────────────────────────────


def _on_session_end(
    session_id: str = "",
    turn_id: str = "",
    completed: bool = False,
    interrupted: bool = False,
    **kwargs,
) -> None:
    """Auto-trigger refine at session end, config-gated."""
    if not config.auto_enabled():
        return

    # Skip interrupted/empty sessions
    if interrupted:
        return

    # Run in background thread to not block session teardown
    def _run():
        try:
            from .core import collect_evidence

            evidence = collect_evidence(session_id=session_id, limit=30)
            msg_count = len(evidence.get("messages", []))
            if msg_count < config.auto_min_messages():
                logger.debug(
                    "refine auto: not enough messages (%d < %d)", msg_count, config.auto_min_messages()
                )
                return

            llm = PluginLlm(plugin_id="refine")
            core.refine_run(llm=llm, session_id=session_id, auto=True)
        except Exception:
            logger.exception("refine auto hook failed")

    t = threading.Thread(target=_run, daemon=True, name="refine-auto")
    t.start()


# ── plugin entry ───────────────────────────────────────────────────────────


REFINE_RUN_SCHEMA = {
    "name": "refine_run",
    "description": (
        "Trigger a self-improvement pass. Reads the agent's recent trajectory, "
        "proposes one minimal edit to a skill or memory to fix a recurring problem "
        "or capture a reusable tactic. Edits are journaled with rollback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional specific issue or area to focus on.",
            },
        },
        "required": [],
    },
}


def register(ctx) -> None:
    """Register /refine command, refine_run tool, and on_session_end hook."""

    # Slash command
    ctx.register_command(
        "refine",
        _handle_refine_command,
        description="Self-improve skills/memory from trajectory. Usage: /refine [reason|audit|rollback <id>]",
        args_hint="[reason | audit | rollback <id>]",
    )
    logger.info("refine plugin: registered /refine command")

    # Tool
    ctx.register_tool(
        "refine_run",
        "refine",
        REFINE_RUN_SCHEMA,
        _handle_refine_run,
        description="Run one self-improvement pass over trajectory → skill/memory edit",
        emoji="🧠",
    )
    logger.info("refine plugin: registered refine_run tool")

    # Hook
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("refine plugin: registered on_session_end hook")
