"""LLM proposal engine for the refine plugin.

Wraps ``PluginLlm`` to call the user's active model with structured output
and produce a validated refine proposal (or no_op).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from agent.plugin_llm import (
    PluginLlm,
    PluginLlmInput,
    PluginLlmStructuredResult,
    PluginLlmTextInput,
    PluginLlmTrustError,
)

logger = logging.getLogger(__name__)

# ── schema ──────────────────────────────────────────────────────────────────

REFINE_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "patch", "no_op"],
            "description": "The smallest useful action: create a new skill/memory, patch an existing one (tiny targeted fix), or no_op when nothing worthwhile found."
        },
        "kind": {
            "type": "string",
            "enum": ["skill", "memory"],
            "description": "What to edit — a Hermes skill or a memory entry."
        },
        "name": {
            "type": "string",
            "description": "Target name. For create: new skill/memory name (lowercase, hyphens, max 64 chars for skills). For patch: existing name."
        },
        "content": {
            "type": "string",
            "description": "For create: full SKILL.md content (YAML frontmatter + body). For patch: the new replacement text that fixes the issue. Empty for no_op."
        },
        "category": {
            "type": "string",
            "description": "Optional category folder for skill creation (e.g. 'devops', 'research')."
        },
        "reason": {
            "type": "string",
            "description": "Why this edit, citing the specific failure/pattern from the trajectory."
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short verbatim quotes from the trajectory that justify this edit (2-4 items)."
        },
        "pattern_fingerprint": {
            "type": "string",
            "description": "The fp value of the listed repeated failure this edit addresses, if any. Enables later checking whether the edit actually stopped that failure."
        },
    },
    "required": ["action", "reason"],
}

REFINE_SYSTEM_PROMPT: str = (
    "You are a self-improvement mechanic for an AI agent named Hermes. "
    "Your job: read the agent's recent conversation trajectory and find "
    "ONE specific, recurring problem or ONE useful reusable tactic. "
    "Propose the SMALLEST possible edit — a create or patch of ONE skill "
    "or ONE memory entry — that would prevent the problem or capture the tactic.\n\n"
    "RULES:\n"
    "1. Only ONE edit per proposal. If you see multiple issues, pick the most impactful.\n"
    "2. Only act when there is real evidence in the trajectory — do not guess.\n"
    "3. Do not duplicate existing skills (existing skill names are listed above).\n"
    "4. NEVER edit built-in or bundled skills. Only user/agent-created ones.\n"
    "5. If patch: provide the NEW full content (the fix), not a diff.\n"
    "6. Skills MUST have YAML frontmatter (name, description) + markdown body.\n"
    "7. If nothing worthwhile — return action='no_op' with a brief reason.\n"
    "8. Be MINIMAL. The smallest edit that fixes the problem. No scope creep.\n"
    "9. ALWAYS use exactly these fields: action (create|patch|no_op), kind (skill|memory), "
    "name, content, category (optional), reason, evidence. "
    "NEVER invent new field names (e.g. 'type') and never combine action+kind "
    "into one value like 'create_memory'.\n"
    "Example: {\"action\": \"create\", \"kind\": \"skill\", \"name\": \"my-tip\", "
    "\"content\": \"---\\nname: my-tip\\ndescription: ...\\n---\\n\\n# Body\", "
    "\"reason\": \"why\", \"evidence\": [\"quote\"]}\n"
)

# ── engine ──────────────────────────────────────────────────────────────────


def _salvage_parsed(result: Any) -> Any:
    """Return ``result.parsed``, or salvage a JSON object from raw text when
    the provider's json_mode parsing failed (parsed is None/str)."""
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    text = getattr(result, "text", "") or ""
    if isinstance(parsed, str):
        text = parsed if not text else text
    if text:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return parsed


def _propose_structured(llm: PluginLlm, instructions: str, input_blocks: List[PluginLlmInput]) -> Any:
    """Call complete_structured, preferring json_schema and falling back to
    json_mode for providers that reject ``response_format.type=json_schema``
    (e.g. opencode-go "This response_format type is unavailable now").

    Returns ``result.parsed`` (dict) or raises on final failure.
    """
    common = dict(
        instructions=instructions,
        input=input_blocks,
        schema_name="refine_proposal",
        purpose="refine",
        temperature=0.0,
        max_tokens=2048,
    )
    try:
        result = llm.complete_structured(
            system_prompt=REFINE_SYSTEM_PROMPT, json_schema=REFINE_PROPOSAL_SCHEMA, **common
        )
        return _salvage_parsed(result)
    except PluginLlmTrustError:
        raise
    except Exception as first_exc:
        logger.warning(
            "complete_structured(json_schema) failed (%s) — falling back to json_mode", first_exc
        )
        try:
            result = llm.complete_structured(
                system_prompt=REFINE_SYSTEM_PROMPT
                + "\n\nReply with a single JSON object ONLY. No markdown fences, no commentary.",
                json_mode=True,
                **common,
            )
            return _salvage_parsed(result)
        except Exception as exc2:
            logger.error("json_mode fallback also failed: %s", exc2)
            raise


def propose(
    llm: PluginLlm,
    evidence_text: str,
    existing_skills: List[str],
    existing_memories: List[str],
    *,
    error_patterns: Optional[List[Dict[str, Any]]] = None,
    user_corrections: Optional[List[str]] = None,
    unused_skills: Optional[List[str]] = None,
    purpose: str = "refine",
) -> Dict[str, Any]:
    """Call the LLM to propose a refine edit.

    Args:
        llm: A ``PluginLlm`` instance (can be ``ctx.llm`` or a fresh one).
        evidence_text: Recent trajectory content (truncated by caller).
        existing_skills: List of current skill names (to avoid duplicates).
        existing_memories: List of current memory entry short summaries.
        error_patterns: Aggregated repeated failures from ``patterns.extract_patterns``.
        user_corrections: Short quotes where the user corrected the agent.
        unused_skills: Refine-created skills that were never used — negative examples.
        purpose: Audit purpose string.

    Returns:
        A dict with shape ``{"action": str, "kind": str, "name": str,
        "content": str, "category": str, "reason": str, "evidence": list}``,
        or ``{"action": "no_op", "reason": "..."}`` on failure.
    """
    # Build instructions with context. The aggregated signal goes FIRST: a model
    # reasons far better over pre-counted patterns than over a transcript it has
    # to count through itself.
    try:
        from . import patterns as _patterns
    except ImportError:
        import patterns as _patterns  # noqa: F811 — standalone test

    skills_list = "\n".join(f"  - {s}" for s in sorted(existing_skills)[:50]) or "  (none)"
    mems_list = "\n".join(f"  - {m[:100]}" for m in existing_memories[:30]) or "  (none)"
    patterns_block = _patterns.format_patterns(error_patterns or [])
    corrections_block = (
        "\n".join(f"  - {c[:200]}" for c in (user_corrections or [])[:5]) or "  (none)"
    )

    unused_block = ""
    if unused_skills:
        unused_block = (
            "\n=== YOUR PREVIOUS SKILLS THAT WERE NEVER USED ===\n"
            + "\n".join(f"  - {s}" for s in unused_skills[:10])
            + "\nDo not create skills of this shape again. A skill must change what "
            "the agent DOES in a specific situation, not merely record a fact.\n"
        )

    instructions = (
        "Below is aggregated evidence from the agent's recent sessions. "
        "Ground your proposal in ONE of the listed repeated failures or user "
        "corrections and propose the smallest edit that would prevent it.\n\n"
        "=== REPEATED FAILURES (aggregated, strongest first) ===\n"
        f"{patterns_block}\n\n"
        "=== USER CORRECTIONS ===\n"
        f"{corrections_block}\n\n"
        "=== EXISTING SKILLS (do NOT duplicate) ===\n"
        f"{skills_list}\n\n"
        "=== EXISTING MEMORIES ===\n"
        f"{mems_list}\n"
        f"{unused_block}\n"
        "=== RECENT TRAJECTORY (context only — the signal is above) ===\n"
        f"{evidence_text[-8000:]}\n\n"
        "Return a single JSON object with your proposal. When your proposal "
        "addresses a listed failure, set pattern_fingerprint to its fp value."
    )

    # Short imperative instruction goes to ``instructions=``; the full context
    # is passed as the input block. Passing the same text in both places would
    # double the tokens on every call.
    short_instructions = "Propose one minimal skill or memory edit."
    input_blocks: List[PluginLlmInput] = [
        PluginLlmTextInput(text=instructions),
    ]

    try:
        parsed = _propose_structured(llm, short_instructions, input_blocks)

        if not isinstance(parsed, dict):
            # Provider returned non-object (e.g. raw text) — try to salvage JSON.
            if isinstance(parsed, str):
                import re as _re
                m = _re.search(r"\{.*\}", parsed, _re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
        if not isinstance(parsed, dict):
            logger.warning("LLM returned non-dict parsed: %s", type(parsed))
            return {"action": "no_op", "reason": "LLM returned non-object output"}

        # Retry once if the model proposed create without content (a common
        # json_mode miss) — nudge it to include the content field.
        if (
            str(parsed.get("action", "")).strip() == "create"
            and not str(parsed.get("content") or "").strip()
        ):
            retry_instructions = (
                instructions
                + "\n\nIMPORTANT: action='create' REQUIRES a non-empty 'content' "
                + "field containing the full SKILL.md or memory text. Include it."
            )
            retry = _propose_structured(
                llm, short_instructions, [PluginLlmTextInput(text=retry_instructions)]
            )
            if isinstance(retry, dict) and str(retry.get("content") or "").strip():
                parsed = retry

        # Validate + normalize
        action = str(parsed.get("action", "no_op")).strip().lower()

        # Normalize "<verb>_<kind>" style (create_memory, patch_skill, ...)
        # that some models emit in json_mode instead of separate fields.
        for _verb in ("create", "patch"):
            for _kind in ("skill", "memory"):
                if action == f"{_verb}_{_kind}":
                    parsed.setdefault("kind", _kind)
                    action = _verb

        if action not in ("create", "patch", "no_op"):
            return {"action": "no_op", "reason": f"Invalid action: {action}"}

        if action == "no_op":
            return {
                "action": "no_op",
                "kind": "",
                "name": "",
                "content": "",
                "category": "",
                "reason": str(parsed.get("reason", "No actionable improvement found")),
                "evidence": _ensure_list(parsed.get("evidence")),
            }

        kind = str(parsed.get("kind") or parsed.get("type") or "").strip()
        name = str(parsed.get("name", "")).strip()
        content = str(parsed.get("content", "")).strip()
        category = str(parsed.get("category", "")).strip()

        if kind not in ("skill", "memory"):
            # Heuristic for models that omit kind: YAML-frontmatter content
            # looks like a SKILL.md, anything else is a memory entry.
            if action == "create":
                if re.search(r"^---\s*$", content[:200], re.M) and "name:" in content[:200]:
                    kind = "skill"
                else:
                    kind = "memory"
        if kind not in ("skill", "memory"):
            return {"action": "no_op", "reason": f"Invalid kind: {kind}"}

        if action == "create" and not content:
            return {"action": "no_op", "reason": "Create requires content (SKILL.md or memory text)"}

        if not name:
            return {"action": "no_op", "reason": "Name is required for create/patch"}

        # Normalize skill name
        if kind == "skill":
            name = _normalize_skill_name(name)

        return {
            "action": action,
            "kind": kind,
            "name": name,
            "content": content,
            "category": category,
            "reason": str(parsed.get("reason", "")),
            "evidence": _ensure_list(parsed.get("evidence")),
            "pattern_fingerprint": str(parsed.get("pattern_fingerprint", "") or "")[:12],
        }

    except PluginLlmTrustError as exc:
        logger.warning("PluginLlm trust denied: %s", exc)
        return {"action": "no_op", "reason": f"LLM trust policy denied: {exc}"}

    except Exception as exc:
        logger.error("LLM proposal failed: %s", exc, exc_info=True)
        return {"action": "no_op", "reason": f"LLM call failed: {exc}"}


def _ensure_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v) for v in val[:10]]
    return []


def _normalize_skill_name(name: str) -> str:
    """Normalize to lowercase-hyphens, max 64 chars."""
    import re
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name[:64]
