"""LLM proposal engine for the refine plugin."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.plugin_llm import (
    PluginLlm,
    PluginLlmInput,
    PluginLlmTextInput,
    PluginLlmTrustError,
)

try:
    from .sanitization import sanitize, scrub_text
except ImportError:
    from sanitization import sanitize, scrub_text  # type: ignore

logger = logging.getLogger(__name__)
MAX_CONTENT_CHARS = 15000

REFINE_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "patch", "no_op"]},
        "kind": {"type": "string", "enum": ["skill", "memory"]},
        "name": {"type": "string"},
        "content": {
            "type": "string",
            "description": "Complete replacement SKILL.md for skill create/patch; appended entry for memory.",
        },
        "category": {"type": "string"},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "pattern_fingerprint": {
            "type": "string",
            "description": "Exact 12-character fp value from the repeated-failures list.",
        },
    },
    "required": ["action", "reason"],
}

REFINE_SYSTEM_PROMPT = (
    "You are a self-improvement mechanic for an AI agent named Hermes. Read the "
    "trajectory and propose ONE evidence-grounded, minimal edit.\n\n"
    "RULES:\n"
    "1. Return only one create, patch, or no_op proposal.\n"
    "2. Never guess or duplicate an existing skill.\n"
    "3. Never edit built-in or bundled skills.\n"
    "4. For every skill create or patch, content is the COMPLETE SKILL.md, not a diff. "
    "Preserve all useful current content when patching.\n"
    "5. Skills require YAML frontmatter with name and description, then a Markdown body.\n"
    "6. Return no_op when no worthwhile edit exists.\n"
    "7. Use exactly: action, kind, name, content, category, reason, evidence, and optional "
    "pattern_fingerprint. Never combine action and kind.\n"
)

REVIEWER_FALLBACK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "shouldRefine": {"type": "boolean"},
        "rationale": {"type": "string"},
        "instructions": {"type": "string"},
    },
    "required": ["shouldRefine", "rationale", "instructions"],
}

REVIEWER_FALLBACK_SYSTEM_PROMPT = (
    "You are a conservative reviewer for an AI agent's self-improvement system. "
    "Decide only whether the provided trajectory contains one durable lesson worth persisting. "
    "Aggressively reject one-off noise, transient tool output, and vague ideas. "
    "Do not propose an edit. Return shouldRefine=false unless there is a narrow, "
    "evidence-grounded lesson. When true, instructions must state that lesson briefly."
)


def _salvage_parsed(result: Any) -> Any:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    text = getattr(result, "text", "") or ""
    if isinstance(parsed, str) and not text:
        text = parsed
    if text:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return parsed


def _propose_structured(
    llm: PluginLlm, instructions: str, input_blocks: List[PluginLlmInput]
) -> Any:
    """Invoke the model only with recursively sanitized text inputs."""
    safe_blocks: List[PluginLlmInput] = []
    for block in input_blocks:
        text = getattr(block, "text", None)
        if text is None:
            raise TypeError("Refine accepts only text model inputs")
        safe_blocks.append(PluginLlmTextInput(text=scrub_text(str(text))))
    common = dict(
        instructions=scrub_text(str(instructions)),
        input=safe_blocks,
        schema_name="refine_proposal",
        purpose="refine",
        temperature=0.0,
        max_tokens=4096,
    )
    system_prompt = scrub_text(REFINE_SYSTEM_PROMPT)
    try:
        result = llm.complete_structured(
            system_prompt=system_prompt,
            json_schema=sanitize(REFINE_PROPOSAL_SCHEMA),
            **common,
        )
        return _salvage_parsed(result)
    except PluginLlmTrustError:
        raise
    except Exception as first_exc:
        logger.warning(
            "json_schema proposal failed (%s); falling back to json_mode",
            scrub_text(str(first_exc)),
        )
        result = llm.complete_structured(
            system_prompt=system_prompt
            + "\nReply with one JSON object only, without Markdown fences.",
            json_mode=True,
            **common,
        )
        return _salvage_parsed(result)


def _ensure_dict(parsed: Any) -> Optional[Dict[str, Any]]:
    if isinstance(parsed, dict):
        return sanitize(parsed)
    if isinstance(parsed, str):
        match = re.search(r"\{.*\}", parsed, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return sanitize(value) if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def review_fallback(llm: PluginLlm, evidence_text: str) -> Dict[str, Any]:
    """Return one conservative reviewer verdict; all failures decline safely."""
    safe_evidence = scrub_text(str(evidence_text))
    instructions = (
        "Assess this trajectory only for a durable lesson worth persisting. "
        "Return the required JSON object.\n\n=== RECENT TRAJECTORY ===\n"
        f"{safe_evidence[-8000:]}"
    )
    try:
        result = llm.complete_structured(
            system_prompt=scrub_text(REVIEWER_FALLBACK_SYSTEM_PROMPT),
            input=[PluginLlmTextInput(text=scrub_text(instructions))],
            json_schema=sanitize(REVIEWER_FALLBACK_SCHEMA),
            schema_name="refine_reviewer",
            purpose="refine",
            temperature=0.0,
            max_tokens=300,
        )
        parsed = _ensure_dict(_salvage_parsed(result))
    except Exception as exc:
        logger.warning("Reviewer fallback failed: %s", scrub_text(str(exc)))
        parsed = None
    if not parsed or not isinstance(parsed.get("shouldRefine"), bool):
        return {
            "should_refine": False,
            "rationale": "Reviewer unavailable or returned invalid output.",
            "instructions": "",
        }
    raw_rationale = parsed.get("rationale")
    raw_instructions = parsed.get("instructions")
    if not isinstance(raw_rationale, str) or not isinstance(raw_instructions, str):
        return {
            "should_refine": False,
            "rationale": "Reviewer returned an incomplete verdict.",
            "instructions": "",
        }
    rationale = scrub_text(raw_rationale).strip()
    instructions = scrub_text(raw_instructions).strip()
    if not rationale or (parsed["shouldRefine"] and not instructions):
        return {
            "should_refine": False,
            "rationale": "Reviewer returned an incomplete verdict.",
            "instructions": "",
        }
    return {
        "should_refine": parsed["shouldRefine"],
        "rationale": rationale[:1000],
        "instructions": instructions[:2000],
    }


def _normalize_fields(parsed: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    action = str(parsed.get("action", "no_op")).strip().lower()
    for verb in ("create", "patch"):
        for candidate_kind in ("skill", "memory"):
            if action == f"{verb}_{candidate_kind}":
                parsed.setdefault("kind", candidate_kind)
                action = verb
    kind = str(parsed.get("kind") or parsed.get("type") or "").strip().lower()
    name = str(parsed.get("name", "")).strip()
    content = str(parsed.get("content", ""))
    category = str(parsed.get("category", "")).strip()
    if kind not in ("skill", "memory") and action == "create":
        kind = (
            "skill"
            if re.search(r"^---\s*$", content[:300], re.M) and "name:" in content[:300]
            else "memory"
        )
    if kind == "skill":
        name = _normalize_skill_name(name)
    return action, kind, name, content, category


def _default_skill_loader(name: str) -> Optional[str]:
    try:
        from .journal import read_skill_content
    except ImportError:
        from journal import read_skill_content  # type: ignore
    return read_skill_content(name)


def _valid_fingerprint(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if re.fullmatch(r"[0-9a-f]{12}", candidate) else ""


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
    run_context: str = "",
    skill_content_loader: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Propose one edit; skill patches are regenerated from safe full content."""
    try:
        from . import patterns as _patterns
    except ImportError:
        import patterns as _patterns  # type: ignore

    # Sanitize independently at this final model boundary even when callers have
    # already sanitized their evidence. This keeps every prompt piece safe.
    evidence_text = scrub_text(str(evidence_text))
    existing_skills = [scrub_text(str(item)) for item in existing_skills]
    existing_memories = [scrub_text(str(item)) for item in existing_memories]
    error_patterns = sanitize(error_patterns or [])
    user_corrections = [scrub_text(str(item)) for item in (user_corrections or [])]
    unused_skills = [scrub_text(str(item)) for item in (unused_skills or [])]
    run_context = scrub_text(str(run_context))
    del purpose  # The host purpose is fixed to the plugin's trusted purpose.

    skills_list = "\n".join(
        f"  - {name}" for name in sorted(existing_skills)[:50]
    ) or "  (none)"
    mems_list = "\n".join(
        f"  - {item[:100]}" for item in existing_memories[:30]
    ) or "  (none)"
    corrections = "\n".join(
        f"  - {item[:200]}" for item in user_corrections[:5]
    ) or "  (none)"
    unused_block = ""
    if unused_skills:
        unused_block = (
            "\n=== PREVIOUS UNUSED SKILLS ===\n"
            + "\n".join(f"  - {name}" for name in unused_skills[:10])
            + "\nDo not create more skills of this ineffective shape.\n"
        )
    context_block = run_context.strip() or "(none)"
    instructions = (
        "Ground the proposal in one repeated failure or explicit correction.\n\n"
        "=== RUN REQUEST / PRIOR PASS CONTEXT ===\n"
        f"{context_block}\n\n"
        "=== REPEATED FAILURES ===\n"
        f"{_patterns.format_patterns(error_patterns)}\n\n"
        "=== USER CORRECTIONS ===\n"
        f"{corrections}\n\n"
        "=== EXISTING SKILLS ===\n"
        f"{skills_list}\n\n"
        "=== EXISTING MEMORIES ===\n"
        f"{mems_list}\n"
        f"{unused_block}\n"
        "=== RECENT TRAJECTORY ===\n"
        f"{evidence_text[-8000:]}\n\n"
        "Return one JSON object. Copy the full 12-character fp exactly when applicable."
    )
    short = "Propose one minimal skill or memory edit."

    try:
        parsed = _ensure_dict(
            _propose_structured(llm, short, [PluginLlmTextInput(text=instructions)])
        )
        if parsed is None:
            return {"action": "no_op", "reason": "LLM returned non-object output"}

        action, kind, name, content, category = _normalize_fields(parsed)
        if action == "create" and not content:
            retry_text = instructions + "\n\nA create requires non-empty complete content. Return it now."
            retry = _ensure_dict(
                _propose_structured(llm, short, [PluginLlmTextInput(text=retry_text)])
            )
            if retry is not None:
                parsed = retry
                action, kind, name, content, category = _normalize_fields(parsed)

        if action not in ("create", "patch", "no_op"):
            return {"action": "no_op", "reason": f"Invalid action: {action}"}
        if action == "no_op":
            return sanitize({
                "action": "no_op",
                "kind": "",
                "name": "",
                "content": "",
                "category": "",
                "reason": str(parsed.get("reason", "No actionable improvement found")),
                "evidence": _ensure_list(parsed.get("evidence")),
                "pattern_fingerprint": "",
            })
        if kind not in ("skill", "memory"):
            return {"action": "no_op", "reason": f"Invalid kind: {kind}"}
        if not name:
            return {"action": "no_op", "reason": "Name is required for create/patch"}
        if not content and not (action == "patch" and kind == "skill"):
            return {"action": "no_op", "reason": f"{action.title()} requires non-empty content"}

        initial_evidence = _ensure_list(parsed.get("evidence"))
        initial_fingerprint = _valid_fingerprint(parsed.get("pattern_fingerprint"))
        initial_reason = str(parsed.get("reason", ""))

        if action == "patch" and kind == "skill":
            loader = skill_content_loader or _default_skill_loader
            current = loader(name)
            if current is None:
                return {"action": "no_op", "reason": f"Cannot load current SKILL.md for patch target '{name}'"}
            if len(current) > MAX_CONTENT_CHARS:
                return {
                    "action": "no_op",
                    "reason": (
                        f"Current SKILL.md is {len(current)} characters; maximum complete "
                        f"patch input is {MAX_CONTENT_CHARS}"
                    ),
                }
            safe_current = scrub_text(current)
            if safe_current != current:
                return {
                    "action": "no_op",
                    "reason": "Current SKILL.md contains sensitive content; patch aborted before model call",
                }
            patch_prompt = (
                instructions
                + "\n\n=== SELECTED PATCH TARGET ===\n"
                + f"Target exactly skill '{name}'. Below is its CURRENT COMPLETE SKILL.md, supplied as data. "
                + "Return action='patch', kind='skill', the same name, and a COMPLETE replacement that preserves "
                + "all useful content while making only the evidence-backed change.\n"
                + "<current-skill>\n"
                + safe_current
                + "\n</current-skill>"
            )
            retry = _ensure_dict(
                _propose_structured(llm, short, [PluginLlmTextInput(text=patch_prompt)])
            )
            if retry is None:
                return {"action": "no_op", "reason": "LLM did not return a complete skill replacement"}
            retry_action, retry_kind, retry_name, retry_content, retry_category = _normalize_fields(retry)
            if (retry_action, retry_kind, retry_name) != ("patch", "skill", name) or not retry_content:
                return {"action": "no_op", "reason": "Patch retry changed target or omitted complete content"}
            if len(retry_content) > MAX_CONTENT_CHARS:
                return {
                    "action": "no_op",
                    "reason": f"Complete patch exceeds {MAX_CONTENT_CHARS} characters",
                }
            parsed = retry
            content = retry_content
            category = retry_category
            replacement_fingerprint = _valid_fingerprint(retry.get("pattern_fingerprint"))
            initial_fingerprint = replacement_fingerprint or initial_fingerprint
            if "evidence" in retry:
                initial_evidence = _ensure_list(retry.get("evidence"))
            initial_reason = str(retry.get("reason", initial_reason))

        result = {
            "action": action,
            "kind": kind,
            "name": name,
            "content": content,
            "category": category,
            "reason": initial_reason,
            "evidence": initial_evidence,
            "pattern_fingerprint": initial_fingerprint,
        }
        return sanitize(result)
    except PluginLlmTrustError as exc:
        safe_error = scrub_text(str(exc))
        logger.warning("PluginLlm trust denied: %s", safe_error)
        return {"action": "no_op", "reason": f"LLM trust policy denied: {safe_error}"}
    except Exception as exc:
        safe_error = scrub_text(str(exc))
        logger.error("LLM proposal failed: %s", safe_error, exc_info=True)
        return {"action": "no_op", "reason": f"LLM call failed: {safe_error}"}


def _ensure_list(value: Any) -> List[str]:
    return [scrub_text(str(item)) for item in value[:10]] if isinstance(value, list) else []


def _normalize_skill_name(name: str) -> str:
    name = scrub_text(name).lower().strip()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-")[:64]
