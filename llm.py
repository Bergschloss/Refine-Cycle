"""LLM proposal engine for the refine plugin."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from agent.plugin_llm import (
    PluginLlm,
    PluginLlmInput,
    PluginLlmTextInput,
    PluginLlmTrustError,
)

try:
    from . import config
    from .sanitization import sanitize, scrub_text
except ImportError:
    import config  # type: ignore
    from sanitization import sanitize, scrub_text  # type: ignore

logger = logging.getLogger(__name__)

# A complete skill can be this large. Keep the output budget derived from the
# same source of truth: JSON-escaped Markdown tokenizes worse than prose, and
# under-budgeting silently truncates the proposals this limit permits.
MAX_CONTENT_CHARS = 15000
MAX_EXPECTED_OUTCOME_CHARS = 300
_CHARS_PER_TOKEN = 3
_PROPOSAL_ENVELOPE_TOKENS = 1024
PROPOSAL_MAX_TOKENS = (
    MAX_CONTENT_CHARS // _CHARS_PER_TOKEN + _PROPOSAL_ENVELOPE_TOKENS
)
# Reviewer fallback must remain materially cheaper than a full proposal pass.
REVIEWER_MAX_TOKENS = 300

REFINE_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "patch", "no_op"]},
        "kind": {"type": "string", "enum": ["skill", "memory", "prompt"]},
        "name": {"type": "string"},
        "content": {
            "type": "string",
            "description": "Complete replacement SKILL.md for skill create/patch; appended entry for memory; a short, narrow behavioral policy for prompt create.",
        },
        "category": {"type": "string"},
        "reason": {"type": "string"},
        "expected_outcome": {
            "type": "string",
            "description": "One-sentence falsifiable prediction of what this edit should improve.",
        },
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
    "2. Never guess or duplicate an existing skill, memory, or prompt note.\n"
    "3. Never edit built-in or bundled skills.\n"
    "4. For every skill create or patch, content is the COMPLETE SKILL.md, not a diff. "
    "Preserve all useful current content when patching.\n"
    "5. Skills require YAML frontmatter with name and description, then a Markdown body.\n"
    "6. A prompt note must be action=create and kind=prompt, with one or two lines in the "
    "exact format 'When <specific condition>, <one action>.' It must be a narrow behavioral "
    "policy, never a procedure, broad/global instruction, memory, skill, or replacement system prompt.\n"
    "7. Return no_op when no worthwhile edit exists.\n"
    "8. expected_outcome is optional; when present, make it one falsifiable sentence about "
    "what the edit should improve and how to check it. It must not restate reason.\n"
    "9. Use exactly: action, kind, name, content, category, reason, expected_outcome, evidence, and optional "
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


class _Reply(NamedTuple):
    """Structured reply plus a failure that must not be disguised as no_op."""

    parsed: Optional[Dict[str, Any]]
    failure: str = ""
    detail: str = ""


def _output_tokens(result: Any) -> int:
    """Read optional host usage without requiring test doubles to expose it."""
    usage = getattr(result, "usage", None)
    try:
        return max(0, int(getattr(usage, "output_tokens", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _has_incomplete_json_structure(text: str) -> bool:
    """Detect unclosed strings/braces without attempting to repair model JSON."""
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return in_string or depth > 0


def _salvage_parsed(result: Any, *, requested_max_tokens: int) -> _Reply:
    """Parse one reply and name incomplete output instead of returning a false no_op."""
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict):
        return _Reply(parsed)
    text = getattr(result, "text", "") or ""
    if isinstance(parsed, str) and not text:
        text = parsed
    if text:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return _Reply(value)
            except json.JSONDecodeError:
                pass
    output_tokens = _output_tokens(result)
    if not text:
        if output_tokens:
            model = scrub_text(str(getattr(result, "model", "")))
            logger.warning(
                "Refine model returned output but no final text (model=%s)",
                model or "unknown",
            )
            return _Reply(
                None,
                "no_final_text",
                "Model returned output but no final structured answer.",
            )
        return _Reply(None, "malformed", "Model returned no structured answer.")
    if output_tokens >= requested_max_tokens:
        return _Reply(
            None,
            "truncated",
            "Model output reached its token limit before the proposal completed.",
        )
    if _has_incomplete_json_structure(text):
        return _Reply(
            None,
            "truncated",
            "Model output ended before the proposal JSON completed.",
        )
    return _Reply(None, "malformed", "Model returned malformed structured output.")


def _incomplete_proposal(reply: _Reply) -> Dict[str, Any]:
    return {
        "action": "no_op",
        "failure": reply.failure,
        "reason": reply.detail,
    }


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
        max_tokens=PROPOSAL_MAX_TOKENS,
    )
    system_prompt = scrub_text(REFINE_SYSTEM_PROMPT)
    try:
        result = llm.complete_structured(
            system_prompt=system_prompt,
            json_schema=sanitize(REFINE_PROPOSAL_SCHEMA),
            **common,
        )
        reply = _salvage_parsed(result, requested_max_tokens=PROPOSAL_MAX_TOKENS)
        return reply.parsed or _incomplete_proposal(reply)
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
        reply = _salvage_parsed(result, requested_max_tokens=PROPOSAL_MAX_TOKENS)
        return reply.parsed or _incomplete_proposal(reply)


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
            max_tokens=REVIEWER_MAX_TOKENS,
        )
        reply = _salvage_parsed(result, requested_max_tokens=REVIEWER_MAX_TOKENS)
        if reply.failure:
            rationale = (
                "Reviewer returned no final answer."
                if reply.failure == "no_final_text"
                else "Reviewer unavailable or returned invalid output."
            )
            return {
                "should_refine": False,
                "rationale": rationale,
                "instructions": "",
                "failure": reply.failure,
            }
        parsed = _ensure_dict(reply.parsed)
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


def normalize_expected_outcome(value: Any) -> str:
    """Return a compact, sanitized prediction or an empty optional value."""
    if not isinstance(value, str):
        return ""
    return scrub_text(value).strip()[:MAX_EXPECTED_OUTCOME_CHARS]


def _normalize_fields(parsed: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    action = str(parsed.get("action", "no_op")).strip().lower()
    for verb in ("create", "patch"):
        for candidate_kind in ("skill", "memory", "prompt"):
            if action == f"{verb}_{candidate_kind}":
                parsed.setdefault("kind", candidate_kind)
                action = verb
    kind = str(parsed.get("kind") or parsed.get("type") or "").strip().lower()
    name = str(parsed.get("name", "")).strip()
    content = str(parsed.get("content", ""))
    category = str(parsed.get("category", "")).strip()
    if kind not in ("skill", "memory", "prompt") and action == "create":
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


def _overview_text(value: Any) -> str:
    """Sanitize untrusted host metadata into one physical prompt-line value."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", scrub_text(str(value))).strip()


def _truncate_overview_line(value: str, limit: int) -> str:
    """Keep a bounded line readable without leaving an incomplete escape."""
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"
    prefix = value[:limit - 1]
    last_escape = prefix.rfind("\\")
    if last_escape >= 0:
        escape = prefix[last_escape:]
        if escape == "\\" or (escape.startswith("\\u") and len(escape) < 6):
            prefix = prefix[:last_escape]
    return prefix + "…"


def _render_overview(
    entries: List[Any], *, entry_kind: str, max_entries: int, max_chars: int
) -> str:
    """Render safe, bounded existing-context entries for the proposal prompt."""
    indent = "  " if max_chars > 2 else ""
    text_limit = max_chars - len(indent)
    lines: List[str] = []
    for entry in entries[:max_entries]:
        if entry_kind == "skill":
            if isinstance(entry, dict):
                name = _overview_text(entry.get("name", ""))
                description = _overview_text(entry.get("description", ""))
                category = _overview_text(entry.get("category", ""))
                version = entry.get("version")
            else:
                name = _overview_text(entry)
                description = ""
                category = ""
                version = None
            if not name:
                continue
            details = [category] if category else []
            if isinstance(version, int) and version >= 1:
                details.append(f"v{version}")
            line = f"[skill:{name}]"
            if description:
                line += f" {description}"
            if details:
                line += f" ({', '.join(details)})"
        else:
            if isinstance(entry, dict):
                raw_snippet = entry.get("snippet", entry.get("content", ""))
            else:
                raw_snippet = entry
            snippet = _overview_text(raw_snippet)
            if not snippet:
                continue
            line = f"[memory] {snippet}"
        lines.append(indent + _truncate_overview_line(line, text_limit))
    remaining = max(0, len(entries) - max_entries)
    if remaining:
        lines.append(indent + _truncate_overview_line(f"… +{remaining} more", text_limit))
    if lines:
        return "\n".join(lines)
    return indent + _truncate_overview_line("(none)", text_limit)


def propose(
    llm: PluginLlm,
    evidence_text: str,
    existing_skills: List[Any],
    existing_memories: List[Any],
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
    existing_skills = list(existing_skills or [])
    existing_memories = list(existing_memories or [])
    error_patterns = sanitize(error_patterns or [])
    user_corrections = [scrub_text(str(item)) for item in (user_corrections or [])]
    unused_skills = [scrub_text(str(item)) for item in (unused_skills or [])]
    run_context = scrub_text(str(run_context))
    overview_max_entries = config.overview_max_entries()
    overview_max_chars = config.overview_max_chars()
    del purpose  # The host purpose is fixed to the plugin's trusted purpose.

    skills_list = _render_overview(
        existing_skills,
        entry_kind="skill",
        max_entries=overview_max_entries,
        max_chars=overview_max_chars,
    )
    mems_list = _render_overview(
        existing_memories,
        entry_kind="memory",
        max_entries=overview_max_entries,
        max_chars=overview_max_chars,
    )
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
    short = "Propose one minimal skill, memory, or prompt-note edit."

    try:
        parsed = _ensure_dict(
            _propose_structured(llm, short, [PluginLlmTextInput(text=instructions)])
        )
        if parsed is None:
            return {"action": "no_op", "reason": "LLM returned non-object output"}
        if parsed.get("failure"):
            return sanitize(parsed)

        action, kind, name, content, category = _normalize_fields(parsed)
        if action == "create" and not content:
            retry_text = instructions + "\n\nA create requires non-empty complete content. Return it now."
            retry = _ensure_dict(
                _propose_structured(llm, short, [PluginLlmTextInput(text=retry_text)])
            )
            if retry is not None:
                if retry.get("failure"):
                    return sanitize(retry)
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
                "expected_outcome": normalize_expected_outcome(
                    parsed.get("expected_outcome")
                ),
                "evidence": _ensure_list(parsed.get("evidence")),
                "pattern_fingerprint": "",
            })
        if kind not in ("skill", "memory", "prompt"):
            return {"action": "no_op", "reason": f"Invalid kind: {kind}"}
        if not name and kind != "prompt":
            return {"action": "no_op", "reason": "Name is required for skill and memory create/patch"}
        if not content and not (action == "patch" and kind == "skill"):
            return {"action": "no_op", "reason": f"{action.title()} requires non-empty content"}

        initial_evidence = _ensure_list(parsed.get("evidence"))
        initial_fingerprint = _valid_fingerprint(parsed.get("pattern_fingerprint"))
        initial_reason = str(parsed.get("reason", ""))
        initial_expected_outcome = normalize_expected_outcome(
            parsed.get("expected_outcome")
        )

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
            if retry.get("failure"):
                return sanitize(retry)
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
            if "expected_outcome" in retry:
                initial_expected_outcome = normalize_expected_outcome(
                    retry.get("expected_outcome")
                )

        result = {
            "action": action,
            "kind": kind,
            "name": name,
            "content": content,
            "category": category,
            "reason": initial_reason,
            "expected_outcome": initial_expected_outcome,
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
