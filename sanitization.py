"""Recursive credential redaction shared by evidence and persistence paths."""

import re
from typing import Any

_REDACTED = "[REDACTED]"

_FIXED_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"gh[po]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"ntn_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{15,}"),
    re.compile(r"SG\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}"),
    re.compile(r"dop_v1_[a-f0-9]{60,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.S,
    ),
]

# Match exact generic labels and common compounds such as client_secret,
# access-token, refreshToken, and github_api_key.
_SECRET_KEY = (
    r"(?:authorization|bearer|"
    r"[A-Za-z0-9_-]*(?:api[_-]?key|password|passwd|secret|token)[A-Za-z0-9_-]*)"
)
_QUOTED_SECRET = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SECRET_KEY}[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>(?:\\.|(?!\2).)+)(?P=quote)"
)
_UNQUOTED_SECRET = re.compile(
    rf"(?i)(?P<prefix>\b{_SECRET_KEY}\b\s*[:=]\s*)"
    r"(?P<value>[^\s,;\}\]\[]{6,})"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_.+/=-]{8,}")
_URL_CREDENTIALS = re.compile(r"(https?://)[^/\s:@]+:[^/\s:@]+@")
_ENV_SECRET = re.compile(
    r"(?m)^(\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*\s*=\s*)\S+$"
)
_NON_SECRETS = {"true", "false", "null", "none", "enabled", "disabled"}


def _replace_quoted(match: re.Match) -> str:
    return (
        f"{match.group('prefix')}{match.group('quote')}"
        f"{_REDACTED}{match.group('quote')}"
    )


def _replace_unquoted(match: re.Match) -> str:
    value = match.group("value")
    if value.lower() in _NON_SECRETS or value.isdigit():
        return match.group(0)
    # A prior pass in the same _scrub_chunk may have already inserted the marker
    # (e.g. _ENV_SECRET produced API_KEY=[REDACTED]). Do not re-match it.
    if value.upper().startswith("REDACTED") or value.startswith("[REDACTED"):
        return match.group(0)
    return f"{match.group('prefix')}{_REDACTED}"


def _scrub_chunk(text: str) -> str:
    for pattern in _FIXED_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    text = _ENV_SECRET.sub(r"\1[REDACTED]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _QUOTED_SECRET.sub(_replace_quoted, text)
    return _UNQUOTED_SECRET.sub(_replace_unquoted, text)


def scrub_text(text: str) -> str:
    """Redact credentials while preserving existing redaction markers exactly."""
    if not text:
        return text
    # Sanitized proposals are deliberately scrubbed at several trust boundaries.
    # Splitting around the marker prevents a later generic ``token=...`` match
    # from consuming part of it and makes sanitation strictly idempotent.
    return _REDACTED.join(_scrub_chunk(chunk) for chunk in text.split(_REDACTED))


def sanitize(value: Any) -> Any:
    """Recursively scrub every string in a journal- or model-bound value."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub_text(str(key)): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize(item) for item in value)
    if isinstance(value, set):
        return {sanitize(item) for item in value}
    return value
