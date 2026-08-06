"""Config reader for the refine plugin.

Reads ``plugins.entries.refine.*`` from the Hermes config.yaml.
All values have sensible defaults — config.yaml only provides overrides.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def hermes_home() -> Path:
    """Resolve the Hermes data directory the way Hermes itself does.

    Hardcoding ``~/.hermes`` is wrong on Windows, where the data lives in
    ``%LOCALAPPDATA%\\hermes``, and wrong under profiles. Getting it wrong is
    not loud: the plugin simply finds no trajectory and returns no_op forever
    without ever explaining why.
    """
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def state_db_path() -> Path:
    return hermes_home() / "state.db"


def _load_raw_config() -> Optional[Dict[str, Any]]:
    """Load the full Hermes config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        logger.warning("Cannot load Hermes config; using defaults")
        return None


def _get_refine_entry() -> Dict[str, Any]:
    raw = _load_raw_config()
    if not raw:
        return {}
    plugins_cfg = raw.get("plugins", {})
    if not isinstance(plugins_cfg, dict):
        return {}
    entries = plugins_cfg.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    entry = entries.get("refine", {})
    if not isinstance(entry, dict):
        return {}
    return entry


def get_bool(key: str, default: bool) -> bool:
    """Read a boolean config key with a default."""
    entry = _get_refine_entry()
    val = entry.get(key)
    if isinstance(val, bool):
        return val
    return default


def get_int(key: str, default: int, min_val: int = 1) -> int:
    """Read an integer config key with a default and floor."""
    entry = _get_refine_entry()
    val = entry.get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return max(int(val), min_val)
    return default


def get_str(key: str, default: str = "") -> str:
    """Read a string config key."""
    entry = _get_refine_entry()
    val = entry.get(key)
    if isinstance(val, str):
        return val
    return default


# Convenience accessors
def auto_enabled() -> bool:
    return get_bool("auto_enabled", True)


def auto_min_messages() -> int:
    return get_int("auto_min_messages", 15, min_val=5)


def auto_turn_interval() -> int:
    """Assistant turns between automatic refine attempts; zero disables it."""
    return get_int("auto_turn_interval", 25, min_val=0)


def auto_cooldown_minutes() -> int:
    """Minimum elapsed time between durable automatic-attempt records."""
    return get_int("auto_cooldown_minutes", 20)


def max_edits_per_run() -> int:
    return get_int("max_edits_per_run", 1, min_val=1)


def max_edits_per_proposal() -> int:
    """Maximum inseparable edits one proposal may apply as a single transaction."""
    return get_int("max_edits_per_proposal", 3, min_val=1)


def max_edits_per_day() -> int:
    return get_int("max_edits_per_day", 3, min_val=1)


def only_agent_created() -> bool:
    return get_bool("only_agent_created", True)


def min_pattern_count() -> int:
    """How many times a failure must repeat before it counts as a signal."""
    return get_int("min_pattern_count", 2, min_val=1)


def min_signal_required() -> bool:
    """Skip the LLM call entirely when nothing repeated and nothing was corrected."""
    return get_bool("min_signal_required", True)


def reviewer_fallback_enabled() -> bool:
    """Allow a small reviewer call when the mechanical signal gate finds nothing."""
    return get_bool("reviewer_fallback_enabled", True)


def reviewer_min_messages() -> int:
    """Minimum session size before a reviewer fallback may run."""
    return get_int("reviewer_min_messages", 20, min_val=3)


def reviewer_cooldown_minutes() -> int:
    """Minimum gap between durable reviewer decisions."""
    return get_int("reviewer_cooldown_minutes", 60)


def cross_session_enabled() -> bool:
    return get_bool("cross_session_enabled", True)


def cross_session_days() -> int:
    return get_int("cross_session_days", 7, min_val=1)


def cross_session_max_sessions() -> int:
    return get_int("cross_session_max_sessions", 25, min_val=1)


def dedup_window_days() -> int:
    """Refuse a proposal identical to one already applied within this window."""
    return get_int("dedup_window_days", 7, min_val=1)


def overview_max_entries() -> int:
    """Maximum existing entries of each kind included in a proposal prompt."""
    return get_int("overview_max_entries", 40, min_val=1)


def overview_max_chars() -> int:
    """Maximum characters in each structured overview line."""
    return get_int("overview_max_chars", 240, min_val=1)


def history_max_entries() -> int:
    """Maximum prior create/patch outcomes included in a proposal prompt."""
    return get_int("history_max_entries", 20, min_val=1)


def journal_dir() -> Path:
    default = hermes_home() / "plugins" / "refine"
    return Path(get_str("journal_dir", str(default)))


def prompt_notes_enabled() -> bool:
    """Whether refine may persist and inject plugin-owned prompt notes."""
    return get_bool("prompt_notes_enabled", True)


def prompt_notes_max_count() -> int:
    """Maximum prompt notes injected into one LLM call."""
    return get_int("prompt_notes_max_count", 5, min_val=1)


def prompt_notes_max_chars() -> int:
    """Maximum characters for one complete rendered prompt-note block."""
    return get_int("prompt_notes_max_chars", 600, min_val=1)


def prompt_notes_default_scope() -> str:
    """Default lifetime for new prompt notes; invalid values fail closed to global."""
    scope = get_str("prompt_notes_default_scope", "global").strip().lower()
    return scope if scope in ("global", "session") else "global"
