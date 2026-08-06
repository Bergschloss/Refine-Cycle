"""Config reader for the refine plugin.

Reads ``plugins.entries.refine.*`` from the Hermes config.yaml.
All values have sensible defaults — config.yaml only provides overrides.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
    return get_bool("auto_enabled", False)


def auto_min_messages() -> int:
    return get_int("auto_min_messages", 15, min_val=5)


def max_edits_per_run() -> int:
    return get_int("max_edits_per_run", 1, min_val=1)


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


def cross_session_enabled() -> bool:
    return get_bool("cross_session_enabled", True)


def cross_session_days() -> int:
    return get_int("cross_session_days", 7, min_val=1)


def cross_session_max_sessions() -> int:
    return get_int("cross_session_max_sessions", 25, min_val=1)


def dedup_window_days() -> int:
    """Refuse a proposal identical to one already applied within this window."""
    return get_int("dedup_window_days", 7, min_val=1)


def journal_dir() -> Path:
    default = Path.home() / ".hermes" / "plugins" / "refine"
    return Path(get_str("journal_dir", str(default)))
