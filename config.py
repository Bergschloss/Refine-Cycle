"""Config reader for the refine plugin.

Reads ``plugins.entries.refine.*`` from the Hermes config.yaml.
All values have sensible defaults — config.yaml only provides overrides.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    if not isinstance(raw, dict) or not raw:
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


def config_available() -> bool:
    """Whether the Hermes config was both readable and shaped like a config.

    A file that parses into a non-mapping (a bare list, a string) is as
    unusable as one that fails to parse, so it must not be reported as
    available: every accessor below would raise on it.
    """
    return isinstance(_load_raw_config(), dict)


# Convenience accessors
def auto_enabled() -> bool:
    """Automatic refinement is on by default, but never on an unreadable config.

    The default is ``True`` so refinement works right after install. That default
    may only apply when the config was actually readable: defaulting to ``True``
    while the file cannot be parsed would silently override an explicit
    ``auto_enabled: false`` the user did set, and resume model-bound trajectory
    analysis they had turned off. An unreadable config therefore fails closed,
    and ``/refine status`` reports that as the reason.
    """
    if not config_available():
        return False
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


def skip_session_sources() -> List[str]:
    """Session sources to skip for automatic and manual refinement.

    A session whose ``source`` column matches one of these values is not
    analysed. Intended for machine-generated sessions (cron, batch) whose
    trajectory is noise rather than signal. Invalid config (not a list of
    strings) falls back to the default rather than raising.
    """
    entry = _get_refine_entry()
    val = entry.get("skip_session_sources")
    if isinstance(val, list) and all(isinstance(item, str) for item in val):
        return [item.strip().lower() for item in val if item.strip()]
    return ["cron"]


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


def _llm_entry() -> Dict[str, Any]:
    block = _get_refine_entry().get("llm")
    return block if isinstance(block, dict) else {}


def llm_provider() -> str:
    """Provider to request for refine's own calls; empty means host default."""
    value = _llm_entry().get("provider")
    return value.strip() if isinstance(value, str) else ""


def llm_model() -> str:
    """Model to request for refine's own calls; empty means host default.

    Unset, Hermes resolves refine's model through its ``auto`` path, which
    prefers the live main model. Pinning makes the target deterministic and
    immune to the host's auxiliary client cache keeping an older model. The
    host still gates it: without ``allow_model_override`` (and
    ``allow_provider_override``) the request is refused rather than applied.
    """
    value = _llm_entry().get("model")
    return value.strip() if isinstance(value, str) else ""


def llm_allow_model_override() -> bool:
    """Whether the trust policy allows refine to request a specific model."""
    return bool(_llm_entry().get("allow_model_override", False))


def llm_allow_provider_override() -> bool:
    """Whether the trust policy allows refine to request a specific provider."""
    return bool(_llm_entry().get("allow_provider_override", False))


def live_main_target() -> Dict[str, str]:
    """Best-effort read of the host's live main provider/model.

    Uses a private Hermes API (``_read_main_provider`` / ``_read_main_model`` in
    ``agent.auxiliary_client``) because the host exposes no public accessor. Both
    names were confirmed present in a real installation; a private name can still
    move, so the import is guarded and yields no live value on failure. The caller
    must not treat that as an error — it only means no live model is available,
    and ``effective_llm_target`` then reports ``host_default`` instead of naming a
    target it cannot confirm.
    """
    try:
        from agent.auxiliary_client import _read_main_provider, _read_main_model

        provider = _read_main_provider()
        model = _read_main_model()
        result: Dict[str, str] = {}
        if provider:
            result["provider"] = provider
        if model:
            result["model"] = model
        return result
    except Exception:
        return {}


def effective_llm_target() -> Dict[str, Any]:
    """Resolve one effective model/provider target for refine.

    Priority:
      1. Command override (``/refine model <target>``)
      2. Config (``plugins.entries.refine.llm.model`` / ``.provider``)
      3. Live Hermes main model (best-effort, internal API)
      4. Nothing — let the host decide

    A command override fills any field it leaves unset from the config, because a
    configured provider is an explicit instruction the command did not revoke;
    dropping it would send the new model to whatever default the host picks.

    An unset field is deliberately **not** filled from the live model. Omitting a
    field already means "let Hermes resolve it", and Hermes resolves it to the
    live value — naming it here would claim the user chose something they did not,
    and would spend a trust flag to reach the same result.

    A configured ``llm.model`` may be namespaced (``vendor/name``); a provider may
    not. Both are checked against the same rule the override store applies, so a
    value refused from ``/refine model`` cannot slip in through ``config.yaml``.

    Returns ``{"provider": ..., "model": ..., "source": ..., "issues": [...]}``.
    Provider/model may be empty strings; ``source`` is always set; ``issues`` lists
    every configured or stored value that was discarded, and why. It is a list
    rather than one string because several can fail at once, and reporting only
    the last would have the user fix one and rediscover the next.
    """
    try:
        from . import journal
    except ImportError:
        import journal  # type: ignore

    issues: list = []
    cfg_provider = llm_provider()
    cfg_model = llm_model()
    # The override store refuses unusable values. The config writes the same
    # field, so it meets the same rule here — otherwise the check would hold on
    # one path and not the other, and a value refused from /refine model would be
    # accepted from config.yaml. The reason is reported without the value, which
    # may be a pasted credential.
    for key, value, namespaced in (
        ("provider", cfg_provider, False),
        ("model", cfg_model, True),
    ):
        problem = (
            journal.model_override_field_problem(value, allow_namespace=namespaced)
            if value
            else ""
        )
        if problem:
            issues.append(f"config llm.{key} was ignored because {problem}")
            if key == "provider":
                cfg_provider = ""
            else:
                cfg_model = ""

    # 1. Command override, with the config filling any field it leaves unset.
    override, state = journal.read_model_override_state()
    if state == "rejected":
        issues.append("model_override.json is present but unusable, so it was ignored")
    elif state == "unreadable":
        issues.append("model_override.json could not be read, so it was not applied")
    if override:
        return {
            "provider": override.get("provider", "") or cfg_provider,
            "model": override.get("model", "") or cfg_model,
            "source": "command",
            "issues": issues,
        }

    # 2. Config
    if cfg_provider or cfg_model:
        return {
            "provider": cfg_provider,
            "model": cfg_model,
            "source": "config",
            "issues": issues,
        }

    # 3. Live Hermes main model. Not put through the rule above: the value comes
    # from the host's own runtime, not from user input, and refusing the host's
    # answer would leave refine with no target at all.
    live = live_main_target()
    if live.get("model") or live.get("provider"):
        return {
            "provider": live.get("provider", ""),
            "model": live.get("model", ""),
            "source": "live",
            "issues": issues,
        }

    # 4. Nothing
    return {"provider": "", "model": "", "source": "host_default", "issues": issues}


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
