"""Usefulness ledger for refine-created skills.

Without this, refine is a write-only loop: it creates skills and never learns
whether any of them helped. That matters because skills enter the context of
later sessions — a useless skill is not neutral, it actively costs attention.

The ledger records what refine created and answers two questions:
  1. has this skill been used since it was written?
  2. did the failure it was written for stop happening?

Question 2 is only answerable because the proposal stores the fingerprint of the
pattern it addressed (see ``patterns.py``).

Deliberately dependency-free apart from ``config``: this module must not import
``core``, which imports it back through the command handler.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .config import journal_dir
except ImportError:
    from config import journal_dir  # noqa: F811 — standalone test

logger = logging.getLogger(__name__)

_STATS_FILE_NAME = "skill_stats.json"


def stats_path() -> Path:
    d = journal_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / _STATS_FILE_NAME


def load_stats() -> Dict[str, Any]:
    p = stats_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Cannot read skill stats: %s", exc)
        return {}


def _save_stats(stats: Dict[str, Any]) -> None:
    try:
        stats_path().write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error("Cannot write skill stats: %s", exc)


def record_edit(proposal: Dict[str, Any], journal_id: str) -> None:
    """Register an applied edit so it can be audited later."""
    name = str(proposal.get("name", "")).strip()
    if not name:
        return
    stats = load_stats()
    stats[name] = {
        "created_ts": time.time(),
        "journal_id": journal_id,
        "kind": proposal.get("kind", "skill"),
        "action": proposal.get("action", ""),
        "pattern_fingerprint": proposal.get("pattern_fingerprint", ""),
    }
    _save_stats(stats)


# ── usage counting ──────────────────────────────────────────────────────────


def count_uses(name: str, since_ts: float) -> Optional[int]:
    """How many times a skill was used since *since_ts*.

    Prefers a real counter from the host. The state.db fallback is a
    **heuristic**: it counts messages mentioning the skill name, which
    over-counts discussion about the skill and under-counts silent loads.
    Never present its output as exact — ``audit`` renders it with a "~".

    Returns None when neither source is available.
    """
    # 1. Host API, if Hermes tracks this itself.
    try:
        from tools import skill_usage as _su
        for fn_name in ("get_usage_count", "usage_count", "get_use_count"):
            fn = getattr(_su, fn_name, None)
            if callable(fn):
                try:
                    return int(fn(name))
                except Exception:
                    continue
    except ImportError:
        pass

    # 2. Fallback: approximate from the trajectory store.
    try:
        import sqlite3

        db_path = Path.home() / ".hermes" / "state.db"
        if not db_path.is_file():
            return None
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM messages "
                "WHERE active = 1 AND timestamp > ? AND content LIKE ?",
                (since_ts, f"%{name}%"),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()
    except Exception as exc:
        logger.debug("Usage fallback failed for %s: %s", name, exc)
        return None


def unused_skills(min_age_days: int = 14) -> List[str]:
    """Refine-created skills old enough to judge that were never used.

    Fed back into the proposal prompt as negative examples — the cheapest
    available defence against the model writing plausible-sounding trivia.
    """
    out: List[str] = []
    cutoff = time.time() - (min_age_days * 86400)
    for name, meta in load_stats().items():
        if meta.get("kind") != "skill":
            continue
        created = meta.get("created_ts", 0)
        if created > cutoff:
            continue  # too young to judge
        uses = count_uses(name, created)
        if uses == 0:
            out.append(name)
    return out[:10]


# ── audit ───────────────────────────────────────────────────────────────────


def audit(current_patterns: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Build the audit table.

    Args:
        current_patterns: recent error patterns (from
            ``core.collect_cross_session_patterns``). Used to answer "did the
            failure this skill was written for come back?"
    """
    by_fp: Dict[str, Dict[str, Any]] = {
        str(p.get("fingerprint", "")): p for p in (current_patterns or [])
    }
    now = time.time()
    rows: List[Dict[str, Any]] = []

    for name, meta in sorted(load_stats().items()):
        created = meta.get("created_ts", 0) or now
        age_days = max(0, int((now - created) // 86400))
        uses = count_uses(name, created)
        fp = str(meta.get("pattern_fingerprint", "") or "")

        recurred: Optional[bool] = None
        if fp:
            hit = by_fp.get(fp)
            recurred = bool(hit and (hit.get("last_ts") or 0) > created)

        if recurred is True:
            verdict = "did not help"
        elif uses == 0 and age_days >= 14:
            verdict = "unused"
        elif uses and uses > 0 and recurred is False:
            verdict = "working"
        else:
            verdict = "too early" if age_days < 14 else "unclear"

        rows.append({
            "name": name,
            "kind": meta.get("kind", "skill"),
            "age_days": age_days,
            "uses": uses,
            "pattern_recurred": recurred,
            "verdict": verdict,
            "journal_id": meta.get("journal_id", ""),
        })

    return rows


def format_audit(rows: List[Dict[str, Any]]) -> str:
    """Render the audit table for the chat. Read-only — never deletes."""
    if not rows:
        return "No refine-created skills recorded yet."

    lines = [f"Refine-created entries ({len(rows)}):", ""]
    lines.append(f"  {'name':<28} {'age':>5}  {'uses':>5}  {'recurred':>8}  verdict")
    for r in rows:
        uses = "?" if r["uses"] is None else f"~{r['uses']}"
        rec = {True: "yes", False: "no", None: "—"}[r["pattern_recurred"]]
        lines.append(
            f"  {r['name'][:28]:<28} {str(r['age_days']) + 'd':>5}  {uses:>5}  {rec:>8}  {r['verdict']}"
        )

    dead = [r for r in rows if r["verdict"] in ("unused", "did not help")]
    if dead:
        lines.append("")
        lines.append("Candidates for removal:")
        for r in dead:
            lines.append(f"  {r['name']} — /refine rollback {r['journal_id']}")
        lines.append("")
        lines.append("Nothing was deleted. Run the command yourself if you agree.")

    lines.append("")
    lines.append("Note: use counts are approximate unless Hermes exposes a real counter.")
    return "\n".join(lines)
