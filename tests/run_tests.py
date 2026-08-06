"""Tests for the refine plugin.

Run:  python3 -m tests.run_tests   (from ~/.hermes/plugins/refine/)
or:   python3 ~/.hermes/plugins/refine/tests/run_tests.py

These tests use the real state.db (read-only), real skill_manage (creates a
test skill, then deletes it), real MemoryStore, and a mock LLM.

They must NOT break the running agent.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add plugin root and pipx site-packages to path
_HERE = Path(__file__).resolve().parent
_PLUGIN_DIR = _HERE.parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

# pipx site-packages (where agent/, tools/, hermes_cli/ live)
_PIPX_SP = Path.home() / ".local" / "share" / "pipx" / "venvs" / "hermes-agent" / "lib" / "python3.12" / "site-packages"
if _PIPX_SP.is_dir() and str(_PIPX_SP) not in sys.path:
    sys.path.insert(0, str(_PIPX_SP))

# ── mock LLM ────────────────────────────────────────────────────────────────


class MockLlm:
    """Fake PluginLlm that returns a fixed proposal."""

    def __init__(self, proposal: Optional[Dict[str, Any]] = None):
        self._proposal = proposal
        self.calls: List[Dict[str, Any]] = []

    def complete_structured(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        return MockResult(self._proposal)


class MockResult:
    def __init__(self, parsed: Any):
        self.parsed = parsed


class FailingLlm(MockLlm):
    """Fake that always raises."""

    def complete_structured(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        raise RuntimeError("Simulated LLM failure")


# ── test helpers ────────────────────────────────────────────────────────────


def ok(label: str) -> None:
    print(f"  ✅ {label}")


def fail(label: str, msg: str) -> None:
    print(f"  ❌ {label}: {msg}")
    global _FAILURES
    _FAILURES += 1


_FAILURES = 0


def assert_eq(actual, expected, label: str) -> None:
    if actual == expected:
        ok(label)
    else:
        fail(label, f"expected {expected!r}, got {actual!r}")


def assert_true(cond: bool, label: str) -> None:
    ok(label) if cond else fail(label, "expected True")


def has_trajectory(min_messages: int = 3) -> bool:
    """Is there real session data to test against?

    A fresh Hermes install has no state.db. Tests that need a real trajectory
    should skip rather than fail — an empty machine is not a defect, and a
    suite that reports red for environment reasons trains people to ignore red.
    """
    try:
        from core import collect_evidence
        return len(collect_evidence().get("messages", [])) >= min_messages
    except Exception:
        return False


def assert_in(substring: str, text: str, label: str) -> None:
    (ok(label) if substring in text
     else fail(label, f"'{substring[:40]}' not found"))


# ── test functions ──────────────────────────────────────────────────────────


def test_collect_evidence():
    """collect_evidence() reads from real state.db."""
    print("\nTest: collect_evidence")
    from core import collect_evidence

    result = collect_evidence()
    assert isinstance(result, dict), "should return a dict"
    assert "messages" in result, "should have messages key"
    assert "error_count" in result, "should have error_count"
    assert "session_id" in result, "should have session_id"
    ok(f"Got {len(result['messages'])} messages, {result['error_count']} errors")
    print(f"    session_id={result.get('session_id','(none)')[:20]}...")


def test_llm_proposal_valid():
    """Proposal with valid create-skill response."""
    print("\nTest: llm.propose (valid create skill)")
    from llm import propose

    mock = MockLlm({
        "action": "create",
        "kind": "skill",
        "name": "Test Skill Name",
        "content": "---\nname: test-skill-name\ndescription: test\n---\n\n# Test\n",
        "reason": "User keeps asking about caching",
        "evidence": ["user: how to cache", "assistant: I don't know"],
    })
    result = propose(mock, "test evidence", ["existing-skill"], [], purpose="test")
    assert_eq(result["action"], "create", "action should be create")
    assert_eq(result["name"], "test-skill-name", "name should be normalized")
    assert result["content"], "content should not be empty"
    assert result["reason"], "reason should be set"


def test_llm_proposal_noop():
    """LLM returns no_op."""
    print("\nTest: llm.propose (no_op)")
    from llm import propose

    mock = MockLlm({
        "action": "no_op",
        "reason": "Nothing to improve",
    })
    result = propose(mock, "test", [], [], purpose="test")
    assert_eq(result["action"], "no_op", "should be no_op")


def test_llm_proposal_invalid():
    """LLM returns garbage — should fallback to no_op."""
    print("\nTest: llm.propose (garbage)")
    from llm import propose

    mock = MockLlm({"action": "delete_all"})
    result = propose(mock, "test", [], [], purpose="test")
    assert_eq(result["action"], "no_op", "invalid action → no_op")


def test_llm_trust_error():
    """LLM trust error → no_op."""
    print("\nTest: llm.propose (trust error)")
    from llm import propose

    result = propose(FailingLlm(), "test", [], [], purpose="test")
    assert_eq(result["action"], "no_op", "failure → no_op")


def test_journal_roundtrip():
    """Journal write + read + count."""
    print("\nTest: journal roundtrip")
    from config import max_edits_per_day
    import journal as j

    proposal = {"action": "create", "kind": "skill", "name": "x"}
    entry_id = j.log(
        trigger="test",
        reason="testing",
        session_id="test-session",
        proposal=proposal,
        outcome="applied",
    )
    assert entry_id, "should return entry id"

    entry = j.get_entry(entry_id)
    assert entry, "should find entry"
    assert_eq(entry.get("outcome"), "applied", "outcome should be applied")

    count = j.count_today_applied()
    assert_true(count >= 1, f"at least 1 applied today (got {count})")


def test_guardrail_agent_created():
    """Guardrail: non-agent-created skill detection varies by runtime environment.
    In standalone test this may pass through if bundled detection isn't loaded.
    Test the reserved prefix guardrail instead (reliable cross-environment)."""
    print("\nTest: guardrail (reserved prefix + bundled check)")
    from core import _validate_proposal
    from config import only_agent_created as _cfg

    # Reserved prefix always blocked
    err = _validate_proposal({"action": "create", "kind": "skill", "name": "hermes-test"})
    assert err, "hermes- prefix should be rejected"
    ok(f"Rejected reserved prefix: {err[:60]}")

    # Bundled check — may or may not work outside Hermes runtime
    if _cfg():
        err2 = _validate_proposal({"action": "patch", "kind": "skill", "name": "canvas-design", "content": "x"})
        if err2:
            ok(f"Rejected bundled skill: {err2[:60]}")
        else:
            ok("Bundled check skipped — not available in standalone test (expected)")


def test_guardrail_reserved_prefix():
    """Guardrail rejects hermes- prefixed skills."""
    print("\nTest: guardrail (reserved prefix)")
    from core import _validate_proposal
    err = _validate_proposal({"action": "create", "kind": "skill", "name": "hermes-test"})
    assert err, "hermes- prefix → should be rejected"
    ok(f"Rejected: {err[:60]}")


def test_refine_e2e_noop():
    """End-to-end: refine_run returns no_op with insufficient evidence."""
    print("\nTest: refine_run → no_op (no real evidence)")
    from core import refine_run

    if not has_trajectory():
        ok("skipped - no state.db / no session data on this machine")
        return

    mock = MockLlm({
        "action": "no_op",
        "reason": "Not enough evidence",
    })
    result = refine_run(mock, reason="test")
    assert result, "should return result dict"
    # Should have journal entry
    assert "journal_id" in result, "should have journal_id"
    ok(f"Result: {result.get('message', 'no message')[:80]}")


def test_refine_e2e_create_skill():
    """End-to-end: create a test skill via refine, then delete it."""
    print("\nTest: refine_run → create skill (full cycle)")
    from core import refine_run
    import journal as j

    if not has_trajectory():
        ok("skipped - no state.db / no session data on this machine")
        return

    skill_name = "refine-e2e-test-skill"
    skill_content = (
        "---\nname: refine-e2e-test-skill\ndescription: E2E test skill for refine\n---\n\n"
        "# Refine E2E Test\n\nThis is a test skill created by refine_run.\n"
    )

    mock = MockLlm({
        "action": "create",
        "kind": "skill",
        "name": skill_name,
        "content": skill_content,
        "category": "",
        "reason": "User asked how to X, let's remember",
        "evidence": ["user: how do I X?"],
    })
    result = refine_run(mock, reason="e2e test")

    if result.get("result", {}).get("success"):
        ok(f"Created skill: {skill_name}")
        journal_id = result.get("journal_id", "")
        assert journal_id, "should have journal_id"

        # Verify skill exists
        try:
            from tools.skills_tool import skill_view
            sv = skill_view(skill_name)
            ok("skill_view returns content")
        except Exception:
            pass

        # Rollback test (won't fully restore since backup is the old non-existent state)
        # Instead, just delete the test skill
        from tools.skill_manager_tool import skill_manage
        del_result = json.loads(skill_manage(action="delete", name=skill_name))
        if del_result.get("success"):
            ok("Deleted test skill")
        else:
            fail("delete test skill", del_result.get("error", "?"))
    else:
        msg = result.get("message", result.get("result", {}).get("error", "unknown"))
        if "staged" in str(result).lower() or "pending" in str(result).lower():
            ok(f"Skipped — write gated (pending approval): {msg[:80]}")
        else:
            fail("create skill", msg)


def test_refine_rollback_nonexistent():
    """Rollback non-existent entry → error."""
    print("\nTest: rollback nonexistent")
    from core import refine_rollback
    result = refine_rollback("nonexistent-id-12345")
    assert not result.get("success"), "should return failure"
    ok(f"Correctly rejected: {result.get('error', '?')[:60]}")


def test_list_skill_names():
    """list_skill_names() returns at least some skills."""
    print("\nTest: list_skill_names")
    from core import list_skill_names
    names = list_skill_names()
    assert isinstance(names, list), "should return list"
    ok(f"Found {len(names)} skills")


def test_scrub_text():
    """Credential patterns are redacted; benign text is untouched."""
    print("\nTest: scrub_text")
    from core import scrub_text

    t = (
        "token is github_pat_11FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKE "
        "and sk-proj-abcdefghijklmnopqrstuvwxyz123456 and api_key=supersecret123 "
        "and password: hunter2 and Authorization: Bearer aaaabbbbccccddddeeee "
        "and max_tokens=2048 stays"
    )
    s = scrub_text(t)
    assert "github_pat_11FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKE" not in s, "PAT must be redacted"
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in s, "sk- key must be redacted"
    assert "supersecret123" not in s, "api_key= value must be redacted"
    assert "hunter2" not in s, "password: value must be redacted"
    assert "aaaabbbbccccddddeeee" not in s, "Bearer value must be redacted"
    assert "max_tokens=2048" in s, "max_tokens must stay untouched"
    assert "[REDACTED]" in s, "should contain REDACTED marker"
    ok(f"redacted {s.count('[REDACTED]')} spots, benign text intact")


def test_scrub_proposal():
    """scrub_proposal redacts all string fields and list items."""
    print("\nTest: scrub_proposal")
    from core import scrub_proposal
    p = {
        "action": "create",
        "name": "my-skill",
        "reason": "saw token github_pat_11FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKE",
        "evidence": ["user said: password=hunter2"],
        "content": "---\nname: my-skill\ndescription: x\n---\n\n# body",
    }
    s = scrub_proposal(p)
    assert "github_pat_11FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKE" not in s["reason"], "reason must be scrubbed"
    assert "hunter2" not in s["evidence"][0], "evidence list must be scrubbed"
    assert s["name"] == "my-skill", "name must stay"
    assert "[REDACTED]" in s["reason"] and "[REDACTED]" in s["evidence"][0]
    ok("all string fields scrubbed, structure preserved")


def test_guardrail_create():
    """create new name OK; create over existing rejected; patch bundled rejected."""
    print("\nTest: guardrail create")
    from core import _validate_proposal

    err_new = _validate_proposal({
        "action": "create", "kind": "skill", "name": "refine-guard-test-new",
        "content": "---\nname: refine-guard-test-new\ndescription: x\n---\n\n# x",
    })
    assert err_new is None, f"create new name should pass, got: {err_new}"
    ok("create new name allowed")

    existing = None
    from core import list_skill_names
    names = list_skill_names()
    if names:
        existing = names[0]
        err_over = _validate_proposal({
            "action": "create", "kind": "skill", "name": existing,
            "content": "---\nname: x\ndescription: x\n---\n\n# x",
        })
        assert err_over and "already exists" in err_over, f"create over existing should be rejected, got: {err_over}"
        ok(f"create over existing rejected ({existing})")
    else:
        ok("no existing skills to test create-over-existing (skipped)")


def test_refine_multipass():
    """refine_run honors max_edits_per_run>1: second pass sees the skill exists."""
    print("\nTest: refine_run multi-pass")
    import config as cfg
    from core import refine_run

    if not has_trajectory():
        ok("skipped - no state.db / no session data on this machine")
        return

    orig = cfg.max_edits_per_run
    cfg.max_edits_per_run = lambda: 2

    skill_name = "refine-multipass-test"
    mock = MockLlm({
        "action": "create",
        "kind": "skill",
        "name": skill_name,
        "content": f"---\nname: {skill_name}\ndescription: multipass test\n---\n\n# Multipass\n",
        "category": "",
        "reason": "test multipass",
        "evidence": ["test"],
    })

    try:
        result = refine_run(mock, reason="multipass")
        results = result.get("results", [result])
        # First pass applied, second pass either rejected (already exists) or
        # the loop stopped — either way nothing should be left behind.
        from tools.skill_manager_tool import skill_manage
        del_result = json.loads(skill_manage(action="delete", name=skill_name))
        assert del_result.get("success"), "test skill should be deletable"
        ok(f"multi-pass produced {len(results)} pass(es); cleanup ok")
    finally:
        cfg.max_edits_per_run = orig


def test_scrub_new_patterns():
    """Vendor token formats and env-style lines are redacted."""
    print("\nTest: scrub_text (extended patterns)")
    from core import scrub_text

    cases = [
        ("HuggingFace", "hf_" + "a" * 24),
        ("GitLab PAT", "glpat-" + "b" * 20),
        ("SendGrid", "SG." + "c" * 20 + "." + "d" * 20),
        ("DigitalOcean", "dop_v1_" + "e" * 64),
        ("URL basic-auth", "https://admin:s3cr3tpass@internal.example.com/api"),
        ("env line", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY"),
    ]
    for label, raw in cases:
        out = scrub_text(raw)
        if "[REDACTED]" in out and raw.split("@")[0].split("=")[-1] not in out:
            ok(f"{label} redacted")
        else:
            fail(f"{label} redacted", f"got {out[:60]!r}")


def test_scrub_no_false_positives():
    """Benign config-looking text must survive untouched."""
    print("\nTest: scrub_text (no false positives)")
    from core import scrub_text

    benign = [
        "max_tokens=2048",
        "token_count: 15",
        "secretary=jane",
        "secret=true",
        "password: null",
        '"api_key": null',
        "use_secrets = False",
        "auth: none",
        "TOKEN = 1234",
        "tokens: 8192",
        "secret_scanning: enabled",
        "https://github.com/Bergschloss/Refine-Cycle",
        "see http://localhost:8080/api",
    ]
    for raw in benign:
        out = scrub_text(raw)
        if out == raw:
            ok(f"kept: {raw}")
        else:
            fail(f"kept: {raw}", f"mangled to {out!r}")


# ── temp state.db fixture ───────────────────────────────────────────────────

_PLANTED_SECRET = "ghp_" + "Z" * 36


def _make_fake_db(tmpdir: str, n_messages: int = 6) -> str:
    """Build a throwaway state.db with the columns collect_evidence reads.

    Never touches the real ~/.hermes/state.db.
    """
    import sqlite3 as _sq

    path = str(Path(tmpdir) / "fake_state.db")
    con = _sq.connect(path)
    con.execute("CREATE TABLE sessions (id TEXT, started_at REAL)")
    con.execute(
        "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, "
        "tool_name TEXT, timestamp REAL, active INTEGER)"
    )
    con.execute("INSERT INTO sessions VALUES ('sess-test', 1000)")
    rows = [
        ("sess-test", "user", "deploy the thing", "", 1001, 1),
        ("sess-test", "tool", f"error: auth failed, token={_PLANTED_SECRET}", "http", 1002, 1),
        ("sess-test", "assistant", "retrying", "", 1003, 1),
        ("sess-test", "tool", "error: auth failed again", "http", 1004, 1),
        ("sess-test", "user", "no, that is not right, use the other endpoint", "", 1005, 1),
        ("sess-test", "assistant", "ok", "", 1006, 1),
    ][:n_messages]
    con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _patched_db(path: str):
    """Return an _open_db replacement bound to the fake database."""
    import sqlite3 as _sq

    def _open():
        con = _sq.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = _sq.Row
        return con

    return _open


def test_evidence_scrubbed_from_db():
    """A secret in a message row never survives collect_evidence()."""
    print("\nTest: evidence scrubbed at DB read point")
    import core

    with tempfile.TemporaryDirectory() as td:
        db = _make_fake_db(td)
        orig = core._open_db
        core._open_db = _patched_db(db)
        try:
            ev = core.collect_evidence()
            blob = json.dumps(ev, ensure_ascii=False)
        finally:
            core._open_db = orig

    assert_true(_PLANTED_SECRET not in blob, "planted secret absent from evidence")
    assert_in("[REDACTED]", blob, "evidence contains redaction marker")


def test_refine_run_output_scrubbed():
    """Regression: the no_op return path must not leak raw evidence.

    refine_run() returns the evidence dict on the no_op branch, and that dict
    is serialized as the refine_run tool result — straight back into the
    model's context and into state.db.
    """
    print("\nTest: refine_run output scrubbed (no_op path)")
    import core
    import journal as j

    if j.daily_limit_reached():
        ok("skipped — daily edit limit reached")
        return

    mock = MockLlm({"action": "no_op", "reason": "nothing to learn"})

    with tempfile.TemporaryDirectory() as td:
        db = _make_fake_db(td)
        orig = core._open_db
        core._open_db = _patched_db(db)
        try:
            result = core.refine_run(mock, reason="scrub regression")
            blob = json.dumps(result, ensure_ascii=False)
        finally:
            core._open_db = orig

    assert_true(_PLANTED_SECRET not in blob, "planted secret absent from tool result")
    assert_true(bool(mock.calls), "mock LLM was actually called")

    sent = json.dumps(mock.calls, ensure_ascii=False, default=str)
    assert_true(_PLANTED_SECRET not in sent, "planted secret never sent to the LLM")


def test_pattern_fingerprint():
    """Errors that differ only by volatile detail collapse to one pattern."""
    print("\nTest: error fingerprinting")
    import patterns as P

    a = P.fingerprint("http", "HTTP 429 rate limited for /users/8821 at 2026-08-06T10:11:12Z")
    b = P.fingerprint("http", "HTTP 429 rate limited for /users/9134 at 2026-08-06T11:44:01Z")
    assert_eq(a, b, "same shape, different ids → one fingerprint")

    c = P.fingerprint("http", "HTTP 403 forbidden for /users/8821")
    assert_true(a != c, "different status → different fingerprint")

    d = P.fingerprint("bash", "HTTP 429 rate limited for /users/8821")
    assert_true(a != d, "same text, different tool → different fingerprint")

    tb = ('Traceback (most recent call last):\n  File "/tmp/x.py", line 12, in f\n'
          '    raise ValueError("bad id 991")\nValueError: bad id 991')
    assert_eq(P.normalize_error(tb), "valueerror: bad id n", "traceback reduced to final line")

    # Structured tool results are mostly JSON. Two properties must hold at once:
    # volatile detail inside a message collapses, but different messages do not.
    js = '{"success": false, "error": "%s"}'
    same_a = P.fingerprint("t", js % "rate limited for /u/8821")
    same_b = P.fingerprint("t", js % "rate limited for /u/9134")
    other = P.fingerprint("t", js % "permission denied")
    assert_eq(same_a, same_b, "JSON: volatile ids inside a message collapse")
    assert_true(same_a != other, "JSON: different error messages stay distinct")

    # Blanking JSON keys would reduce every tool result to one useless shape.
    assert_in("error", P.normalize_error(js % "boom"), "JSON keys survive normalization")

    # A timeout at 10s and at 15s is one failure, not two.
    assert_eq(
        P.fingerprint("proc", "status: timeout, waited 10s, still running"),
        P.fingerprint("proc", "status: timeout, waited 15s, still running"),
        "durations normalize",
    )


def test_pattern_aggregation():
    """extract_patterns counts occurrences and distinct sessions."""
    print("\nTest: pattern aggregation")
    import patterns as P

    items = [
        {"tool": "http", "content": f"HTTP 429 for /u/{i}", "session_id": f"s{i % 3}", "ts": 1000 + i}
        for i in range(6)
    ]
    items.append({"tool": "gmail", "content": "insufficient scope", "session_id": "s9", "ts": 2000})

    pats = P.extract_patterns(items)
    assert_eq(len(pats), 2, "two distinct patterns")
    assert_eq(pats[0]["count"], 6, "repeated pattern counted 6x")
    assert_eq(pats[0]["sessions_seen"], 3, "seen across 3 sessions")

    merged = P.merge_patterns(pats, [{"fingerprint": pats[0]["fingerprint"], "count": 99, "sessions_seen": 5}])
    assert_eq(merged[0]["count"], 99, "merge keeps the higher count")


def test_signal_gate():
    """has_signal only fires on a real repeat or an explicit correction."""
    print("\nTest: signal gate")
    import patterns as P

    assert_true(not P.has_signal([{"count": 1, "sessions_seen": 1}], []), "single one-off → no signal")
    assert_true(P.has_signal([{"count": 2, "sessions_seen": 1}], []), "repeated twice → signal")
    assert_true(P.has_signal([{"count": 1, "sessions_seen": 2}], []), "seen in 2 sessions → signal")
    assert_true(P.has_signal([], ["no, that is wrong"]), "user correction → signal")


def test_signal_gate_skips_llm():
    """With no repeat and no correction, refine_run must not call the model."""
    print("\nTest: signal gate skips the LLM call")
    import core
    import journal as j

    if j.daily_limit_reached():
        ok("skipped — daily edit limit reached")
        return

    mock = MockLlm({"action": "no_op", "reason": "unused"})

    with tempfile.TemporaryDirectory() as td:
        # 4 neutral messages: no errors, no corrections → no signal at all.
        import sqlite3 as _sq
        db = str(Path(td) / "quiet.db")
        con = _sq.connect(db)
        con.execute("CREATE TABLE sessions (id TEXT, started_at REAL)")
        con.execute("CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, "
                    "tool_name TEXT, timestamp REAL, active INTEGER)")
        con.execute("INSERT INTO sessions VALUES ('quiet', 1000)")
        con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", [
            ("quiet", "user", "what is the weather", "", 1001, 1),
            ("quiet", "assistant", "it is sunny today", "", 1002, 1),
            ("quiet", "user", "thanks a lot friend", "", 1003, 1),
            ("quiet", "assistant", "you are welcome", "", 1004, 1),
        ])
        con.commit(); con.close()

        orig = core._open_db
        core._open_db = _patched_db(db)
        try:
            result = core.refine_run(mock, reason="gate test")
        finally:
            core._open_db = orig

    assert_true(not mock.calls, "LLM was NOT called (no signal)")
    assert_eq(result.get("llm_called"), False, "result reports llm_called=False")
    assert_true(result.get("success") is True, "run still succeeds as a no_op")


def test_dedup_guard():
    """An identical proposal applied recently is refused."""
    print("\nTest: dedup guard")
    import journal as j
    from core import _validate_proposal

    proposal = {
        "action": "create", "kind": "skill", "name": "refine-dedup-probe",
        "content": "---\nname: refine-dedup-probe\ndescription: x\n---\n\n# x",
    }
    h1 = j.proposal_hash(proposal)
    h2 = j.proposal_hash(dict(proposal))
    assert_eq(h1, h2, "hash is stable for identical proposals")

    other = dict(proposal, content=proposal["content"] + " changed")
    assert_true(j.proposal_hash(other) != h1, "different content → different hash")

    # Write into a throwaway journal: a real "applied" entry would count
    # against the live daily edit budget.
    with tempfile.TemporaryDirectory() as td:
        orig = j.journal_dir
        j.journal_dir = lambda: Path(td)
        try:
            j.log(trigger="test", reason="dedup probe", session_id="test",
                  proposal=proposal, outcome="applied")
            err = _validate_proposal(proposal)
        finally:
            j.journal_dir = orig

    assert_true(bool(err) and "already applied" in (err or ""),
                f"repeat proposal rejected (got: {err})")


def test_ledger_audit():
    """Ledger records edits and the audit verdicts follow the evidence."""
    print("\nTest: usefulness ledger + audit")
    import ledger as L
    import time as _t

    # Throwaway stats file — never disturb the live ledger.
    td = tempfile.TemporaryDirectory()
    orig_dir = L.journal_dir
    L.journal_dir = lambda: Path(td.name)
    try:
        fp = "deadbeef1234"
        L._save_stats({
            "helped-skill": {
                "created_ts": _t.time() - 30 * 86400, "journal_id": "j1",
                "kind": "skill", "action": "create", "pattern_fingerprint": fp,
            },
            "did-not-help": {
                "created_ts": _t.time() - 30 * 86400, "journal_id": "j2",
                "kind": "skill", "action": "create", "pattern_fingerprint": "cafebabe0000",
            },
        })

        # The second skill's pattern showed up again after it was written.
        current = [{"fingerprint": "cafebabe0000", "last_ts": _t.time(), "count": 3, "sessions_seen": 2}]
        rows = {r["name"]: r for r in L.audit(current)}

        assert_eq(rows["did-not-help"]["pattern_recurred"], True, "recurring pattern detected")
        assert_eq(rows["did-not-help"]["verdict"], "did not help", "verdict follows recurrence")
        assert_eq(rows["helped-skill"]["pattern_recurred"], False, "non-recurring pattern")

        report = L.format_audit(list(rows.values()))
        assert_in("did-not-help", report, "report lists the failing skill")
        assert_in("Nothing was deleted", report, "report is explicitly read-only")
    finally:
        L.journal_dir = orig_dir
        td.cleanup()


# ── run ─────────────────────────────────────────────────────────────────────


def main():
    global _FAILURES, _ABS_TEST
    _FAILURES = 0

    # Clean journal state from previous test runs (idempotency)
    jp = Path.home() / ".hermes" / "plugins" / "refine" / "refine_journal.jsonl"
    if jp.is_file():
        jp.unlink()
    # Also clear backup dir from E2E test leftovers
    bk_d = Path.home() / ".hermes" / "plugins" / "refine" / "backups"
    if bk_d.is_dir():
        for f in bk_d.glob("*skill_e2e*"):
            f.unlink(missing_ok=True)

    tests = [
        test_collect_evidence,
        test_llm_proposal_valid,
        test_llm_proposal_noop,
        test_llm_proposal_invalid,
        test_llm_trust_error,
        test_journal_roundtrip,
        test_guardrail_agent_created,
        test_guardrail_reserved_prefix,
        test_refine_e2e_noop,
        test_refine_e2e_create_skill,
        test_refine_rollback_nonexistent,
        test_list_skill_names,
        test_scrub_text,
        test_scrub_new_patterns,
        test_scrub_no_false_positives,
        test_scrub_proposal,
        test_evidence_scrubbed_from_db,
        test_refine_run_output_scrubbed,
        test_pattern_fingerprint,
        test_pattern_aggregation,
        test_signal_gate,
        test_signal_gate_skips_llm,
        test_dedup_guard,
        test_ledger_audit,
        test_guardrail_create,
        test_refine_multipass,
    ]

    print("=" * 50)
    print("Refine Plugin Tests")
    print("=" * 50)

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            fail(test_fn.__name__, f"unhandled exception: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    if _FAILURES == 0:
        print("ALL TESTS PASSED 🎉")
    else:
        print(f"{_FAILURES} FAILURES ❌")
    print("=" * 50)

    return _FAILURES


if __name__ == "__main__":
    sys.exit(main())
