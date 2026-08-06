"""Hermetic stdlib regression suite for Refine Cycle.

Run from the repository root with ``python -m tests.run_tests``. The suite
installs a fake Hermes host before importing the plugin and stores every file
under a fresh TemporaryDirectory; it never reads or writes live Hermes state.
"""

import importlib.util
import inspect
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None


# Minimal agent.plugin_llm contract installed before plugin imports.
agent_module = types.ModuleType("agent")
plugin_module = types.ModuleType("agent.plugin_llm")


class PluginLlmTrustError(Exception):
    pass


class PluginLlmInput:
    pass


class PluginLlmTextInput(PluginLlmInput):
    def __init__(self, text):
        self.text = text


class MockResult:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = ""


class PluginLlm:
    def __init__(self, plugin_id=""):
        self.plugin_id = plugin_id

    def complete_structured(self, **kwargs):
        return MockResult({"action": "no_op", "reason": "stub"})


class MockLlm:
    def __init__(self, *responses):
        self.responses = list(responses) or [{"action": "no_op", "reason": "none"}]
        self.calls = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return MockResult(response)


plugin_module.PluginLlm = PluginLlm
plugin_module.PluginLlmInput = PluginLlmInput
plugin_module.PluginLlmTextInput = PluginLlmTextInput
plugin_module.PluginLlmStructuredResult = object
plugin_module.PluginLlmTrustError = PluginLlmTrustError
agent_module.plugin_llm = plugin_module
sys.modules.update({"agent": agent_module, "agent.plugin_llm": plugin_module})


class FakeHost:
    root = Path(".")
    skills = {}
    agent_created = set()
    actions = []
    stage_writes = False
    fail_next = ""
    memory_entries = []
    user_entries = []
    usage_counts = {}
    config = {}
    pending = {}
    pending_counter = 0

    @classmethod
    def reset(cls, root):
        cls.root = root
        cls.skills = {}
        cls.agent_created = set()
        cls.actions = []
        cls.stage_writes = False
        cls.fail_next = ""
        cls.memory_entries = []
        cls.user_entries = []
        cls.usage_counts = {}
        cls.pending = {}
        cls.pending_counter = 0
        cls.config = {"plugins": {"entries": {"refine": {
            "journal_dir": str(root / "journal"),
            "max_edits_per_day": 20,
            "max_edits_per_run": 1,
            "min_signal_required": False,
            "only_agent_created": True,
            "cross_session_enabled": True,
        }}}}
        cls.make_db()

    @classmethod
    def entry_config(cls):
        return cls.config["plugins"]["entries"]["refine"]

    @classmethod
    def make_db(cls, messages=None):
        path = cls.root / "state.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        now = time.time()
        rows = messages or [
            ("session", "user", "No, that is not right; use the other endpoint instead", "", now - 4, 1),
            ("session", "tool", "ERROR: request failed for /item/100", "http", now - 3, 1),
            ("session", "assistant", "Retrying", "", now - 2, 1),
            ("session", "tool", "ERROR: request failed for /item/200", "http", now - 1, 1),
        ]
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE sessions (id TEXT, started_at REAL)")
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, "
            "tool_name TEXT, timestamp REAL, active INTEGER)"
        )
        connection.execute("INSERT INTO sessions VALUES ('session', ?)", (now - 10,))
        connection.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
        connection.commit()
        connection.close()

    @classmethod
    def next_pending_id(cls, name):
        cls.pending_counter += 1
        return f"pending-{name}-{cls.pending_counter}"

    @classmethod
    def approve_pending(cls, subsystem, pending_id):
        record = cls.pending.pop((subsystem, pending_id))
        if subsystem == "skills":
            action = record["action"]
            name = record["name"]
            if action in ("create", "edit"):
                cls.add_skill(name, record.get("content") or "")
            elif action == "delete":
                cls.skills.pop(name, None)
                cls.agent_created.discard(name)
                shutil.rmtree(cls.root / "skills" / name, ignore_errors=True)
        else:
            target = record["target"]
            entries = cls.user_entries if target == "user" else cls.memory_entries
            entries.append(record["content"])
            filename = "USER.md" if target == "user" else "MEMORY.md"
            (cls.root / filename).write_text(
                "\n\n---\n\n".join(entries), encoding="utf-8"
            )

    @classmethod
    def reject_pending(cls, subsystem, pending_id):
        cls.pending.pop((subsystem, pending_id))

    @classmethod
    def add_skill(cls, name, content):
        cls.skills[name] = content
        cls.agent_created.add(name)
        directory = cls.root / "skills" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(content, encoding="utf-8")


def install_fake_host():
    tools = types.ModuleType("tools")
    tools.__path__ = []
    skills = types.ModuleType("tools.skills_tool")
    manager = types.ModuleType("tools.skill_manager_tool")
    usage = types.ModuleType("tools.skill_usage")
    memory = types.ModuleType("tools.memory_tool")
    approval = types.ModuleType("tools.write_approval")

    skills.skills_list = lambda: json.dumps({
        "skills": [{"name": name} for name in sorted(FakeHost.skills)]
    })

    def skill_view(name):
        if name not in FakeHost.skills:
            return json.dumps({"success": False, "error": "not found"})
        return json.dumps({
            "success": True,
            "skill_dir": str(FakeHost.root / "skills" / name),
            "content": FakeHost.skills[name],
        })

    def skill_manage(action, name, content=None, category=None):
        FakeHost.actions.append({
            "action": action, "name": name, "content": content, "category": category
        })
        if FakeHost.fail_next:
            error, FakeHost.fail_next = FakeHost.fail_next, ""
            return json.dumps({"success": False, "error": error})
        if FakeHost.stage_writes and action in ("create", "edit", "delete"):
            pending_id = FakeHost.next_pending_id(name)
            FakeHost.pending[("skills", pending_id)] = {
                "action": action,
                "name": name,
                "content": content,
                "category": category,
            }
            return json.dumps({
                "success": True, "staged": True, "pending_id": pending_id
            })
        if action == "create":
            if name in FakeHost.skills:
                return json.dumps({"success": False, "error": "exists"})
            FakeHost.add_skill(name, content or "")
        elif action == "edit":
            if name not in FakeHost.skills:
                return json.dumps({"success": False, "error": "not found"})
            FakeHost.add_skill(name, content or "")
        elif action == "delete":
            if name not in FakeHost.skills:
                return json.dumps({"success": False, "error": "not found"})
            del FakeHost.skills[name]
            FakeHost.agent_created.discard(name)
            shutil.rmtree(FakeHost.root / "skills" / name, ignore_errors=True)
        else:
            return json.dumps({"success": False, "error": f"unsupported {action}"})
        return json.dumps({"success": True, "message": f"{action} ok"})

    class MemoryStore:
        def __init__(self):
            self.memory_entries = FakeHost.memory_entries
            self.user_entries = FakeHost.user_entries

        def load_from_disk(self):
            return None

        def _entries_for(self, target):
            return FakeHost.user_entries if target == "user" else FakeHost.memory_entries

        def add(self, target, content):
            if FakeHost.stage_writes:
                pending_id = FakeHost.next_pending_id(target)
                FakeHost.pending[("memory", pending_id)] = {
                    "target": target,
                    "content": content,
                }
                return {"success": True, "staged": True, "pending_id": pending_id}
            self._entries_for(target).append(content)
            return {"success": True}

        def save_to_disk(self, target):
            filename = "USER.md" if target == "user" else "MEMORY.md"
            (FakeHost.root / filename).write_text(
                "\n\n---\n\n".join(self._entries_for(target)), encoding="utf-8"
            )

    skills.skill_view = skill_view
    manager.skill_manage = skill_manage
    usage.is_agent_created = lambda name: name in FakeHost.agent_created
    usage.get_usage_count = lambda name: FakeHost.usage_counts.get(name, 0)
    memory.MemoryStore = MemoryStore
    memory.get_memory_dir = lambda: str(FakeHost.root)
    approval.get_pending = lambda subsystem, pending_id: FakeHost.pending.get(
        (subsystem, pending_id)
    )
    tools.skills_tool, tools.skill_manager_tool = skills, manager
    tools.skill_usage, tools.memory_tool = usage, memory
    tools.write_approval = approval
    sys.modules.update({
        "tools": tools,
        "tools.skills_tool": skills,
        "tools.skill_manager_tool": manager,
        "tools.skill_usage": usage,
        "tools.memory_tool": memory,
        "tools.write_approval": approval,
    })

    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: str(FakeHost.root)
    cli = types.ModuleType("hermes_cli")
    cli.__path__ = []
    cli_config = types.ModuleType("hermes_cli.config")
    cli_config.load_config = lambda: FakeHost.config
    cli.config = cli_config
    sys.modules.update({
        "hermes_constants": constants,
        "hermes_cli": cli,
        "hermes_cli.config": cli_config,
    })


install_fake_host()
import config
import core
import journal
import ledger
import llm
import patterns


def load_plugin_init():
    spec = importlib.util.spec_from_file_location("refine_plugin_init", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


plugin_init = load_plugin_init()


def skill_content(name, body="# Guidance\n\nKeep this guidance."):
    return f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n"


def skill_proposal(name, body="# Guidance\n\nNew guidance."):
    return {
        "action": "create", "kind": "skill", "name": name,
        "content": skill_content(name, body), "reason": "Repeated failure",
        "evidence": ["request failed"], "pattern_fingerprint": "deadbeef1234",
    }


class RefineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        FakeHost.reset(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_proposal(self, proposal, **kwargs):
        with patch.object(core._llm, "propose", return_value=proposal):
            return core.refine_run(MockLlm(), **kwargs)

    def test_evidence_is_sandboxed_scrubbed_and_classified(self):
        secret = "ghp_" + "Z" * 36
        now = time.time()
        FakeHost.make_db([
            ("session", "user", "No, that is not right; use another endpoint instead", "", now - 3, 1),
            ("session", "tool", f'ERROR: denied, "api_key": "abc!{secret}"', "http", now - 2, 1),
            ("session", "assistant", "retry", "", now - 1, 1),
        ])
        result = core.collect_evidence()
        self.assertNotIn(secret, json.dumps(result))
        self.assertIn("[REDACTED]", json.dumps(result))
        self.assertEqual(result["error_count"], 1)
        self.assertTrue(str(config.state_db_path()).startswith(str(self.root)))
        self.assertTrue(str(journal.journal_path()).startswith(str(self.root)))

    def test_recursive_sanitation_covers_every_journal_field(self):
        entry_id = journal.log(
            trigger="manual", reason='password: "p@ss:w,rd!"', session_id="session",
            proposal={"action": "no_op", "reason": {"nested": ['"api_key":"aB!@#$[]"']}},
            outcome="no_op", error='{"token":"abc.DEF+/=!?"}',
        )
        raw = journal.journal_path().read_text(encoding="utf-8")
        for secret in ("p@ss:w,rd", "aB!@#$", "abc.DEF"):
            self.assertNotIn(secret, raw)
        self.assertIn("[REDACTED]", raw)
        self.assertEqual(journal.get_entry(entry_id)["outcome"], "no_op")

    def test_error_status_head_and_tail_classification(self):
        self.assertFalse(core._is_error_content('{"success":true,"exit_code":0,"error":null}'))
        self.assertTrue(core._is_error_content('{"success":false,"exit_code":2,"error":"boom"}'))
        self.assertTrue(core._is_error_content("ERROR: failed" + "x" * 10000))
        self.assertTrue(core._is_error_content("x" * 10000 + " timeout"))
        self.assertFalse(core._is_error_content("exit_code: 0\ncompleted normally"))

    def test_correction_requires_explicit_context(self):
        routine = (
            "Use the API for this task", "Do not forget the tests", "Try again tomorrow",
            "Use JSON instead of YAML for this new file", "Перероби документ у короткому форматі",
        )
        explicit = (
            "No, that is not right; use the other endpoint instead",
            "You used the old API; use the new API instead",
            "Це неправильно, перероби через інший endpoint",
        )
        self.assertTrue(all(not core._is_correction(item) for item in routine))
        self.assertTrue(all(core._is_correction(item) for item in explicit))

    def test_full_fingerprint_and_unbounded_audit_collection(self):
        fingerprint = patterns.fingerprint("http", "ERROR 42 for /item/123")
        self.assertEqual(len(fingerprint), 12)
        rendered = patterns.format_patterns([{
            "fingerprint": fingerprint, "count": 2, "sessions_seen": 1,
            "tool": "http", "sample": "ERROR",
        }])
        self.assertIn(f"fp:{fingerprint}", rendered)
        proposal = skill_proposal("fp-skill")
        proposal["pattern_fingerprint"] = fingerprint
        self.assertEqual(
            llm.propose(MockLlm(proposal), "evidence", [], [])["pattern_fingerprint"],
            fingerprint,
        )

        now = time.time()
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo",
        ]
        FakeHost.make_db([
            ("session", "tool", f"ERROR: unique failure {word}", "tool", now - index, 1)
            for index, word in enumerate(words)
        ])
        found = core.collect_cross_session_patterns(
            since_ts=now - 100, max_rows=None, max_sessions=None
        )
        self.assertEqual(len(found), 11)

    def test_skill_patch_gets_current_complete_content(self):
        name = "existing-skill"
        current = skill_content(name, "# Existing\n\nImportant old guidance.")
        replacement = skill_content(name, "# Existing\n\nImportant old guidance.\n\nNew fix.")
        FakeHost.add_skill(name, current)
        initial = {
            "action": "patch", "kind": "skill", "name": name,
            "content": "New fix only", "reason": "failure", "evidence": [],
        }
        model = MockLlm(initial, dict(initial, content=replacement))
        result = llm.propose(
            model, "evidence", [name], [], skill_content_loader=journal.read_skill_content
        )
        self.assertEqual(result["content"], replacement)
        self.assertIn(current, model.calls[1]["input"][0].text)

    def test_patch_maps_to_edit_and_invalid_content_never_applies(self):
        name = "patch-map"
        FakeHost.add_skill(name, skill_content(name))
        result = core._apply_skill({
            "action": "patch", "kind": "skill", "name": name,
            "content": skill_content(name, "# Updated"), "category": "",
        })
        self.assertTrue(result["success"])
        self.assertEqual(FakeHost.actions[-1]["action"], "edit")
        self.assertIn("non-empty", core._validate_proposal({
            "action": "patch", "kind": "skill", "name": "x", "content": ""
        }))
        self.assertIn("frontmatter", core._validate_proposal({
            "action": "create", "kind": "skill", "name": "x", "content": "body"
        }))

    def test_backup_and_prepare_failures_abort_before_mutation(self):
        name = "backup-fail"
        original = skill_content(name)
        FakeHost.add_skill(name, original)
        patch_proposal = {
            "action": "patch", "kind": "skill", "name": name,
            "content": skill_content(name, "# Changed"), "reason": "why", "evidence": [],
        }
        with patch.object(core._llm, "propose", return_value=patch_proposal), patch.object(
            journal, "backup_skill", return_value=None
        ):
            result = core.refine_run(MockLlm())
        self.assertFalse(result["success"])
        self.assertEqual(FakeHost.skills[name], original)
        self.assertFalse(FakeHost.actions)

        with patch.object(core._llm, "propose", return_value=skill_proposal("prepare-fail")), patch.object(
            journal, "prepare", side_effect=OSError("disk full")
        ):
            result = core.refine_run(MockLlm(), reason='token="unsafe!value"')
        self.assertFalse(result["success"])
        self.assertNotIn("prepare-fail", FakeHost.skills)
        self.assertNotIn("unsafe!value", json.dumps(result))

    def test_journal_append_preserves_history_and_recovers_after_corrupt_tail(self):
        first = journal.log(
            trigger="test", reason="first", session_id="s",
            proposal={"action": "no_op"}, outcome="no_op",
        )
        path = journal.journal_path()
        with path.open("ab") as handle:
            handle.write(b'{"id":"broken"')
            handle.flush()
        second = journal.log(
            trigger="test", reason="second", session_id="s",
            proposal={"action": "no_op"}, outcome="no_op",
        )
        raw = path.read_bytes()
        self.assertIn(b'{"id":"broken"\n', raw)
        self.assertEqual(journal.get_entry(first)["reason"], "first")
        self.assertEqual(journal.get_entry(second)["reason"], "second")
        self.assertNotIn("os.replace", inspect.getsource(journal._append_entry))

    def test_finalize_failure_keeps_prepared_recovery(self):
        original_finalize = journal.finalize
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("finalize disk error")
            return original_finalize(*args, **kwargs)

        with patch.object(core._llm, "propose", return_value=skill_proposal("finalize-fail")), patch.object(
            journal, "finalize", side_effect=fail_once
        ):
            result = core.refine_run(MockLlm())
            self.assertFalse(result["success"])
            self.assertEqual(journal.get_entry(result["journal_id"])["outcome"], "prepared")
            rollback = core.refine_rollback(result["journal_id"])
        self.assertTrue(rollback["success"])
        self.assertNotIn("finalize-fail", FakeHost.skills)

    def test_apply_failures_propagate_without_rollback_id(self):
        FakeHost.fail_next = 'failed with "password":"bad!secret"'
        result = self.run_proposal(
            skill_proposal("apply-fail"), reason='manual token="reason!secret"'
        )
        self.assertFalse(result["success"])
        self.assertNotIn("journal_id", result)
        self.assertEqual(journal.get_entry(result["record_id"])["outcome"], "error")
        raw = journal.journal_path().read_text(encoding="utf-8")
        self.assertNotIn("bad!secret", raw)
        self.assertNotIn("reason!secret", raw)

        with patch.object(core._llm, "propose", return_value=skill_proposal("bad-stage")), patch.object(
            core, "_apply_skill", return_value={"success": False, "staged": True, "error": "denied"}
        ):
            staged = core.refine_run(MockLlm())
        self.assertEqual(journal.get_entry(staged["record_id"])["outcome"], "error")
        self.assertNotIn("bad-stage", ledger.load_stats())

    def test_create_rollback_deletes_only_unchanged_skill(self):
        result = self.run_proposal(skill_proposal("created-skill"))
        self.assertTrue(result["reversible"])
        self.assertTrue(core.refine_rollback(result["journal_id"])["success"])
        self.assertNotIn("created-skill", FakeHost.skills)

        changed = self.run_proposal(skill_proposal("changed-after-create"))
        later = skill_content("changed-after-create", "# User change")
        FakeHost.add_skill("changed-after-create", later)
        conflict = core.refine_rollback(changed["journal_id"])
        self.assertFalse(conflict["success"])
        self.assertEqual(FakeHost.skills["changed-after-create"], later)

    def test_patch_rollback_restores_backup_without_overwriting_later_edit(self):
        name = "patch-rollback"
        old = skill_content(name, "# Old\n\nPreserve me.")
        new = skill_content(name, "# Old\n\nPreserve me.\n\nFixed.")
        proposal = {
            "action": "patch", "kind": "skill", "name": name,
            "content": new, "reason": "failure", "evidence": [],
        }
        FakeHost.add_skill(name, old)
        result = self.run_proposal(proposal)
        self.assertTrue(Path(result["backup_path"]).is_file())
        self.assertTrue(core.refine_rollback(result["journal_id"])["success"])
        self.assertEqual(FakeHost.skills[name], old)

        FakeHost.add_skill(name, old)
        changed = self.run_proposal(proposal)
        later = skill_content(name, "# Manual later change")
        FakeHost.add_skill(name, later)
        conflict = core.refine_rollback(changed["journal_id"])
        self.assertFalse(conflict["success"])
        self.assertEqual(FakeHost.skills[name], later)

    def test_memory_rollback_removes_exact_append_only(self):
        FakeHost.memory_entries[:] = ["before"]
        proposal = {
            "action": "create", "kind": "memory", "name": "lesson",
            "content": "exact appended lesson", "reason": "why", "evidence": [],
        }
        result = self.run_proposal(proposal)
        FakeHost.memory_entries.append("unrelated later entry")
        self.assertTrue(core.refine_rollback(result["journal_id"])["success"])
        self.assertEqual(FakeHost.memory_entries, ["before", "unrelated later entry"])

        result = self.run_proposal(dict(proposal, content="second lesson"))
        FakeHost.memory_entries[0] = "changed before"
        conflict = core.refine_rollback(result["journal_id"])
        self.assertFalse(conflict["success"])
        self.assertIn("second lesson", FakeHost.memory_entries)

    def test_pending_consumes_budget_and_is_reported_as_pending(self):
        FakeHost.entry_config()["max_edits_per_day"] = 1
        FakeHost.stage_writes = True
        first = self.run_proposal(skill_proposal("pending-skill"))
        second = self.run_proposal(skill_proposal("pending-other"))
        self.assertTrue(first["success"])
        self.assertFalse(first["reversible"])
        self.assertFalse(second["success"])
        self.assertEqual(journal.count_today_applied(), 1)
        self.assertEqual(ledger.load_stats()["pending-skill"]["outcome"], "pending_approval")
        self.assertEqual(ledger.audit([])[0]["verdict"], "pending approval")

    def test_concurrent_runs_serialize_and_recheck_budget(self):
        FakeHost.entry_config()["max_edits_per_day"] = 1
        barrier = threading.Barrier(3)
        results = []

        def worker(name):
            barrier.wait()
            results.append(core.refine_run(MockLlm(skill_proposal(name))))

        threads = [threading.Thread(target=worker, args=(f"concurrent-{index}",)) for index in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(sum(bool(result["success"]) for result in results), 1)
        self.assertEqual(len(FakeHost.skills), 1)

    def test_reason_and_multipass_context_reach_model(self):
        FakeHost.entry_config()["max_edits_per_run"] = 2
        model = MockLlm(skill_proposal("first-pass"), {"action": "no_op", "reason": "done"})
        result = core.refine_run(model, reason="focus on command parsing")
        self.assertTrue(result["success"])
        self.assertIn("focus on command parsing", model.calls[0]["input"][0].text)
        self.assertIn("Already completed or reserved", model.calls[1]["input"][0].text)
        self.assertIn("first-pass", model.calls[1]["input"][0].text)

    def test_command_parsing_is_exact_and_rollback_is_real(self):
        with patch.object(plugin_init.core, "refine_audit", return_value={"report": "AUDIT"}), patch.object(
            plugin_init.core, "refine_run", return_value={
                "success": True, "message": "OK", "journal_id": "abcdef123456", "reversible": False
            }
        ) as run, patch.object(
            plugin_init.core, "refine_rollback", return_value={"success": True, "message": "rolled"}
        ) as rollback:
            self.assertEqual(plugin_init._handle_refine_command("audit"), "AUDIT")
            ordinary = plugin_init._handle_refine_command("audit logging failures")
            self.assertEqual(run.call_args.kwargs["reason"], "audit logging failures")
            self.assertNotIn("rollback:", ordinary)
            plugin_init._handle_refine_command("rollback abcdef123456")
            rollback.assert_called_once_with("abcdef123456")
            plugin_init._handle_refine_command("rollback not-an-id")
            self.assertEqual(run.call_args.kwargs["reason"], "rollback not-an-id")

    def test_ledger_uses_only_supported_post_edit_evidence(self):
        created = time.time() - 30 * 86400
        base = {
            "created_ts": created, "journal_id": "abcdef123456", "kind": "skill",
            "action": "create", "pattern_fingerprint": "deadbeef1234",
        }
        FakeHost.usage_counts["old-skill"] = 4
        ledger._save_stats({"old-skill": base})
        row = ledger.audit([])[0]
        self.assertEqual(row["usage_scope"], "all_time")
        self.assertEqual(row["verdict"], "unclear")

        usage = sys.modules["tools.skill_usage"]
        original = usage.get_usage_count
        usage.get_usage_count = lambda name, since_ts=None: 2
        try:
            row = ledger.audit([])[0]
        finally:
            usage.get_usage_count = original
        self.assertEqual(row["usage_scope"], "since_exact")
        self.assertEqual(row["verdict"], "working")

    def test_audit_requests_full_post_edit_period(self):
        created = time.time() - 100
        ledger._save_stats({"audit-skill": {
            "created_ts": created, "journal_id": "abcdef123456", "kind": "skill",
            "action": "create", "pattern_fingerprint": "deadbeef1234",
        }})
        with patch.object(core, "collect_cross_session_patterns", return_value=[]) as collect:
            core.refine_audit()
        collect.assert_called_once_with(since_ts=created, max_rows=None, max_sessions=None)
    def test_sanitization_is_idempotent_and_all_prompt_inputs_are_scrubbed(self):
        marker_text = 'token=[REDACTED] and password: "[REDACTED]"'
        self.assertEqual(core.scrub_text(marker_text), marker_text)
        secrets = [
            "reason-secret-123!", "evidence-secret-123!", "name-secret-123!",
            "correction-secret-123!", "pattern-secret-123!", "memory-secret-123!",
        ]
        model = MockLlm({"action": "no_op", "reason": "done"})
        llm.propose(
            model,
            'api_key="evidence-secret-123!"',
            ['token="name-secret-123!"'],
            ['password="memory-secret-123!"'],
            error_patterns=[{
                "fingerprint": "deadbeef1234", "count": 2, "sessions_seen": 1,
                "tool": "tool", "sample": 'secret="pattern-secret-123!"',
            }],
            user_corrections=['password="correction-secret-123!"'],
            unused_skills=['token="name-secret-123!"'],
            run_context='token="reason-secret-123!"',
        )
        sent = json.dumps(model.calls[0], default=lambda value: getattr(value, "text", str(value)))
        for secret in secrets:
            self.assertNotIn(secret, sent)
        self.assertIn("[REDACTED]", sent)

    def test_sensitive_current_skill_aborts_before_complete_patch_request(self):
        name = "sensitive-current"
        current = skill_content(name, '# Guidance\n\napi_key="current-secret-123!"')
        FakeHost.add_skill(name, current)
        initial = {
            "action": "patch", "kind": "skill", "name": name,
            "reason": "failure", "evidence": [],
        }
        model = MockLlm(initial)
        result = llm.propose(
            model, "evidence", [name], [], skill_content_loader=journal.read_skill_content
        )
        self.assertEqual(result["action"], "no_op")
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("current-secret-123", model.calls[0]["input"][0].text)

    def test_redacted_create_patch_and_memory_match_journal_and_rollback(self):
        create = skill_proposal(
            "redacted-create", '# Guidance\n\napi_key="create-secret-123!"'
        )
        created = self.run_proposal(create)
        created_entry = journal.get_entry(created["journal_id"])
        self.assertNotIn("create-secret-123", FakeHost.skills["redacted-create"])
        self.assertEqual(created_entry["proposal"]["content"], FakeHost.skills["redacted-create"])
        self.assertTrue(core.refine_rollback(created["journal_id"])["success"])

        name = "redacted-patch"
        original = skill_content(name, "# Guidance\n\nOriginal.")
        FakeHost.add_skill(name, original)
        patched = self.run_proposal({
            "action": "patch", "kind": "skill", "name": name,
            "content": skill_content(name, '# Guidance\n\ntoken="patch-secret-123!"'),
            "reason": "why", "evidence": [],
        })
        patched_entry = journal.get_entry(patched["journal_id"])
        self.assertEqual(patched_entry["proposal"]["content"], FakeHost.skills[name])
        self.assertNotIn("patch-secret-123", FakeHost.skills[name])
        self.assertTrue(core.refine_rollback(patched["journal_id"])["success"])
        self.assertEqual(FakeHost.skills[name], original)

        memory_result = self.run_proposal({
            "action": "create", "kind": "memory", "name": "redacted-memory",
            "content": 'password="memory-secret-123!"', "reason": "why", "evidence": [],
        })
        memory_entry = journal.get_entry(memory_result["journal_id"])
        self.assertEqual(memory_entry["proposal"]["content"], FakeHost.memory_entries[-1])
        self.assertNotIn("memory-secret-123", FakeHost.memory_entries[-1])
        self.assertTrue(core.refine_rollback(memory_result["journal_id"])["success"])

    def test_new_malformed_lock_is_not_deleted_until_mtime_is_stale(self):
        lock_path = journal.ensure_dirs() / journal._LOCK_FILE_NAME
        lock_path.write_bytes(b"")
        modified = 1000.0
        os_module = __import__("os")
        os_module.utime(lock_path, (modified, modified))
        with patch.object(journal.time, "time", return_value=modified + 299):
            journal._try_clear_stale_lock(lock_path)
        self.assertTrue(lock_path.exists())
        with patch.object(journal.time, "time", return_value=modified + 301):
            journal._try_clear_stale_lock(lock_path)
        self.assertFalse(lock_path.exists())

    def test_forward_approval_reconciles_approved_rejected_and_memory(self):
        FakeHost.stage_writes = True
        approved = self.run_proposal(skill_proposal("approved-skill"))
        approved_entry = journal.get_entry(approved["journal_id"])
        pending_id = approved_entry["pending_id"]
        self.assertEqual(approved_entry["recovery"]["pending_id"], pending_id)
        self.assertEqual(ledger.load_stats()["approved-skill"]["pending_id"], pending_id)
        FakeHost.approve_pending("skills", pending_id)
        core.refine_audit()
        self.assertEqual(journal.get_entry(approved["journal_id"])["outcome"], "applied")
        self.assertEqual(ledger.load_stats()["approved-skill"]["outcome"], "applied")
        self.assertTrue(journal.is_reversible(journal.get_entry(approved["journal_id"])))

        rejected = self.run_proposal(skill_proposal("rejected-skill"))
        rejected_id = journal.get_entry(rejected["journal_id"])["pending_id"]
        FakeHost.reject_pending("skills", rejected_id)
        core.refine_audit()
        self.assertEqual(journal.get_entry(rejected["journal_id"])["outcome"], "rejected")
        self.assertEqual(ledger.load_stats()["rejected-skill"]["outcome"], "rejected")

        memory_result = self.run_proposal({
            "action": "create", "kind": "memory", "name": "pending-memory",
            "content": "exact pending memory", "reason": "why", "evidence": [],
        })
        memory_pending = journal.get_entry(memory_result["journal_id"])["pending_id"]
        FakeHost.approve_pending("memory", memory_pending)
        core.refine_audit()
        self.assertEqual(journal.get_entry(memory_result["journal_id"])["outcome"], "applied")
        self.assertEqual(FakeHost.memory_entries, ["exact pending memory"])

    def test_matching_target_does_not_bypass_unresolved_approval(self):
        name = "already-matching"
        content = skill_content(name, "# Guidance\n\nAlready current.")
        FakeHost.add_skill(name, content)
        FakeHost.stage_writes = True
        pending = self.run_proposal({
            "action": "patch", "kind": "skill", "name": name,
            "content": content, "reason": "verify approval ordering", "evidence": [],
        })
        entry = journal.get_entry(pending["journal_id"])
        core.refine_audit()
        self.assertEqual(
            journal.get_entry(pending["journal_id"])["outcome"], "pending_approval"
        )
        FakeHost.approve_pending("skills", entry["pending_id"])
        core.refine_audit()
        self.assertEqual(journal.get_entry(pending["journal_id"])["outcome"], "applied")

        rollback = core.refine_rollback(pending["journal_id"])
        rollback_entry = journal.get_entry(pending["journal_id"])
        self.assertTrue(rollback["staged"])
        core.refine_audit()
        self.assertEqual(
            journal.get_entry(pending["journal_id"])["outcome"], "pending_rollback"
        )
        FakeHost.approve_pending("skills", rollback_entry["pending_id"])
        core.refine_audit()
        self.assertEqual(journal.get_entry(pending["journal_id"])["outcome"], "rolled_back")

    def test_removed_pending_record_waits_when_target_state_is_unavailable(self):
        FakeHost.stage_writes = True
        result = self.run_proposal(skill_proposal("unknown-target"))
        entry = journal.get_entry(result["journal_id"])
        FakeHost.reject_pending("skills", entry["pending_id"])
        skills_module = sys.modules["tools.skills_tool"]
        with patch.object(skills_module, "skill_view", side_effect=OSError("temporarily unavailable")):
            core.refine_audit()
        self.assertEqual(
            journal.get_entry(result["journal_id"])["outcome"], "pending_approval"
        )
        core.refine_audit()
        self.assertEqual(journal.get_entry(result["journal_id"])["outcome"], "rejected")

    def test_staged_rollback_waits_for_target_proof_and_reconciles(self):
        applied = self.run_proposal(skill_proposal("rollback-approved"))
        FakeHost.stage_writes = True
        pending = core.refine_rollback(applied["journal_id"])
        entry = journal.get_entry(applied["journal_id"])
        self.assertTrue(pending["staged"])
        self.assertEqual(entry["outcome"], "pending_rollback")
        self.assertEqual(
            ledger.load_stats()["rollback-approved"]["pending_id"], entry["pending_id"]
        )
        self.assertIn("rollback-approved", FakeHost.skills)
        FakeHost.approve_pending("skills", entry["pending_id"])
        completed = core.refine_rollback(applied["journal_id"])
        self.assertTrue(completed["success"])
        self.assertEqual(journal.get_entry(applied["journal_id"])["outcome"], "rolled_back")

        FakeHost.stage_writes = False
        other = self.run_proposal(skill_proposal("rollback-rejected"))
        FakeHost.stage_writes = True
        core.refine_rollback(other["journal_id"])
        other_entry = journal.get_entry(other["journal_id"])
        FakeHost.reject_pending("skills", other_entry["pending_id"])
        core.refine_audit()
        restored = journal.get_entry(other["journal_id"])
        self.assertEqual(restored["outcome"], "applied")
        self.assertTrue(journal.is_reversible(restored))
        self.assertIn("rollback-rejected", FakeHost.skills)

    def test_rollback_finalization_failure_is_reconciled_from_target_state(self):
        applied = self.run_proposal(skill_proposal("rollback-finalize-fail"))
        original_finalize = journal.finalize
        failed = False

        def fail_rolled_back(entry_id, outcome, **kwargs):
            nonlocal failed
            if outcome == "rolled_back" and not failed:
                failed = True
                raise OSError("finalization failed")
            return original_finalize(entry_id, outcome, **kwargs)

        with patch.object(journal, "finalize", side_effect=fail_rolled_back):
            result = core.refine_rollback(applied["journal_id"])
        self.assertFalse(result["success"])
        self.assertNotIn("rollback-finalize-fail", FakeHost.skills)
        self.assertEqual(
            journal.get_entry(applied["journal_id"])["outcome"], "rollback_prepared"
        )
        retried = core.refine_rollback(applied["journal_id"])
        self.assertTrue(retried["success"])
        self.assertEqual(journal.get_entry(applied["journal_id"])["outcome"], "rolled_back")

    def test_partial_success_preserves_all_recovery_ids_and_command_warns(self):
        FakeHost.entry_config()["max_edits_per_run"] = 2
        model = MockLlm(
            skill_proposal("partial-first"),
            {
                "action": "create", "kind": "skill", "name": "partial-bad",
                "content": "not a skill", "reason": "later failure", "evidence": [],
            },
        )
        result = core.refine_run(model)
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "partial_success")
        self.assertEqual(len(result["recoveries"]), 1)
        recovery = result["recoveries"][0]
        self.assertIn("rollback_command", recovery)
        with patch.object(plugin_init.core, "refine_run", return_value=result):
            output = plugin_init._handle_refine_command("")
        self.assertIn("⚠️", output)
        self.assertIn(recovery["journal_id"], output)
        self.assertIn(recovery["rollback_command"], output)

    def test_patch_selection_without_content_reaches_complete_replacement(self):
        name = "contentless-patch"
        current = skill_content(name, "# Existing\n\nKeep.")
        replacement = skill_content(name, "# Existing\n\nKeep.\n\nFix.")
        FakeHost.add_skill(name, current)
        model = MockLlm(
            {"action": "patch", "kind": "skill", "name": name, "reason": "why"},
            {"action": "patch", "kind": "skill", "name": name, "content": replacement, "reason": "why"},
        )
        result = llm.propose(
            model, "evidence", [name], [], skill_content_loader=journal.read_skill_content
        )
        self.assertEqual(result["content"], replacement)
        self.assertEqual(len(model.calls), 2)

    def test_oversized_current_skill_is_not_truncated_or_sent(self):
        name = "oversized-skill"
        current = skill_content(name, "x" * llm.MAX_CONTENT_CHARS)
        FakeHost.add_skill(name, current)
        model = MockLlm({
            "action": "patch", "kind": "skill", "name": name, "reason": "why"
        })
        result = llm.propose(
            model, "evidence", [name], [], skill_content_loader=journal.read_skill_content
        )
        self.assertEqual(result["action"], "no_op")
        self.assertIn(str(llm.MAX_CONTENT_CHARS), result["reason"])
        self.assertEqual(len(model.calls), 1)

    def test_patch_retry_preserves_or_replaces_valid_evidence_metadata(self):
        name = "metadata-patch"
        current = skill_content(name, "# Existing\n\nKeep.")
        replacement = skill_content(name, "# Existing\n\nKeep.\n\nFix.")
        FakeHost.add_skill(name, current)
        initial = {
            "action": "patch", "kind": "skill", "name": name, "reason": "initial",
            "evidence": ["initial evidence"], "pattern_fingerprint": "deadbeef1234",
        }
        preserved = llm.propose(
            MockLlm(initial, {
                "action": "patch", "kind": "skill", "name": name,
                "content": replacement, "reason": "replacement",
            }),
            "evidence", [name], [], skill_content_loader=journal.read_skill_content,
        )
        self.assertEqual(preserved["evidence"], ["initial evidence"])
        self.assertEqual(preserved["pattern_fingerprint"], "deadbeef1234")

        replaced = llm.propose(
            MockLlm(initial, {
                "action": "patch", "kind": "skill", "name": name,
                "content": replacement, "reason": "replacement",
                "evidence": ["replacement evidence"],
                "pattern_fingerprint": "cafebabefeed",
            }),
            "evidence", [name], [], skill_content_loader=journal.read_skill_content,
        )
        self.assertEqual(replaced["evidence"], ["replacement evidence"])
        self.assertEqual(replaced["pattern_fingerprint"], "cafebabefeed")

    def test_full_history_patterns_stream_without_fetchall(self):
        self.assertNotIn("fetchall", inspect.getsource(core.collect_cross_session_patterns))
        now = time.time()
        labels = [chr(ord("a") + index) * 3 for index in range(20)]
        rows = [
            (f"session-{index}", "tool", f"ERROR: streamed failure {labels[index % 20]}",
             "stream", now - index, 1)
            for index in range(1000)
        ]
        FakeHost.make_db(rows)
        found = core.collect_cross_session_patterns(
            since_ts=now - 2000, max_rows=None, max_sessions=None
        )
        self.assertEqual(sum(item["count"] for item in found), 1000)
        self.assertLessEqual(len(found), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
