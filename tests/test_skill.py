from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from duckduckcode.core.agent import Agent
from duckduckcode.core.context import ContextManager
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from duckduckcode.core.skill import SkillManager
from duckduckcode.permissions import PermissionChecker, PermissionDecision
from duckduckcode.tools.tool import ToolCall, ToolManager, create_load_skill_tool


class SkillTest(unittest.TestCase):
    def test_discovers_project_skill_over_user_skill(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            user = Path(home) / ".duckduckcode" / "skills"
            project = workspace / ".duckduckcode" / "skills"
            user.mkdir(parents=True)
            project.mkdir(parents=True)
            (user / "lazy.md").write_text(
                "---\nname: same\ndescription: user copy\n---\nUser body\n",
                encoding="utf-8",
            )
            (project / "same").mkdir()
            (project / "same" / "SKILL.md").write_text(
                "---\nname: same\ndescription: project copy\nmode: inline\nextra: kept\n---\nProject body\n",
                encoding="utf-8",
            )

            manager = SkillManager(workspace, home=home)
            skills, warning = manager.refresh()

            self.assertIsNone(warning)
            self.assertEqual([skill.scope for skill in skills], ["project"])
            self.assertEqual(skills[0].metadata["extra"], "kept")

    def test_rejects_duplicates_and_conflicting_commands(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            root = Path(wd) / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            for filename in ("one.md", "two.md"):
                (root / filename).write_text(
                    "---\nname: same\ndescription: duplicate\n---\nBody\n",
                    encoding="utf-8",
                )
            (root / "help.md").write_text(
                "---\nname: help\ndescription: conflict\n---\nBody\n",
                encoding="utf-8",
            )

            manager = SkillManager(Path(wd), home=home, builtin_commands={"/help"})
            skills, warning = manager.refresh()

            self.assertEqual(skills, [])
            self.assertIn("duplicate skill name 'same'", warning or "")
            self.assertIn("conflicts", warning or "")

    def test_validates_frontmatter_names_links_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            root = Path(wd) / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            valid = "---\r\nname: unicode\r\ndescription: 中文说明\r\n---\r\n正文\r\n"
            (root / "unicode.md").write_text(valid, encoding="utf-8")
            invalid = {
                "duplicate.md": "---\nname: duplicate\nname: again\ndescription: x\n---\nBody\n",
                "empty.md": "---\nname: empty\ndescription: x\n---\n   \n",
                "mode.md": "---\nname: mode\ndescription: x\nmode: other\n---\nBody\n",
                "name.md": "---\nname: two--dashes\ndescription: x\n---\nBody\n",
                "marker.md": "---\nname: marker\ndescription: x\n---oops\nBody\n",
            }
            for filename, content in invalid.items():
                (root / filename).write_text(content, encoding="utf-8")
            target = root / "target.md"
            target.write_text(
                "---\nname: target\ndescription: x\n---\nBody\n", encoding="utf-8"
            )
            (root / "linked.md").symlink_to(target)
            prefix = b"---\nname: limit\ndescription: x\n---\n"
            (root / "limit.md").write_bytes(prefix + b"x" * (256 * 1024 - len(prefix)))
            (root / "large.md").write_bytes(
                b"---\nname: large\ndescription: x\n---\n" + b"x" * (256 * 1024)
            )

            skills, warning = SkillManager(Path(wd), home=home).refresh()

            self.assertEqual(
                [skill.name for skill in skills], ["limit", "target", "unicode"]
            )
            self.assertEqual(skills[-1].description, "中文说明")
            self.assertIn("duplicate YAML field", warning or "")
            self.assertIn("body cannot be empty", warning or "")
            self.assertIn("must not be a symlink", warning or "")
            self.assertIn("exceeds", warning or "")

    def test_catalog_does_not_disclose_body_path_or_mode(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            (root / "demo.md").write_text(
                "---\nname: demo\ndescription: Catalog text\nmode: fork\n---\nSECRET BODY\n",
                encoding="utf-8",
            )
            manager = SkillManager(workspace, home=workspace / "home")
            manager.refresh()

            catalog = manager.catalog_block()

            self.assertIn("demo: Catalog text", catalog)
            self.assertNotIn("SECRET BODY", catalog)
            self.assertNotIn("mode", catalog)
            self.assertNotIn(str(root), catalog)

    def test_inline_load_updates_current_turn_context_only(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills" / "demo"
            root.mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill\n---\nUse demo rules.\n",
                encoding="utf-8",
            )
            manager = SkillManager(workspace, home=workspace / "home")
            manager.refresh()
            context = ContextManager(system_prompt="system")
            tools = ToolManager()
            tools.register(create_load_skill_tool(manager.load))

            result = tools.execute(
                ToolCall("call_1", "LoadSkill", {"name": "demo", "task": "do it"})
            )
            context.set_active_skills(manager.active_block())

            self.assertFalse(result.is_error)
            self.assertIn("Use demo rules.", context.model_messages()[1].content)
            manager.clear_active()
            context.set_active_skills(manager.active_block())
            self.assertNotIn(
                "Use demo rules.",
                "\n".join(message.content for message in context.model_messages()),
            )

    def test_agent_selected_skill_loads_before_model_request(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.seen = ""

            def stream(self, messages, tools=None, reasoning=None):
                self.seen = "\n".join(message.content for message in messages)
                yield DoneEvent()

        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            (root / "demo.md").write_text(
                "---\nname: demo\ndescription: Demo skill\n---\nSkill body marker.\n",
                encoding="utf-8",
            )
            manager = SkillManager(workspace, home=workspace / "home")
            context = ContextManager(system_prompt="system")
            tools = ToolManager()
            tools.register(create_load_skill_tool(manager.load))
            client = Client()

            events = list(
                Agent(client, context, tools, skill_manager=manager).stream(
                    "hello", selected_skills=["demo"]
                )
            )

            selected_result = next(
                event
                for event in events
                if isinstance(event, ToolResultEvent) and event.name == "LoadSkill"
            )
            self.assertTrue(selected_result.call_id.startswith("selected_skill_"))
            self.assertEqual(selected_result.content, "Loaded skill 'demo'.")
            self.assertIn("Skill body marker.", client.seen)
            self.assertEqual(context.active_skills, "")

    def test_selected_skill_call_ids_are_unique_across_turns(self) -> None:
        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                yield DoneEvent()

        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            (root / "demo.md").write_text(
                "---\nname: demo\ndescription: Demo\n---\nBody\n", encoding="utf-8"
            )
            manager = SkillManager(workspace, home=workspace / "home")
            tools = ToolManager()
            tools.register(create_load_skill_tool(manager.load))
            agent = Agent(Client(), ContextManager(), tools, skill_manager=manager)

            first = list(agent.stream("one", selected_skills=["demo"]))
            second = list(agent.stream("two", selected_skills=["demo"]))

            first_call = next(
                event.tool_call
                for event in first
                if isinstance(event, ToolCallEvent)
                and event.tool_call.name == "LoadSkill"
            )
            second_call = next(
                event.tool_call
                for event in second
                if isinstance(event, ToolCallEvent)
                and event.tool_call.name == "LoadSkill"
            )
            self.assertTrue(first_call.call_id.startswith("selected_skill_"))
            self.assertTrue(second_call.call_id.startswith("selected_skill_"))
            self.assertNotEqual(first_call.call_id, second_call.call_id)
            self.assertEqual(first_call.arguments, {"name": "demo", "task": "one"})
            self.assertEqual(second_call.arguments, {"name": "demo", "task": "two"})

    def test_fork_skill_isolated_and_forwards_prefixed_tool_events(self) -> None:
        class AskPolicy:
            permission_mode = "ask_for_approval"

            def check(self, tool_call, *, tool=None):
                return PermissionDecision("ask", "approve", tool_call.name)

            def remember_allow(self, tool_call, *, tool=None):
                pass

            def set_permission_mode(self, mode):
                self.permission_mode = mode

        class ChildClient:
            def __init__(self) -> None:
                self.calls = []
                self.closed = False

            def stream(self, messages, tools=None, reasoning=None):
                self.calls.append(list(messages))
                if len(self.calls) == 1:
                    yield ToolCallEvent(ToolCall("inner", "echo", {"text": "ok"}))
                    yield DoneEvent(2)
                    return
                yield ConversationEvent("fork answer")
                yield DoneEvent(3)

            def close(self):
                self.closed = True

        class ParentClient:
            def __init__(self) -> None:
                self.calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(
                        ToolCall(
                            "load",
                            "LoadSkill",
                            {"name": "delegate", "task": "specific work"},
                        )
                    )
                    yield DoneEvent(5)
                    return
                yield ConversationEvent("parent answer")
                yield DoneEvent(1)

        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills" / "delegate"
            root.mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: delegate\ndescription: Delegate\nmode: fork\n---\nFORK BODY\n",
                encoding="utf-8",
            )
            manager = SkillManager(workspace, home=workspace / "home")
            parent_context = ContextManager(system_prompt="parent system")
            parent_context.add_user("old request")
            parent_context.add_assistant("old answer")
            parent_tools = ToolManager()
            builtin = create_load_skill_tool(manager.load)

            class ProtocolTool:
                name = builtin.name
                description = builtin.description
                params = builtin.params
                is_read_only = builtin.is_read_only
                is_dangerous = builtin.is_dangerous
                is_concurrency_safe = builtin.is_concurrency_safe
                category = builtin.category
                strict = builtin.strict

                def schema(self):
                    return builtin.schema()

                def execute(self, arguments):
                    return builtin.execute(arguments)

                def permission_content(self, arguments):
                    return None

            parent_tools.register(ProtocolTool())
            children = []
            roots = []

            def child_factory():
                tools = ToolManager()
                tools.register(
                    "echo",
                    "Echo",
                    {"type": "object", "properties": {"text": {"type": "string"}}},
                    lambda text: text,
                )
                client = ChildClient()
                child = Agent(
                    client,
                    ContextManager(system_prompt="child system"),
                    tools,
                    permission_checker=PermissionChecker(policy=AskPolicy()),
                    skill_root_callback=lambda value: roots.append(value),
                )
                children.append(child)
                return child

            agent = Agent(
                ParentClient(),
                parent_context,
                parent_tools,
                skill_manager=manager,
                fork_agent_factory=child_factory,
            )
            stream = agent.stream("current request")
            events = []
            try:
                event = next(stream)
                while True:
                    events.append(event)
                    event = (
                        stream.send("allow_once")
                        if isinstance(event, PermissionRequestEvent)
                        else next(stream)
                    )
            except StopIteration:
                pass

            child = children[0]
            child_first_request = child.client.calls[0]
            self.assertTrue(child.client.closed)
            self.assertIn(
                ToolCallEvent(ToolCall("load/inner", "echo", {"text": "ok"})),
                events,
            )
            self.assertIn(
                PermissionRequestEvent("load/inner", "echo", "echo", "approve"),
                events,
            )
            self.assertIn(ToolResultEvent("load/inner", "echo", "ok"), events)
            self.assertIn(ToolResultEvent("load", "LoadSkill", "fork answer"), events)
            self.assertIn(UsageEvent(2), events)
            self.assertIn(UsageEvent(3), events)
            self.assertNotIn(ConversationEvent("fork answer"), events)
            rendered = "\n".join(message.content for message in child_first_request)
            self.assertIn("parent system", rendered)
            self.assertIn("old request", rendered)
            self.assertIn("current request", rendered)
            self.assertIn("FORK BODY", rendered)
            self.assertIn("specific work", rendered)
            self.assertFalse(
                any(
                    message.kind == "tool_call" and message.tool_name == "LoadSkill"
                    for message in child_first_request
                )
            )
            self.assertEqual(roots, [(root.resolve(),), ()])
            parent_rendered = "\n".join(
                message.content for message in parent_context.model_messages()
            )
            self.assertNotIn("FORK BODY", parent_rendered)
            self.assertFalse(
                any(
                    message.tool_name == "echo" for message in parent_context.messages()
                )
            )

    def test_fork_cancellation_cancels_the_parent_turn(self) -> None:
        class InterruptingClient:
            def stream(self, messages, tools=None, reasoning=None):
                raise KeyboardInterrupt
                yield

        class ParentClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ToolCallEvent(
                    ToolCall(
                        "load",
                        "LoadSkill",
                        {"name": "delegate", "task": "stop"},
                    )
                )
                yield DoneEvent()

        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            root = workspace / ".duckduckcode" / "skills"
            root.mkdir(parents=True)
            (root / "delegate.md").write_text(
                "---\nname: delegate\ndescription: Delegate\nmode: fork\n---\nBody\n",
                encoding="utf-8",
            )
            manager = SkillManager(workspace, home=workspace / "home")
            tools = ToolManager()
            tools.register(create_load_skill_tool(manager.load))
            agent = Agent(
                ParentClient(),
                ContextManager(),
                tools,
                skill_manager=manager,
                fork_agent_factory=lambda: Agent(InterruptingClient()),
            )

            events = list(agent.stream("cancel"))

            self.assertIn(ErrorEvent("interrupted", "interrupted"), events)
            self.assertEqual(events[-1], LoopCompleteEvent("cancelled", 1))


if __name__ == "__main__":
    unittest.main()
