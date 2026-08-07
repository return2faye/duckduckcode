"""压缩模块离线压缩率测试。

在"忽略 LLM 调用"的环境中测量：
- 不同规模上下文下的 should_compact 触发点
- compaction_input 的 JSON 输入大小
- 假设不同摘要压缩比下的压缩后 token 数与压缩率

用法:
    uv run python -m pytest tests/test_compaction_ratio.py -v -s
    或直接运行打印报告:
    uv run python tests/test_compaction_ratio.py
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path

from duckduckcode.core.context import (
    COMPACTION_OUTPUT_TOKENS,
    CONTEXT_SAFETY_TOKENS,
    ContextManager,
    Message,
    _estimate_tokens,
    _message_record,
)
from duckduckcode.tools.tool import ToolCall

# ---------------------------------------------------------------------------
# 模拟消息生成
# ---------------------------------------------------------------------------

LOREM_LONG = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum."
)

CODE_SNIPPET = (
    "def process_batch(items: list[dict], config: Config) -> BatchResult:\n"
    '    """Process a batch of items with the given configuration.\n'
    '    Returns a BatchResult containing processed items and metadata."""\n'
    "    results = []\n"
    "    errors = []\n"
    "    for idx, item in enumerate(items):\n"
    "        try:\n"
    "            validator = Validator(config.schema)\n"
    "            validated = validator.validate(item)\n"
    "            processed = Transformer(config.rules).transform(validated)\n"
    "            results.append(processed)\n"
    "        except ValidationError as e:\n"
    '            errors.append({"index": idx, "error": str(e)})\n'
    "        except Exception as e:\n"
    '            logger.exception("Unexpected error processing item %d", idx)\n'
    "    return BatchResult(\n"
    "        items=results, errors=errors,\n"
    "        total=len(items), succeeded=len(results)\n"
    "    )"
)

TOOL_OUTPUT_LONG = json.dumps(
    {
        "files": [
            {
                "path": f"src/services/{name}.py",
                "lines": 120 + i * 17,
                "dependencies": [
                    f"lib.{dep}"
                    for dep in ["auth", "cache", "db", "logging", "metrics"]
                ],
                "complexity": (i % 10) + 1,
            }
            for i, name in enumerate(
                [
                    "user_service",
                    "order_service",
                    "payment_service",
                    "notification",
                    "analytics",
                    "search_index",
                ]
            )
        ],
        "summary": f"Found 6 service files with {LOREM_LONG}",
        "warnings": [f"Warning {i}: {LOREM_LONG[:80]}" for i in range(4)],
    },
    ensure_ascii=False,
)


def _make_tool_call(call_id: str, name: str) -> Message:
    return Message.tool_call(call_id, name, {"path": f"/tmp/{name}", "recursive": True})


def _make_tool_result(call_id: str, output: str) -> Message:
    return Message.tool_result(call_id, output)


def build_mock_conversation(
    turns: int,
    *,
    include_tool_roundtrips: bool = True,
    tool_output_size: str = "medium",
) -> list[Message]:
    """构建模拟对话。

    每轮: user -> assistant (可能带 tool_calls) -> [tool_results]

    Args:
        turns: 对话轮数
        include_tool_roundtrips: 是否包含工具调用
        tool_output_size: small/medium/large — 工具输出大小
    """
    messages: list[Message] = []

    tool_output_map = {
        "small": "OK: done",
        "medium": TOOL_OUTPUT_LONG,
        "large": TOOL_OUTPUT_LONG * 3,
    }
    tool_out = tool_output_map.get(tool_output_size, TOOL_OUTPUT_LONG)

    for turn in range(turns):
        # User message
        user_content = (
            f"请帮我修改 src/services/{['auth','order','payment'][turn%3]}.py "
            f"第 {42 + turn} 行附近的逻辑。{LOREM_LONG[:50 + turn % 80]}"
        )
        messages.append(Message("user", user_content))

        # Assistant text response
        assistant_text = (
            f"I'll update the service file. Let me read it first and then "
            f"apply the changes.\n\n{CODE_SNIPPET}\n\n"
            f"The change involves adding validation for the input parameter "
            f"and handling edge cases. {LOREM_LONG[:60 + turn % 100]}"
        )
        messages.append(Message("assistant", assistant_text))

        if not include_tool_roundtrips:
            continue

        # 1-3 tool calls per turn
        num_tools = 1 + (turn % 3)
        for t in range(num_tools):
            call_id = f"call_{turn}_{t}"
            tool_name = ["ReadFile", "Grep", "EditFile", "Glob"][t % 4]
            messages.append(_make_tool_call(call_id, tool_name))
            messages.append(_make_tool_result(call_id, tool_out))

    return messages


def _token_count(messages: list[Message]) -> int:
    """直接对消息列表进行 token 估算（不含 system prompt），使用 to_openai 格式。"""
    return _estimate_tokens([m.to_openai() for m in messages])


def _token_count_record(messages: list[Message]) -> int:
    """用 _message_record 格式估算（与 compaction_input JSON 一致）。"""
    return _estimate_tokens([_message_record(m) for m in messages])


# ---------------------------------------------------------------------------
# 压缩模拟器
# ---------------------------------------------------------------------------


@dataclass
class CompactionStats:
    """单次压缩的统计数据。"""

    label: str
    total_messages: int
    total_tokens: int  # 压缩前（model_messages 级别）
    raw_message_tokens: int  # 纯消息 token 数（不含系统提示词等）
    system_overhead: int  # 系统提示词等固定开销
    should_compact: bool
    compacted_message_tokens: int  # 被压缩部分（messages[:cutoff]）的 token 数
    compaction_input_tokens: int  # compaction_input JSON 的总 token 数
    compaction_payload_tokens: int  # compaction_input 中 messages 部分的 token 数
    cutoff: int  # 截断的消息条数
    kept_messages: int  # 保留的消息条数
    kept_tokens: int  # 保留消息的 token 数
    summary_assumed_tokens: int  # 模拟摘要 token 数
    after_tokens: int  # 压缩后总 token 数（含系统提示词 + 摘要 + 保留消息）
    compression_ratio: float  # after_tokens / total_tokens
    saved_tokens: int
    saved_pct: float


def simulate_compaction(
    ctx: ContextManager,
    label: str,
    summary_ratio: float = 0.10,
) -> CompactionStats:
    """在给定上下文中模拟一次压缩，返回统计数据。

    Args:
        ctx: 已填充消息的 ContextManager
        label: 场景标签
        summary_ratio: 摘要 token 数占压缩输入 token 数的比例（模拟 LLM 压缩比）

    Returns:
        CompactionStats 包含压缩前后所有数据
    """
    # 压缩前
    total_tokens = ctx.estimated_tokens()
    raw_tokens = _token_count(ctx.messages())
    sys_overhead = total_tokens - raw_tokens
    should = ctx.should_compact()

    # 压缩输入
    candidate = ctx.compaction_input()
    if candidate is None:
        return CompactionStats(
            label=label,
            total_messages=len(ctx.messages()),
            total_tokens=total_tokens,
            raw_message_tokens=raw_tokens,
            system_overhead=sys_overhead,
            should_compact=should,
            compacted_message_tokens=0,
            compaction_input_tokens=0,
            compaction_payload_tokens=0,
            cutoff=0,
            kept_messages=len(ctx.messages()),
            kept_tokens=raw_tokens,
            summary_assumed_tokens=0,
            after_tokens=total_tokens,
            compression_ratio=1.0,
            saved_tokens=0,
            saved_pct=0.0,
        )

    transcript_json, cutoff = candidate
    input_tokens = _estimate_tokens(transcript_json)

    # 被压缩部分（messages[:cutoff]）+ 解析 payload
    all_messages = ctx.messages()
    compacted_msgs = all_messages[:cutoff]
    compacted_msg_tokens = _token_count_record(compacted_msgs)
    parsed = json.loads(transcript_json)
    payload_tokens = _estimate_tokens(parsed["messages"])

    # 保留的消息
    kept_msgs = all_messages[cutoff:]
    kept_tokens = _token_count(kept_msgs)

    # 模拟摘要
    summary_tokens = max(100, int(input_tokens * summary_ratio))
    mock_summary = "X" * (summary_tokens * 3)  # 粗略模拟 ~1 char = ~3 bytes

    # 应用压缩（模拟）
    ctx_copy = ContextManager(
        system_prompt=ctx.system_prompt,
        context_window_tokens=ctx.context_window_tokens,
        compaction_trigger_tokens=ctx.auto_compact_tokens,
    )
    ctx_copy.restore(list(ctx.messages()), abstraction=ctx.abstraction)
    ctx_copy.apply_compaction(mock_summary, cutoff)
    after_tokens = ctx_copy.estimated_tokens()

    saved_tokens = total_tokens - after_tokens
    ratio = after_tokens / total_tokens if total_tokens > 0 else 1.0

    return CompactionStats(
        label=label,
        total_messages=len(all_messages),
        total_tokens=total_tokens,
        raw_message_tokens=raw_tokens,
        system_overhead=sys_overhead,
        should_compact=should,
        compacted_message_tokens=compacted_msg_tokens,
        compaction_input_tokens=input_tokens,
        compaction_payload_tokens=payload_tokens,
        cutoff=cutoff,
        kept_messages=len(kept_msgs),
        kept_tokens=kept_tokens,
        summary_assumed_tokens=summary_tokens,
        after_tokens=after_tokens,
        compression_ratio=ratio,
        saved_tokens=saved_tokens,
        saved_pct=(1.0 - ratio) * 100,
    )


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------


def format_report(stats_list: list[CompactionStats]) -> str:
    """格式化为可读的表格报告。"""
    lines = []
    sep = "=" * 90
    lines.append(sep)
    lines.append("压缩模块离线压缩率测试报告")
    lines.append(sep)
    lines.append("")

    # 参数说明
    lines.append("测试参数:")
    lines.append(f"  COMPACTION_OUTPUT_TOKENS = {COMPACTION_OUTPUT_TOKENS:,}")
    lines.append(f"  CONTEXT_SAFETY_TOKENS    = {CONTEXT_SAFETY_TOKENS:,}")
    lines.append("  默认 context_window     = 200,000 tokens")
    lines.append("  模拟摘要压缩比          = 10%（compaction_input → summary）")
    lines.append("")

    # 表头
    header = (
        f"{'场景':<22} {'总消息':>6} {'总tokens':>10} "
        f"{'触发':>4} {'输入tokens':>10} {'截断':>5} "
        f"{'保留':>5} {'摘要tokens':>10} {'压缩后':>10} "
        f"{'节省%':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for s in stats_list:
        trigger = "是" if s.should_compact else "否"
        lines.append(
            f"{s.label:<22} {s.total_messages:>6} {s.total_tokens:>10,} "
            f"{trigger:>4} {s.compaction_input_tokens:>10,} {s.cutoff:>5} "
            f"{s.kept_messages:>5} {s.summary_assumed_tokens:>10,} "
            f"{s.after_tokens:>10,} {s.saved_pct:>6.1f}%"
        )

    lines.append("")
    lines.append("-" * len(header))

    # 详细分析
    lines.append("")
    lines.append("详细分析:")
    lines.append("")
    for s in stats_list:
        lines.append(f"── {s.label} ──")
        lines.append(f"  总消息数:            {s.total_messages}")
        lines.append(f"  系统开销 tokens:     {s.system_overhead:,}")
        lines.append(f"  纯消息 tokens:       {s.raw_message_tokens:,}")
        lines.append(f"  总 tokens(含系统):   {s.total_tokens:,}")
        lines.append(f"  触发压缩:            {s.should_compact}")
        if s.cutoff > 0:
            lines.append(
                f"  截断点(cutoff):      {s.cutoff} 条消息 "
                f"(共 {s.total_messages} 条，删除前 {s.cutoff} 条)"
            )
            lines.append(f"  保留消息数:          {s.kept_messages} 条")
            lines.append(f"  保留 tokens:         {s.kept_tokens:,}")
            lines.append("")
            lines.append(f"  ► 被压缩部分（前 {s.cutoff} 条消息）:")
            lines.append(f"    原始 token 数:     {s.compacted_message_tokens:,}")
            lines.append(
                f"    JSON 包装后:       {s.compaction_payload_tokens:,} "
                f"(+{s.compaction_payload_tokens - s.compacted_message_tokens:,} JSON 结构开销)"
            )
            lines.append(
                f"    压缩输入(含摘要):  {s.compaction_input_tokens:,} "
                f"(+{s.compaction_input_tokens - s.compaction_payload_tokens:,} previous_summary)"
            )
            lines.append(
                f"    模拟摘要输出:      {s.summary_assumed_tokens:,} "
                f"(压缩比 {s.summary_assumed_tokens/max(s.compacted_message_tokens,1)*100:.1f}%)"
            )
            lines.append(
                f"    节省 tokens:       "
                f"{s.compacted_message_tokens - s.summary_assumed_tokens:,}"
            )

            lines.append("")
            lines.append(f"  ► 压缩前后总览:")
            lines.append(f"    压缩前总 tokens:   {s.total_tokens:,}")
            lines.append(f"    压缩后总 tokens:   {s.after_tokens:,}")
            lines.append(f"    节省 tokens:       {s.saved_tokens:,}")
            lines.append(
                f"    压缩率:            {s.saved_pct:.1f}% "
                f"(压缩后/压缩前 = {s.compression_ratio:.3f})"
            )
        else:
            lines.append("  (无法压缩：没有完整 turn 或 cutoff=0)")
        lines.append("")

    lines.append(sep)
    lines.append("注意: 摘要大小基于 10% 压缩比模拟，实际取决于 LLM 输出质量。")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 按比例模拟不同摘要压缩比
# ---------------------------------------------------------------------------


def format_ratio_table(stats_list: list[CompactionStats]) -> str:
    """针对不同摘要压缩比（5%/10%/20%/30%）输出对比表。"""
    lines = []
    lines.append("")
    lines.append("不同摘要压缩比下的压缩率对比")
    lines.append("（摘要 tokens = compaction_input_tokens × ratio）")
    lines.append("")
    header = (
        f"{'场景':<22} {'输入tokens':>10} | {'5%':>10} {'10%':>9} {'20%':>9} {'30%':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for s in stats_list:
        if s.compaction_input_tokens == 0:
            continue
        input_t = s.compaction_input_tokens
        ratios = []
        for r in [0.05, 0.10, 0.20, 0.30]:
            summary_t = max(100, int(input_t * r))
            after = s.kept_tokens + summary_t  # 简化：忽略系统提示词
            ratio_val = after / max(s.total_tokens, 1)
            ratios.append(f"{after:>10,}")
        lines.append(
            f"{s.label:<22} {input_t:>10,} | {ratios[0]} {ratios[1]} {ratios[2]} {ratios[3]}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class CompactionRatioTest(unittest.TestCase):
    """压缩率离线分析测试。"""

    def _make_context(
        self,
        turns: int,
        *,
        context_window: int = 200_000,
        trigger: int | None = None,
        tool_size: str = "medium",
    ) -> ContextManager:
        """创建填充了模拟消息的 ContextManager。"""
        ctx = ContextManager(
            system_prompt="You are a test agent.",
            context_window_tokens=context_window,
            compaction_trigger_tokens=trigger,
        )
        messages = build_mock_conversation(turns, tool_output_size=tool_size)
        ctx.restore(messages)
        return ctx

    # ---- 基本功能测试 ----

    def test_empty_context_no_compact(self) -> None:
        ctx = ContextManager(system_prompt="test")
        self.assertFalse(ctx.should_compact())
        self.assertIsNone(ctx.compaction_input())

    def test_single_turn_no_compact(self) -> None:
        ctx = ContextManager(system_prompt="test")
        ctx.add_user("hello")
        ctx.add_assistant("hi")
        self.assertFalse(ctx.should_compact())

    def test_multi_turn_cutoff(self) -> None:
        """验证截断点在完整 turn 边界。"""
        ctx = self._make_context(50)
        candidate = ctx.compaction_input()
        self.assertIsNotNone(candidate)
        _, cutoff = candidate
        # cutoff 应该是某个 user 消息的索引
        messages = ctx.messages()
        self.assertGreater(cutoff, 0)
        self.assertEqual(messages[cutoff].role, "user")
        self.assertEqual(messages[cutoff].kind, "message")

    # ---- 不同规模场景测试 ----

    def test_scenario_small(self) -> None:
        """小型对话 ~20 轮，不触发压缩。"""
        ctx = self._make_context(20, trigger=150_000)
        stats = simulate_compaction(ctx, "小型(20轮)", summary_ratio=0.10)
        self.assertFalse(stats.should_compact)
        # 不触发压缩时 cutoff 可能 >0 但 should_compact 为 False
        self.assertGreater(stats.total_messages, 0)

    def test_scenario_medium(self) -> None:
        """中型对话 ~100 轮，接近但未触发。"""
        ctx = self._make_context(100, trigger=150_000)
        stats = simulate_compaction(ctx, "中型(100轮)", summary_ratio=0.10)
        # 记录数据即可，不强制断言触发
        self.assertGreater(stats.total_tokens, 10_000)

    def test_scenario_large_triggers_compaction(self) -> None:
        """大型对话 ~200 轮，触发压缩。"""
        ctx = self._make_context(200)
        stats = simulate_compaction(ctx, "大型(200轮)", summary_ratio=0.10)
        # 默认 trigger = 200_000 - 20_000 - 13_000 = 167_000
        # 200 轮应该超过此值
        self.assertTrue(stats.should_compact)
        self.assertGreater(stats.cutoff, 0)
        # 压缩后应该比压缩前小
        self.assertLess(stats.after_tokens, stats.total_tokens)
        # 压缩率应该在合理范围
        self.assertGreater(stats.saved_pct, 0)

    def test_scenario_xlarge(self) -> None:
        """超大型对话 ~400 轮。"""
        ctx = self._make_context(400)
        stats = simulate_compaction(ctx, "超大型(400轮)", summary_ratio=0.10)
        self.assertTrue(stats.should_compact)
        self.assertLess(stats.after_tokens, stats.total_tokens)

    def test_scenario_with_large_tool_outputs(self) -> None:
        """工具输出较大的场景。"""
        ctx = self._make_context(100, tool_size="large")
        stats = simulate_compaction(ctx, "大工具输出(100轮)", summary_ratio=0.10)
        # 大型工具输出应该产生更多 tokens
        self.assertGreater(stats.total_tokens, 50_000)

    def test_scenario_no_tools(self) -> None:
        """纯对话无工具调用场景，不触发压缩（token 密度较低）。"""
        ctx = ContextManager(system_prompt="test")
        messages = build_mock_conversation(300, include_tool_roundtrips=False)
        ctx.restore(messages)
        stats = simulate_compaction(ctx, "纯对话(300轮)", summary_ratio=0.10)
        # 纯对话 token 密度低，300 轮不一定会触发默认 trigger
        # 但 cutoff 仍有效
        self.assertGreater(stats.cutoff, 0)
        self.assertGreater(stats.total_messages, 0)

    # ---- apply_compaction 功能测试 ----

    def test_apply_compaction_removes_messages(self) -> None:
        """验证 apply_compaction 正确删除消息并设置 abstraction。"""
        ctx = self._make_context(50)
        candidate = ctx.compaction_input()
        self.assertIsNotNone(candidate)
        _, cutoff = candidate
        original_count = len(ctx.messages())
        ctx.apply_compaction("This is a test summary of the conversation.", cutoff)
        self.assertLess(len(ctx.messages()), original_count)
        self.assertEqual(ctx.abstraction, "This is a test summary of the conversation.")

    def test_apply_compaction_rejects_invalid(self) -> None:
        """验证 apply_compaction 拒绝无效参数。"""
        ctx = self._make_context(50)
        with self.assertRaises(ValueError):
            ctx.apply_compaction("", 1)
        with self.assertRaises(ValueError):
            ctx.apply_compaction("summary", 0)
        with self.assertRaises(ValueError):
            ctx.apply_compaction("summary", len(ctx.messages()) + 1)

    # ---- token 估算一致性 ----

    def test_estimated_tokens_includes_system_prompt(self) -> None:
        """验证 estimated_tokens 包含 system prompt。"""
        ctx = ContextManager(system_prompt="You are helpful.")
        ctx.add_user("hi")
        ctx.add_assistant("hello")
        full = ctx.estimated_tokens()
        raw = _token_count(ctx.messages())
        self.assertGreater(full, raw)

    def test_token_estimation_monotonic(self) -> None:
        """验证添加消息后 token 数单调递增。"""
        ctx = ContextManager(system_prompt="test")
        prev = ctx.estimated_tokens()
        for i in range(10):
            ctx.add_user(f"message {i} " + LOREM_LONG)
            ctx.add_assistant(f"reply {i} " + LOREM_LONG[:50])
            current = ctx.estimated_tokens()
            self.assertGreaterEqual(current, prev)
            prev = current


# ---------------------------------------------------------------------------
# 主入口：打印完整报告
# ---------------------------------------------------------------------------


def run_report() -> None:
    """运行完整的压缩率分析并输出报告。"""
    stats_list: list[CompactionStats] = []

    # 场景定义
    scenarios = [
        # (标签, 轮数, 工具输出大小, context_window, trigger, 摘要比)
        ("小型(20轮)", 20, "medium", 200_000, None, 0.10),
        ("中型(50轮)", 50, "medium", 200_000, None, 0.10),
        ("中型(100轮)", 100, "medium", 200_000, None, 0.10),
        ("大型(200轮)", 200, "medium", 200_000, None, 0.10),
        ("大型(300轮)", 300, "medium", 200_000, None, 0.10),
        ("超大型(500轮)", 500, "medium", 200_000, None, 0.10),
        ("纯对话(300轮)", 300, "medium", 200_000, None, 0.10),
        ("大工具输出(100轮)", 100, "large", 200_000, None, 0.10),
        ("小窗口(100轮/64K)", 100, "medium", 64_000, None, 0.10),
    ]

    for label, turns, tool_size, cw, trigger, ratio in scenarios:
        ctx = ContextManager(
            system_prompt="You are a test agent.",
            context_window_tokens=cw,
            compaction_trigger_tokens=trigger,
        )
        if label.startswith("纯对话"):
            messages = build_mock_conversation(turns, include_tool_roundtrips=False)
        else:
            messages = build_mock_conversation(turns, tool_output_size=tool_size)
        ctx.restore(messages)
        stats = simulate_compaction(ctx, label, summary_ratio=ratio)
        stats_list.append(stats)

    # 打印报告
    report = format_report(stats_list)
    print(report)

    # 不同摘要压缩比对比
    ratio_table = format_ratio_table(stats_list)
    print(ratio_table)


if __name__ == "__main__":
    run_report()
