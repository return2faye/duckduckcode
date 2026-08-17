# 上下文信息保留测试设计与记录

## 目标

这组测试只回答一个问题：旧对话被压缩成摘要后，Agent 是否仍保留继续完成任务所需的信息，并正确丢弃噪声、过期状态和不可信指令。

它不测长期记忆、session 恢复、模型常识或 SWE-bench 修复率。压缩摘要是被测对象；模型最终能否正确修改代码由端到端 context bench 单独验证。

## 两层测试

### 确定性压缩探针

`duckduckcode-retention` 直接调用当前 Agent 模型和生产使用的 `COMPACTION_SYSTEM_PROMPT`。工具、MCP、LSP、memory、skills、subagent 和 session 全部关闭，每个场景固定 800 个输出 token。

评分不使用 LLM Judge。每个必需标记必须逐字出现，禁止提升的标记必须不存在，并且输出必须包含完整 `<summary>...</summary>`。场景只有同时满足三项才通过：

1. `missing == []`；
2. `forbidden == []`；
3. summary 标签完整。

总信息保留率为 `逐字保留的 required 标记数 / required 标记总数`。当前包含 5 个场景、41 个 required 标记：

| 场景 | 检查内容 | 主要失败模式 |
| --- | --- | --- |
| `atomic-facts` | 20 个不透明事实、pending 状态、200 个噪声项 | 用范围或省略号代替独立值；丢失未完成状态 |
| `instruction-supersession` | 最新 separator/case 规则和明确 revoked 规则 | 旧指令覆盖新指令；撤销关系丢失 |
| `task-state` | completed、pending、blocker cleared、tests not run | 把计划写成完成；重新打开已完成任务；保留过期 blocker |
| `tool-evidence` | 文件、symbol、异常和安全决策 | 丢失定位证据；把工具输出中的 prompt injection 提升为任务 |
| `change-verification` | 修改文件、symbol、验证命令、结果和最终状态 | 把机器可读 `key=value` 改写成自然语言；旧 pending 覆盖后续成功验证 |

固定标记是有意设计的：它把“语义差不多”与“可以无损恢复执行状态”分开，也让 CI 可以稳定判分。噪声只用于制造压缩压力，不计入应保留信息。

### 端到端 context bench

`evals/benches/context` 保留 4 个真实 Agent 工作流用例：

- `context-critical-facts`：压缩后按早期数值契约实现代码；
- `context-latest-instruction`：两次压缩后应用最新规则；
- `context-pending-action`：恢复未完成实现而不是误判为完成；
- `context-compression-quality`：从大型工具输出保留 authoritative facts，丢弃候选值和噪声。

这些用例会实际读取 fixture、触发指定次数的压缩、继续编辑并运行测试。确定性校验负责 compaction 次数、文件边界和测试结果；LLM Judge 只评估自然语言摘要中的事实与行为质量。两层不能互相替代：直接探针定位摘要缺陷更快，端到端用例证明摘要仍能驱动正确行动。

## 本轮发现与修复

测评模型：`deepseek-v4-pro`。时间：2026-08-09。

| 阶段 | 场景通过 | 标记保留 | 发现 |
| --- | ---: | ---: | --- |
| 原始 DeepSeek 配置 | 1/5 | 28/41（68.3%） | V4 Pro 将 low effort 实际映射为 high；800-token 预算被 reasoning 消耗，出现空输出和截断标签 |
| 压缩关闭 thinking | 4/5 | 37/41（90.2%） | 格式完整，但验证记录被自然语言改写，旧 pending 状态未被后续结果覆盖 |
| 模板首次补强 | 5/5 | 41/41（100%） | 单次采样全部保留，但不能证明稳定性 |
| 当前模板重复三次 | 13/15 | 118/123（平均 95.9%，最差 92.7%） | 原子事实稳定保留，少量状态记录仍会被随机改写成近义自然语言 |

修复边界：

- 只有 compaction 使用最小/无 reasoning；普通 Agent 回合仍使用配置的 reasoning effort。
- DeepSeek 通过官方 `thinking.type=disabled` 关闭；OpenAI 对同一内部 `none` 请求保持原来的 `low`，不改变既有行为。
- 模板将每个 `key=value` 视为完整记录，不得重命名 key 或释义 value。
- 模板要求按时间应用状态迁移，后续成功修改或验证覆盖较早的 pending 状态。

当前模板三次重复共发出 15 次请求，provider 报告总 token usage 为 14,216；每轮分别为
4,326、4,682 和 5,208。该数字不区分输入、输出或 cache hit。

修复 DeepSeek thinking 工具回合的 `reasoning_content` 回传后，4 个端到端 context bench
均达到严格 Judge 4/4。`context-compression-quality` 首次得到 3/4 时还暴露了评测阈值
`score >= 3` 会误报 PASS；阈值改为必须 4/4 后复测通过。

## 运行方法

```bash
uv run duckduckcode-retention --max-output-tokens 800
uv run duckduckcode-eval --bench evals/benches/context
uv run duckduckcode-eval-report
```

CLI 在任一场景失败时返回非零退出码，并输出每个场景的 missing、forbidden、summary 和 token usage。结果可保存为 JSON 做不同模型或模板版本的对比。

## 验收标准

- 确定性探针：5/5 场景、41/41 required markers、0 forbidden markers、所有标签完整。
- 端到端套件：4/4 case 达到 Judge 4 分，compaction 次数精确匹配，required tests 全部通过，不能修改 forbidden files。
- 修改摘要模板或 provider reasoning 映射后，两层测试都必须回归。

## 已知边界

- 当前只重复三次，不构成统计置信区间；比较模型稳定性时必须同时报告平均值和最差值。
- 精确标记偏向可机器验证状态，不能覆盖所有自然语言等价表达，因此另保留 LLM Judge 的端到端层。
- 测试不模拟超过 provider 最大上下文、跨进程恢复、长期记忆合并或恶意二进制工具结果。
- 800-token 是故意设置的压力预算；生产 compaction 上限为 20,000，但接近上下文窗口时可用预算会降低，因此小预算失败仍是生产风险。
- Prompt 已显著改善但没有保证逐字复制；进一步提高最差值应评估结构化关键记录通道，
  不能仅根据单个漏项继续堆叠样本特定指令。
