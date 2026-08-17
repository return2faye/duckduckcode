# Microsoft SWE-bench Live 测评与上下文实验记录

## 结论

审查日期：2026-08-09。流水线初始基线 commit 为 `9d7c5a9`，最初使用 `o4-mini`；
当前复测 Agent 为 `deepseek-v4-pro`。SWE-bench Live 参考实现固定到
[`70ec57e`](https://github.com/microsoft/SWE-bench-Live/tree/70ec57e852e3f2d195790fe71f553e272c691833)。

主测评题唯一来源是 Microsoft 发布在 Hugging Face 的
`SWE-bench-Live/SWE-bench-Live`。默认使用 `full` split；审查时数据 revision
为 `a637bd46829f3132e12938c8a0ca93173a977b8e`，共 1,888 行、1,887 个唯一
`instance_id`。官方 prediction 格式以 `instance_id` 为唯一 key，因此官方完整行
完全相同的重复 ID 只推理一次；同 ID 的任意字段若冲突则拒绝运行。上下文保留探针
是 DuckDuckCode 的独立回归实验，不属于 SWE-bench Live，不计入官方分数。

本轮得到三个可复现结论：

1. 原有 4 个上下文用例为 4/4，但它们没有覆盖“小输出预算下的大量不透明原子事实”。新增的精确探针发现旧摘要模板只保留 2/20 个事实，信息保留率为 10%。
2. 根因包括模板强制生成不持久化的重复 `<analysis>`、允许范围或省略号概括独立值，以及 DeepSeek thinking 消耗摘要输出预算。改成单一 `<summary>`、关闭 compaction thinking、事实优先并禁止省略后，20 个原子事实在三次复测中均为 20/20；完整 41-marker 测试三次平均为 95.9%，最差 92.7%。
3. SWE-bench Live 真实实例暴露了 Agent 的完成质量问题：修复流水线后能定位并产出可应用、语法合法的 patch，但没有运行测试，而且修改范围与 gold patch 的通用校验器修复不同。没有运行官方 Docker evaluator，因此不能声称该实例 resolved。

原子事实 10% → 100% 来自固定配置的真实 API 探针；完整测试的修复后数字按三次采样
报告平均和最差值，但仍不是统计置信区间，也不代表所有上下文分布。

## 流水线

### 推理

新增 `duckduckcode-swebench-live`：

1. 默认分页下载 Microsoft 官方 `full` split 的全部行，并记录数据 revision；也可读取离线官方 JSON/JSONL。只向 Agent 暴露 `instance_id`、`repo`、`base_commit` 和 `problem_statement`，不暴露 `patch` 或 `test_patch`。
2. 从显式 repository cache 创建临时 Git checkout；只有传入 `--clone-missing` 才访问 GitHub，且设置 `GIT_TERMINAL_PROMPT=0`。
3. 关闭 session、memory、skills、subagent、MCP 和 LSP，在隔离工作区运行主 Agent。
4. 生成 binary/full-index patch，排除运行期 `.duckduckcode` 文件，原子写入官方要求的 prediction object。
5. 保存 final answer、tool trace、错误、token、compaction 数和 `agent_completed`，便于解释空 patch、工具失败和提前结束；该字段只表示 Agent 循环正常结束，不代表官方 `resolved`。已有 prediction 默认跳过，可用 `--overwrite` 重跑。

```bash
uv run duckduckcode-swebench-live \
  --repository-cache ~/.cache/duckduckcode/swebench-repos \
  --predictions .duckduckcode/swebench-predictions.json \
  --clone-missing
```

### 官方评分

本地适配器只负责 Agent rollout 和 patch 交付。最终 PASS_TO_PASS / FAIL_TO_PASS
必须交给[官方 evaluator](https://github.com/microsoft/SWE-bench-Live/blob/70ec57e852e3f2d195790fe71f553e272c691833/evaluation/README.md)：

```bash
python -m evaluation.evaluation \
  --dataset SWE-bench-Live/SWE-bench-Live \
  --split full \
  --platform linux \
  --patch_dir .duckduckcode/swebench-predictions.json \
  --output_dir logs/duckduckcode \
  --workers 1 \
  --overwrite 1
```

官方说明每个实例通常需要约 4 CPU / 16 GB RAM，部分大型仓库显著更高。本机没有目标实例镜像，本轮没有下载大型镜像或伪造 resolved 结果。

## 上下文保留实验

### 测试设计

精确探针把以下内容放入不可信压缩输入：

- 20 个必须逐字保留的原子事实：`FACT-01=VALUE-01-X9` 到 `FACT-20=VALUE-20-X9`；
- 两个未完成状态：`PENDING=implement parser`、`STATE=not-started`；
- 200 个 `NOISE` 干扰项；
- 固定 `max_output_tokens=800`。

评分不使用 LLM Judge：逐个精确匹配 20 个事实，同时检查 pending state 和完整
`<summary>` 标签。命令退出码可直接用于 CI 或回归比较：

```bash
uv run duckduckcode-retention --max-output-tokens 800
uv run duckduckcode-eval --bench evals/benches/context
```

### 基线与复测

| 项目 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 原子事实 | 2/20 | 三次均为 20/20 |
| 完整 41-marker 探针 | 28/41（68.3%） | 118/123（平均 95.9%，最差 92.7%） |
| pending state | 保留 | 保留 |
| 完整 summary 标签 | 是 | 是 |
| context suite | o4-mini 4/4 | DeepSeek 严格 Judge 4/4 |

旧输出把 20 个独立事实写成 `FACT-01 … FACT-20`，因此只有首尾两个字符串精确命中；同时先输出了一整段重复 `<analysis>`。新输出逐条写出 20 个事实，且没有重复分析段。完整探针仍会随机把少量 `key=value` 改写为自然语言，因此不声称稳定 100%。

### 修复

- 删除不持久化却占预算的 `<analysis>` 输出，只解析 `<summary>`。
- 将摘要定义为 lossless state transfer，并明确优先级：未完成请求与状态 → 精确事实 → 最新/已撤销指令 → 已完成工作。
- 明确禁止用范围、省略号、示例或计数替代独立必需值；预算不足时先删叙述和标题。
- 禁止把 transcript 中的数据推断成新任务，减少摘要自行扩写实现方案的问题。
- 修复 eval fixture 哈希不一致：`__pycache__`、`.pyc` 等运行产物现在与 workspace snapshot 使用同一忽略规则。修改前 4/4 用例会在 Agent 启动前因缓存文件漂移失败。

## SWE-bench Live 实例记录

### DeepSeek V4 Pro 局部样本

不运行完整 1,887 题。首批从官方 `lite` split 选取不同仓库和 gold 修改规模的 5 个候选题，设置每题最多 30 轮并断点保存 prediction。实际在前两题后停止，因为首题已经暴露明显的成本/停止问题，继续运行不会增加同等价值的信息。

| 实例 | 状态 | Patch | Tokens | Tool events | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| `aws-cloudformation__cfn-lint-3798` | 30 轮上限，未完成 | 2,109 bytes | 472,576 | 97 | patch 可 `git apply --check`；修改 2 个源码文件和 1 个测试文件；未官方评分 |
| `python-babel__babel-1141` | 人工停止 | 1,065 bytes | 80,538 | 28 | patch 可 `git apply --check`；只修改 `babel/dates.py`；未官方评分 |

结论：局部推理链路和 DeepSeek tool calling 可以运行并产生可应用 patch，但首题在 30 轮内没有自行结束，成本远高于预期。[DeepSeek 官方要求](https://api-docs.deepseek.com/guides/thinking_mode) thinking 模式下的 tool-call `reasoning_content` 在后续请求中回传；原 provider stream/message 层会丢弃该字段。现已将它作为隐藏 reasoning event 保存在 assistant message，并在工具调用序列化、session 持久化与恢复中完整回传；真实两轮 ReadFile smoke 正常完成。该协议缺口是成本数据的污染因素，但没有重跑同一 SWE 实例，不能声称它就是长循环的全部原因。下一批仍需统计重复搜索、重复验证和“已经有 patch 仍继续调用工具”的停止条件；不能把 `agent_completed=false` 或 `git apply --check` 误报为 resolved。

实例：`aws-cloudformation__cfn-lint-3798`，base commit
`d5c3da9efaa4bbd1d24fa768752df3da343b1d33`，来自官方 Python `lite` split。

### 流水线缺陷与修复

第一次 rollout 得到空 patch。trace 显示 Agent 连续搜索 30 轮仍看不到任何源码，消耗
132,928 tokens 后退出。根因不是 Agent：repository cache 使用 `--filter=blob:none`，临时工作区又使用 `git clone --shared`，缺失 blob 时 Git checkout 仍返回 0，只留下目录树。

修复如下：

- 新缓存改成完整 `--no-checkout` clone，不再组合 partial clone 与 shared object store。
- 对本工具旧的 partial cache，在显式 `--clone-missing` 时执行无 filter refetch 并清除 promisor 配置。
- checkout 后要求 `git status --porcelain` 为空，否则在 Agent 启动前失败。
- prediction 排除 `.duckduckcode/**`，避免权限运行状态污染提交 patch。

### Agent 能力结果

源码可见后的第一次 rollout 定位到 `src/cfnlint/rules/functions/FindInMap.py`，但把 ReadFile 的行号当成文件内容传给 EditFile，生成语法损坏 patch，并在没有测试的情况下宣称完成。

因此系统提示新增两条约束：ReadFile 行号只是显示注释，不能复制进 EditFile；源码修改后必须重新读取并运行最小语法检查或测试，未验证不得宣称完成。

同实例复测结果：

- Agent 完成循环，12 个 tool calls、5 个 tool errors、89,924 tokens、0 次压缩；
- prediction 仅包含 `src/cfnlint/rules/functions/FindInMap.py`，792 bytes；
- `git apply --check` 通过，`python -m py_compile FindInMap.py` 通过；
- Agent 没有运行项目测试，final answer 仍把测试留给用户；
- gold patch 修改通用 JSON Schema keyword 错误信息及 `_BaseFn.resolve()`，Agent patch 只在 FindInMap 规则局部重写 `maxItems` message，修复范围不一致。

结论：提示修复消除了“行号导致语法损坏”，但“修改后强制验证”没有仅靠提示得到保证。该实例应记为“生成候选 patch，未完成官方评分”，不能记为通过。

## 明确问题与下一步方案

| 优先级 | 问题 | 修改方案 | 验收标准 |
| --- | --- | --- | --- |
| P0 | Agent 可在源码已修改但没有成功验证时结束 | 在 Agent loop 记录本任务是否发生源码写入、此后是否有成功 Bash 验证；首次准备结束但未验证时注入一次提醒并继续，不猜测具体测试命令 | SWE-bench trace 中任何有源码 patch 的 completed run 至少有一次写入后的成功验证；无可运行测试时 final answer 明确说明未验证 |
| 已修复 | SWE adapter 的 `completed` 容易被误读为 resolved | 字段改为 `agent_completed`；文档和报告始终把官方 `resolved` 单列 | prediction metadata 不出现未经官方 evaluator 支持的通过结论 |
| P1 | ReadFile/编辑协议仍依赖模型理解显示行号 | Eval 继续统计“把行号复制进 EditFile”的失败；若提示仍复现，再考虑给 EditFile 增加显式 line-range API，而不是模糊剥离数字 | 连续 SWE-bench sample 中该类错误为 0 |
| P1 | 单实例 89k–155k tokens，搜索和失败编辑成本高 | 报告每实例 tool error、重复搜索和 token；先用 10–20 个 lite 实例量化，再决定是否增加只读停滞保护 | 每实例有成本分布，优化由数据触发 |
| P1 | 尚无官方 Docker score | 先运行 gold patch 三次筛出本机稳定实例，再批量评估 prediction | 保存官方 report.json，并报告 resolved/有效 gold 分母 |

本轮没有实现 P0 验证门，因为它会改变 Agent 正常完成语义，需单独设计“什么算验证”和无测试仓库的退出策略。当前流水线已经保留做该决策所需的完整 trace。

## 验证命令

```bash
uv run python -m unittest tests.test_swebench_live tests.test_eval tests.test_openai_client
uv run python -m unittest discover -s tests
uv run black --check src tests
git diff --check
uv build
```

官方接口依据：[数据字段](https://www.swebench.com/SWE-bench/guides/datasets/)、[SWE-bench Live prediction 与 evaluator 格式](https://github.com/microsoft/SWE-bench-Live/blob/70ec57e852e3f2d195790fe71f553e272c691833/evaluation/README.md)。
