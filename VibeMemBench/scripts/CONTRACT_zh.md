# VibeMemBench 记忆系统接入契约（CONTRACT）

> 语言：[English](./CONTRACT.md) | **简体中文**

> 面向**外部参赛团队**的唯一接入说明。只看本文档 + [`memory_interface.py`](./memory_interface.py)
> 一个文件，就能把你的记忆系统接进 VibeMemBench 跑评测——无需接触本仓库任何内部代码。

本契约的接口设计对齐 [Agent Memory Leaderboard (AML)](https://agentmemoryleaderboard.ai/api-guide)
的职责边界与硬约束规范。

---

## 1. 职责边界：你只做 Add 和 Search

VibeMemBench 把一次「用记忆辅助解题」的完整链路拆成 4 步，其中**只有前两步归你**：

```
┌────┬────────────┬──────────┬──────────────────────────────────────────┐
│ #  │ 阶段        │ 动作      │ 归属                                      │
├────┼────────────┼──────────┼──────────────────────────────────────────┤
│ 01 │ Store      │ Add      │ ▶ 参赛方：把历史解题轨迹写进你的记忆库        │
│ 02 │ Retrieve   │ Search   │ ▶ 参赛方：按 query 检索出最相关的若干条记忆   │
├────┼────────────┼──────────┼──────────────────────────────────────────┤
│ 03 │ Generate   │ Answer   │ ■ 平台固定：检索结果如何注入 coding agent     │
│ 04 │ Score      │ Eval     │ ■ 平台固定：如何评判最终解题结果              │
└────┴────────────┴──────────┴──────────────────────────────────────────┘
```

**边界含义**：`run_memory.py` 只负责给你喂历史（`add`）和向你要检索结果
（`search`）。检索结果之后怎么用、怎么打分，是平台另一条管线的事，与你的接入代码完全解耦。
你**不需要**、也**不应该**关心 Answer / Eval。

---

## 2. 你要实现的接口：`MemorySystem`

只有 **2 个必选方法** + 1 个可选方法。全部**同步**、**in-process**（当前不做 HTTP）。

```python
class MemorySystem(abc.ABC):
    def add(self, session: HistorySession, *, scope: MemoryScope) -> None: ...
    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> List[RetrievedMemory]: ...
    def count(self) -> Optional[int]: ...   # 可选，默认返回 None
```

### 从 AML 继承的硬约束（必须遵守）

| 约束 | 说明 |
|---|---|
| **严格同步** | `add()` 返回即代表该轨迹**已可被 `search()` 检索到**。**禁止**返回 task id、**禁止**轮询、**禁止**任何异步/延迟/最终一致语义。 |
| **返回前自行排序** | `search()` 必须**自己按相关性降序排好**再返回；harness 不会替你重排（乱序只告警）。 |
| **长度上限** | `search()` 返回长度必须 `<= top_k`；超出会被 harness 截断并告警。 |
| **无结果返回空列表** | 没有相关记忆时返回 `[]`，**不是** `None`、**不是**抛异常。 |
| **`user_id` 精确匹配零泄漏** | `search()` 只能在 `scope.user_id` 对应的记忆范围内检索，**严禁跨 `user_id` 返回**其它命名空间的记忆。 |

---

## 3. 三个数据契约类（逐字段表）

所有字段名严格对齐 AML 的 payload 键名。每个类都实现了 `to_payload() -> dict` 与
`from_payload(d) -> Self`，为将来加一层薄 HTTP transport 留门（届时契约无需改动）。

### 3.1 `MemoryScope`（frozen dataclass）—— 隔离边界，对应 AML 的 `user_id` 语义

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `user_id` | `str` | ✅ | 记忆命名空间键，形如 `vmb:{run_id}:{repo_dirname}:{target_instance_id}`。**你只能在此命名空间内读写。** |
| `repo` | `str` | ✅ | 仓库全名，如 `adamchainz/flake8-comprehensions`。 |
| `repo_dirname` | `str` | ✅ | 仓库目录名（`repo` 的 `/` 换成 `__`），如 `adamchainz__flake8-comprehensions`。 |
| `language` | `str` | ✅ | 语言，如 `python` / `js` / `go` / `ts` / `rust` / `kotlin` / `swift` / `java` / `php`。 |
| `target_instance_id` | `str` | ✅ | 当前待解 target 的 instance_id，如 `adamchainz__flake8-comprehensions-553`。 |
| `base_commit` | `str` | ✅ | target 的基线 commit（时间因果切点的锚）。 |
| `created_at` | `str` | ✅ | target 的创建时间字符串。 |

> **⚠️ 隔离边界不是「用户」，而是「仓库 × 时间因果切点」。**
> 本 benchmark 为**每个 target** 单独建一份记忆，且只能喂「发生在该 target 之前」的同仓库历史。
> `user_id` 里编入 `run_id`（本次运行）+ `repo_dirname`（仓库）+ `target_instance_id`（时间切点），
> 确保任意两个 target 的记忆命名空间互不重叠。harness 另外对每个 target **新建一个 `MemorySystem`
> 实例**，形成「**进程内实例隔离 + `user_id` 命名空间隔离**」的双保险。

### 3.2 `HistorySession`（frozen dataclass）—— 一条历史轨迹，对应 AML 的一次 Add

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `session_id` | `str` | ✅ | 历史 issue 的 instance_id，如 `adamchainz__flake8-comprehensions-143`。 |
| `messages` | `List[Dict[str, Any]]` | ✅ | **剥离后的原始 message dict 列表**，顺序保持（详见第 4、5 节）。 |
| `trajectory_format` | `Optional[str]` | ✅ | `"mini-swe-agent-1.1"`，或 `None`（legacy 变体，见 §4.1）。 |
| `meta` | `Dict[str, Any]` | ✅ | 会话级元信息白名单（见下表）。 |

**整条 trajectory 一次性 Add，分块是你的自由。** 一条历史 issue 的完整 messages（p50 = 113 条，
max = 514 条）**一次性**传给 `add()`，harness **不做任何分块**。AML 的「>20 消息或 >2000 词就切分」
规则**不采纳**——coding 轨迹在任意句子边界切分会把 `assistant(tool_calls)` 和它对应的 `tool` 结果
拆散，两段都会失效。要不要分块、怎么分块，完全由你决定。

`meta` 白名单（只从 traj 的 `info` 里挑这些，**禁止透传 `info.config`**）：

| 键 | 类型 | 出现条件 | 含义 |
|---|---|---|---|
| `num_messages` | `int` | 恒存在 | 本 session 的消息条数（harness 计算补充）。 |
| `issue_number` | `int` | 恒存在 | 本条历史的 PR/issue 尾号，即它在时间轴上的位置。号小 = 发生得早，可直接用来排序或加权 recency。 |
| `target_issue_number` | `int` | 恒存在 | 当前 target 的尾号，也就是因果切点。恒满足 `issue_number < target_issue_number`。 |
| `exit_status` | `str` | 仅标准变体 | 如 `"Submitted"`。 |
| `mini_version` | `str` | 仅标准变体 | mini-swe-agent 版本。 |
| `model_stats` | `{"instance_cost": float, "api_calls": int}` | 仅标准变体 | 成本与调用次数。 |

> legacy 变体（`info` 缺失/为空）时，`exit_status` / `mini_version` / `model_stats` **直接省略**，
> **不填 `None` 占位**——所以 `meta` 里可能只有 `num_messages` / `issue_number` /
> `target_issue_number` 三个键。
>
> 本 benchmark 的时间轴就是尾号，**不提供真实 PR 日期**：发布数据里只有 111 个 target 的
> `created_at`，3634 条历史的没有。轨迹消息里的 `extra.timestamp` 是当初跑 agent 的机器时间，
> 不是 PR 的真实时间，拿来排序没有意义。

### 3.3 `RetrievedMemory`（dataclass）—— 一条检索结果，对应 AML Search 响应 `data[]` 的 item

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `id` | `str` | ✅ 非空 | 记忆唯一 ID（对齐 AML 的 `id`）。 |
| `content` | `str` | ✅ 非空 | 记忆正文（对齐 AML 的 `content`）。 |
| `score` | `float` | ✅ | 相关性分数，**越大越相关**；返回列表必须按它降序。 |
| `created_at` | `Optional[str]` | ⬜ | ISO8601 时间戳，默认 `None`。 |
| `source_session_id` | `Optional[str]` | ⬜ | **本 benchmark 扩展字段**：该记忆来自哪条历史 issue，用于归因分析。 |
| `metadata` | `Dict[str, Any]` | ⬜ | 任意附加信息，默认 `{}`。 |

---

## 4. Wire message schema（`HistorySession.messages` 里每个 dict 的精确结构）

`add()` 收到的每条 message 就是 traj.json 里的**原始 dict**，保留原生字段名与嵌套结构。
**我们不做任何归一化**：不把 `exit` 改成 `user`、不对 `content` 做 `json.dumps`、不丢弃
`tool_calls` / `reasoning_content` / `extra`。你拿到的就是下面这套实测 schema。

### 4.1 轨迹顶层：两种变体都要兼容

| 变体 | 顶层结构 |
|---|---|
| **标准** | `{"info": {...}, "messages": [...], "trajectory_format": "mini-swe-agent-1.1"}` |
| **legacy** | `{"messages": [...]}` —— **无 `trajectory_format`，`info` 缺失或为空 dict** |

> harness 两种都兼容（并保留「顶层直接是 list」的防御分支）。对 legacy 变体，
> `HistorySession.trajectory_format == None`、`meta` 只含 `num_messages`。
> **当前开源数据快照里全部是 legacy 变体**，但你的实现应对标准变体同样健壮。

### 4.2 `role` 取值（5 种，`exit` 是合法 role 不是异常）

`system` / `user` / `assistant` / `tool` / `exit`。

### 4.3 各 role 的顶层键变体（实测全量，共 4 种形态）

```
system / user : ('content', 'role')
tool          : ('content', 'extra', 'role', 'tool_call_id')
assistant     : ('content', 'extra', 'function_call', 'provider_specific_fields',
                 'reasoning_content', 'role', 'tool_calls')      # ← 后三个中 function_call / provider_specific_fields 已被剥离，见 §5
exit          : ('content', 'extra', 'role')
```

> 注：`assistant` 原始形态含 `function_call` 与 `provider_specific_fields`，但这两者已在
> 剥离阶段删除（§5），所以你在 `add()` 里实际看到的 assistant 顶层键是
> `('content', 'extra', 'reasoning_content', 'role', 'tool_calls')`。

### 4.4 `extra` 键变体（实测 5 种）

```
tool（绝大多数）  : ('exception_info', 'raw_output', 'returncode', 'timestamp')
tool（异常）      : ('exception', 'exception_info', 'exception_type', 'raw_output',
                    'returncode', 'timestamp')
assistant         : ('actions', 'cost', 'response', 'timestamp')   # ← response 已被剥离并提升出 usage/model，见 §5
exit              : ('exit_status', 'submission')
exit（异常）      : ('exception_str', 'exit_status', 'submission', 'traceback')
```

> 剥离后 assistant 的 `extra` 实际是 `('actions', 'cost', 'timestamp', 'usage', 'model')`
> （`response` 删除，`usage`/`model` 从 `response` 提升而来）。

### 4.5 字段实例（实测）

```pythonc
# ── assistant ──────────────────────────────────────────────────────────
content           = ""                       # ⚠️ 实测 84.1% 的 assistant content 是空串！
reasoning_content = "Let me analyze the issue..."   # 模型的思考链，高价值
tool_calls        = [{"index": 0, "id": "call_b95145f54569446ca6bd432d",
                      "type": "function",
                      "function": {"name": "bash",
                                   "arguments": "{\"command\": \"cd /testbed && ls -la\"}"}}]
extra.actions     = [{"command": "cd /testbed && ls -la",           # ← 已解析好的命令
                      "tool_call_id": "call_b95145f54569446ca6bd432d"}]
extra.cost        = 0.0                       # float
extra.timestamp   = 1782297212.8508334        # float，Unix **秒**（不是毫秒！见 §5.3）
extra.usage       = {...}                     # 从 response 提升，见 §4.6
extra.model       = "deepseek-v4-flash"       # 从 response 提升，per-call 模型溯源

# ── tool ───────────────────────────────────────────────────────────────
content           = "<returncode>0</returncode>\n<output>\n...\n</output>"
                    # ⚠️ content 里**不含命令本身**，命令在前一条 assistant 的 tool_calls / extra.actions 里
tool_call_id      = "call_b95145f54569446ca6bd432d"   # ← 用它与 assistant 的动作配对
extra.raw_output  = "..."
extra.returncode  = 0
extra.exception_info = ""

# ── exit ───────────────────────────────────────────────────────────────
content           = "<最终 patch diff 全文>"    # 历史 agent 的实际产出，高价值记忆
extra.exit_status = "Submitted"
extra.submission  = "<最终 patch diff 全文>"
```

> **正确的读取姿势**（这是 `EchoMemorySystem` 示例演示的重点）：
> assistant 的 `content` 常是空串，**别把它当记忆**；真正的动作在 `extra.actions[].command`
> （或 `tool_calls[].function.arguments` 里的 `command`），思考在 `reasoning_content`；命令的
> **输出**在紧随其后、`tool_call_id` 相同的 `tool` 消息的 `extra.raw_output` / `extra.returncode`。
> 用 `tool_call_id` 把「命令 ↔ 输出」配对，再组合成一条记忆。

### 4.6 `extra.usage` 形状（子键因 provider 而异，必须原样透传）

```json
{"completion_tokens": 194, "prompt_tokens": 1962, "total_tokens": 2156,
 "completion_tokens_details": {"accepted_prediction_tokens": null, "audio_tokens": null,
   "reasoning_tokens": 144, "rejected_prediction_tokens": null},
 "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}}
```

> 另一个样本里 `completion_tokens_details` 还多了 `text_tokens` / `image_tokens` 键——
> 印证了 **`usage` 必须整体原样透传、不得硬编码字段名**。

---

## 5. 我们剥离了什么、为什么（原样透传的唯一例外）

在「原样透传」的基础上，harness **只剔除以下冗余项**（实测可显著减小体积）。
你的 `add()` 收到的 message 已经过这套剥离。

### 5.1 删除的字段

| 动作 | 字段 | 理由 |
|---|---|---|
| **删除** | `extra.response` | 原始 LLM API 响应对象，占全量字符的 ~35%；其 `choices` 是 `content` / `tool_calls` / `reasoning_content` 的**完整重复**。 |
| **删除** | `provider_specific_fields` | 其 `reasoning_content` 与顶层字段 **100% 重复**；`refusal` 实测恒为 `null`。 |
| **删除** | `function_call` | 实测 **100% 恒为 `null`**，OpenAI legacy 死字段。 |

### 5.2 从 `extra.response` 提升到 `extra` 的字段

| 动作 | 新字段 ← 来源 | 理由 |
|---|---|---|
| **提升** | `extra.usage` ← 原 `extra.response.usage` | usage 是 response 里**唯一不冗余**的信息，原样透传整个 dict。 |
| **提升** | `extra.model` ← 原 `extra.response.model` | 单个字符串，保留 per-call 模型溯源。 |

### 5.3 我们**不做**的事（重要）

- **不做任何字段重命名。**
- **不做任何时间戳单位换算。** `extra.timestamp` 保持 **Unix 秒 `float`** 原样
  （如 `1782297212.8508334`）。
  > ⚠️ **这与 AML 不同**：AML 的 `created_at` / 时间戳用**毫秒 `int`**。本 benchmark 的
  > wire message 里 `extra.timestamp` 是**秒 `float`**。若你的系统需要毫秒 int，请自行
  > `int(ts * 1000)` 换算。

### 5.4 我们**必须保留**的字段

`extra.cost`、`extra.timestamp`、`extra.actions`、`extra.raw_output`、`extra.returncode`、
`extra.exception_info`、`extra.exception`、`extra.exception_type`、`extra.exit_status`、
`extra.submission`、`extra.exception_str`、`extra.traceback`；以及顶层的 `content`、
`reasoning_content`、`tool_calls`、`tool_call_id`、`role`。

---

## 6. 时间因果过滤（harness 侧规则，你需要理解）

给某个 target 建记忆时，harness **只会喂「发生在它之前」的同仓库历史**，否则就是用未来经验
解决过去问题（数据泄漏）。规则：

> 同一 `repo_dirname` 下，`instance_id` **结尾数字 < target 结尾数字**的历史才会被喂给 `add()`。

例如 target `...-553`，只会喂 `...-143` / `...-179` / `...-205` 等结尾编号 `< 553` 的历史；
`...-553`（target 自身）及任何编号更大的都被排除。因此你收到的所有 `HistorySession`
在时间上都**严格早于**当前 target。

注意 target 自身的轨迹**确实躺在磁盘上**（`history/<lang>/<repo>/` 目录里就有那一条），
挡住它的只有这个过滤，所以它是硬约束而非装饰。两个编号都以
`meta["issue_number"]` 与 `meta["target_issue_number"]` 显式给你，不用自己从
`session_id` 里抠字符串。

**大文件防御**：harness 提供 CLI 参数 `--max-session-mb`（默认 64，十进制 MB）。超过阈值的
history session 会在 `json.load` **之前**被跳过、记 WARNING，并写入结果 JSON 的
`skipped_sessions`（含 `session_id` / `size_mb` / `reason`）。否则数据集里那条 ~102 MB 的轨迹
会把进程内存打爆。**这意味着你的 `add()` 不保证收到某仓库的全部历史**——被体积拦下的会缺席。

---

## 7. 金标红线（绝对不可逾越）

任何传给 `add()` / `search()` 的数据、以及任何写进结果 JSON 的字段中，**绝对不得出现 target
的金标答案**。金标清单（target 行的这些字段 harness 结构性地从不透传给你）：

```
patch, test_patch, pr_description, FAIL_TO_PASS, PASS_TO_PASS,
install_config, meta, 以及 target 自己的 info.submission
```

- `search()` 的 `query` **只会是 `target["problem_statement"]`**，一个字符都不会多。
- **注意区分**：`HistorySession` 里**历史轨迹自己的** `exit` 消息 / `extra.submission` 是
  **历史 agent 的实际产出**，**不是** target 的金标，属于**合法记忆材料，必须保留、可用**。
- harness 在 `run_one()` 结束前会调用 `_assert_no_gold_leakage()` 做校验：
  - `query` 必须逐字符等于 `problem_statement`；
  - 结果 JSON 全文不得含 target 的 `patch` / `test_patch` / `pr_description` / `FAIL_TO_PASS` 的**值**。
  > `PASS_TO_PASS` 不作子串指纹——它们是「改动前后都通过」的**既有**测试，同仓库更早的历史
  > 本来就跑了同一套测试（如 pytest 输出里的 `...::test_version`），属仓库共享词汇、不是泄漏。
  > `install_config` / `meta` 同理不作指纹（可能内嵌 repo/image 标识符，会与结果里的合法字段误撞），
  > 改由「harness 从不读取透传」结构性兜底。

---

## 8. 输出格式（harness 产出，供你了解）

每个 target 产出一个目录 `{out_root}/{method}/{language}/{instance_id}/`，里面两个文件：

| 文件 | 作用 |
|---|---|
| `memory.txt` | **注入 agent 的经验正文**。把 `retrieved[]` 按已排好的降序拼成可读文本，`run_agent.py --method <你的名字>` 直接读它。 |
| `memory.json` | 完整溯源，结构如下。不参与注入，用于审计与归因。 |

```json
{
  "method": "demo",
  "run_id": "20260913-142530",
  "instance_id": "adamchainz__flake8-comprehensions-553",
  "repo": "adamchainz/flake8-comprehensions",
  "language": "python",
  "user_id": "vmb:20260913-142530:adamchainz__flake8-comprehensions:adamchainz__flake8-comprehensions-553",
  "elapsed_seconds": 12.34,
  "num_histories_fed": 8,
  "num_messages_fed": 904,
  "skipped_sessions": [{"session_id": "...", "size_mb": 102.3, "reason": "exceeds --max-session-mb"}],
  "total_memories_stored": 123,
  "num_retrieved": 10,
  "retrieved": [
    {"id": "...", "content": "...", "score": 0.8712, "created_at": null,
     "source_session_id": "adamchainz__flake8-comprehensions-528", "metadata": {}}
  ]
}
```

- 顶层键名保持 `retrieved`（比 AML 的 `data` 更可读），但**数组内每个 item 的字段名严格对齐
  AML 的 `data[]` item**（`id` / `content` / `score` / `created_at` + 扩展 `source_session_id` / `metadata`）。
- `elapsed_seconds` = 本 target 的 add + search 总墙钟耗时。
- `total_memories_stored` = 你的 `count()` 返回值（未实现则为 `null`）。

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--method` | `demo` | 记忆系统名，同时决定产物目录 `{out_root}/{method}/`。 |
| `--out-root` | `outputs/memory` | 产物根目录。 |
| `--top-k` | `1` | `search()` 返回的最大条数，也就是写进 `memory.txt` 的条数。协议只注入一条排在最前的经验，调大会使结果不再与已报告的数字可比。 |
| `--filter-lang` | 无 | 只跑某语言（`python` / `js` / ...）。 |
| `--limit` | 无 | 最多跑多少个 target（调试用）。 |
| `--overwrite` | off | 结果已存在时强制重跑（默认跳过，支持断点续跑）。 |
| `--run-id` | 自动 `YYYYMMDD-HHMMSS` | 本次运行 ID，编入 `user_id`。 |
| `--max-session-mb` | `64` | 单条历史轨迹体积上限（十进制 MB），超过则跳过。 |

---

## 9. 30 行接入你自己的记忆系统

从 `memory_interface.py` 继承 `MemorySystem` 实现一个子类，存成单独的一个文件
`memory_systems/<你的名字>.py`。harness 按文件名自动发现 method，所以仓库里其他任何
地方都不用改。可以拷 `memory_systems/demo.py` 当起点。下面是一个可直接套用的最小骨架：

```python
from memory_interface import (
    MemorySystem, HistorySession, MemoryScope, RetrievedMemory,
)

class MyMemorySystem(MemorySystem):
    def __init__(self):
        self._db = {}                       # user_id -> 你自己的存储（向量库/图库/...）

    def add(self, session: HistorySession, *, scope: MemoryScope) -> None:
        # session.messages 是剥离后的原始 mini-swe-agent-1.1 message dict 列表。
        # session.meta["issue_number"] 是本条历史的时序位置，想加权 recency 就用它。
        # 建议：用 tool_call_id 把 assistant 的 extra.actions[].command 与随后的
        #       tool.extra.raw_output 配对，再结合 reasoning_content 组织成记忆单元。
        store = self._db.setdefault(scope.user_id, [])      # ← 必须按 user_id 隔离
        for i, msg in enumerate(session.messages):
            text = msg.get("reasoning_content") or msg.get("content") or ""
            if text.strip():
                store.append({"id": f"{session.session_id}#{i}",
                              "content": text,
                              "source_session_id": session.session_id})
        # ⚠️ 返回即代表已可被 search 检索到——不得异步、不得返回 task id。

    def search(self, query: str, *, scope: MemoryScope, top_k: int):
        store = self._db.get(scope.user_id, [])             # ← 只在本 user_id 内检索
        scored = [(self._relevance(query, m["content"]), m) for m in store]
        scored.sort(key=lambda x: x[0], reverse=True)        # ← 必须自己降序排好
        return [RetrievedMemory(id=m["id"], content=m["content"], score=s,
                                source_session_id=m["source_session_id"])
                for s, m in scored[:top_k]]                  # ← 无结果自然返回 []

    def _relevance(self, query: str, content: str) -> float:
        return 0.0  # 换成你的向量/图/LLM 打分

    def count(self):
        return sum(len(v) for v in self._db.values())

# harness 只认这个名字。文件名就是 --method 的取值。
MEMORY_SYSTEM = MyMemorySystem
```

运行：

```bash
# 1) 建记忆，产出 outputs/memory/mymem/<lang>/<iid>/memory.txt
python scripts/run_memory.py --method mymem --filter-lang python --limit 3

# 2) 把这些记忆注入 agent 跑任务
python scripts/run_agent.py --method mymem
```

> `EchoMemorySystem`（`--method demo`）是一个**可运行的最小教学样例**，演示「怎么正确读取轨迹」，
> 检索用朴素 Jaccard 词重合（无 LLM / 无向量）。**它不代表任何真实记忆系统的效果**，仅作接入示范。
