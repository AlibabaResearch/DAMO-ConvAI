<div align="center">

# VibeMemBench

**在真实仓库编码任务上评测 coding agent 的记忆系统。**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Agent](https://img.shields.io/badge/mini--swe--agent-bc85a456-orange)](https://github.com/SWE-agent/mini-swe-agent)
[![Targets](https://img.shields.io/badge/targets-111-informational)](#数据集)
[![Trajectories](https://img.shields.io/badge/history%20trajectories-3634-informational)](#数据集)
[![Languages](https://img.shields.io/badge/languages-9-informational)](#数据集)
[![arXiv](https://img.shields.io/badge/arXiv-2609.23570-b31b1b)](https://arxiv.org/abs/2609.23570)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace%20coming%20soon-yellow)](#数据集) <!-- TODO: 数据集公开后换成真实 Hugging Face 链接 -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Code](https://img.shields.io/badge/Code-DAMO--ConvAI%2FVibeMemBench-blue)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/VibeMemBench)

[English](README.md) | **简体中文**

</div>

---

Coding agent 在真实仓库任务上工作，而持久化记忆系统承诺能跨任务复用经验。VibeMemBench 检验
这些经验是否真能改善可执行的仓库工作。它对同一个任务跑两遍，agent、工具、沙箱和预算全部固定，
只改变记忆这一个条件，最终由可执行测试判定任务是否被解决。

基准包含来自 **90 个 SWE-rebench V2 仓库**的 **111 个编码 target**，并由同仓库的 **3634 条历史
轨迹**支撑。只有当注入的历史经验能在参考设置下改善某个 target 的可执行结果时，该 target 才会被
保留。因此每个 target 都带着一份「有用性已被执行验证」的先验经验。

<p align="center">
  <img src="assets/overview.png" alt="VibeMemBench 总览" width="100%">
</p>
<p align="center"><em>总览。上：离线构建冻结的已验证经验。下：两条已发布的评测轨道。</em></p>

> **核心发现。** 冻结的已验证经验能在五个 held-out solver 中的四个上把任务解决率提升 1.1 到 4.5
> 个百分点，并在全部五个上降低 agent 步数。然而当四个现有记忆系统必须从同样的历史里自行构建并检索
> 经验时，十二个「solver × 系统」组合里有十一个打不过对应的「关记忆」基线。差距就在于：仓库历史里
> 存在有用的经验，而现有记忆系统交付出来的经验却不对。

## 目录

- [本仓库提供什么](#本仓库提供什么)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [评测管线](#评测管线)
- [接入你自己的记忆系统](#接入你自己的记忆系统)
- [主要结果](#主要结果)
- [致谢](#致谢)
- [论文](#论文)
- [引用](#引用)

## 本仓库提供什么

VibeMemBench 由两部分组成，本仓库同时提供这两部分。

| 部分 | 位置 | 作用 |
| --- | --- | --- |
| **数据集** | [`benchmark/`](benchmark/) | 111 个 target 与 3634 条历史轨迹。被 git 忽略，通过 Hugging Face 分发。 |
| **评测框架** | [`scripts/`](scripts/) | 三阶段管线：接入记忆系统、运行固定 agent、给结果打分。 |

agent 是 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)，不随本仓库一起发布。按
[快速开始](#快速开始) 里的命令安装固定 commit `bc85a456`，保证每次运行都可复现。打分逻辑是
SWE-rebench 的评估器，已 vendor 到 [`scripts/vendor/`](scripts/vendor/)。

## 仓库结构

```
VibeMemBench/
├── README.md                 # 英文版
├── README_zh.md              # 本文件（中文版）
├── LICENSE                   # Apache 2.0 协议
├── .env.example              # 拷成 .env 并填入模型端点
├── assets/                   # 本 README 用的配图
├── benchmark/                # 数据集，被 git 忽略，通过 Hugging Face 分发
│   ├── README.md             # 数据集说明卡
│   ├── test.parquet          # 111 个 target，SWE-bench 格式
│   └── history/              # 3634 条轨迹，按语言分组
├── scripts/
│   ├── CONTRACT.md           # 接入契约（英文）
│   ├── CONTRACT_zh.md        # 接入契约（中文）
│   ├── memory_interface.py   # MemorySystem 抽象基类与三个数据契约类
│   ├── memory_systems/       # 一个记忆系统一个文件，按文件名自动发现
│   │   └── demo.py           # 样板实现 EchoMemorySystem
│   ├── run_memory.py         # 阶段 1：建记忆，再检索
│   ├── run_agent.py          # 阶段 2：注入记忆并运行 agent
│   ├── run_eval.py           # 阶段 3：给运行结果打分
│   ├── docker_pool.py        # 阶段 2 与阶段 3 共用的镜像生命周期管理
│   └── vendor/               # vendor 进来的 SWE-rebench 评估器
└── outputs/                  # 运行产物，被 git 忽略
```

## 快速开始

**1. 获取代码。**

VibeMemBench 位于 [DAMO-ConvAI](https://github.com/AlibabaResearch/DAMO-ConvAI) 仓库的
[`VibeMemBench/`](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/VibeMemBench) 目录下。用 sparse checkout 只拉取这个目录即可。

```bash
git clone --filter=blob:none --sparse https://github.com/AlibabaResearch/DAMO-ConvAI.git
cd DAMO-ConvAI
git sparse-checkout set VibeMemBench
cd VibeMemBench
```

如果你想拉取整个 DAMO-ConvAI 仓库，正常 clone 后 `cd VibeMemBench` 即可。

**2. 克隆固定 commit 的 agent 并安装。**

```bash
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent
git checkout bc85a45654e6348dcc6e4c5a40ad146ed0bb144d   # v2.2.8-5-gbc85a456
pip install -e .
cd ..
```

框架需要 Python 3.12+，加上 `pandas`、`pyarrow` 与 `PyYAML`。如果希望下面那个 `.env`
能被读到，还需要 `python-dotenv`。为兼容参考环境，请保持 `pyarrow<=17.0.0`、
`pandas<=2.2.3`。

```bash
pip install "pandas<=2.2.3" "pyarrow<=17.0.0" PyYAML python-dotenv
```

**3. 把数据集下载到 `benchmark/`。**

```bash
hf download <org>/VibeMemBench --repo-type dataset --local-dir benchmark
```

`local_dir="benchmark"` 会把文件放到代码正好查找的位置。完整下载选项（含只下某一种语言）见
[`benchmark/README.md`](benchmark/README.md)。

**4. 给 agent 指定模型。** 拷一份示例文件填好，或者自己导出同样的环境变量。

```bash
cp .env.example .env
```

真正要紧的是这三个，由阶段 2 读取：

```bash
MODEL=openai/deepseek-v4-flash
API_BASE=https://your-endpoint/v1
API_KEY=sk-...
```

`.env` 被 git 忽略，而且**只有装了 `python-dotenv` 才会被读**。没装的话这个文件会被跳过，
所以脚本会就此告警，而不是装作已生效。

阶段 2 会在每个 target 各自的 Docker 沙箱镜像里运行，因此需要本机可用 Docker。

## 数据集

数据集位于 `benchmark/`，将发布在 Hugging Face 的
`https://huggingface.co/datasets/<org>/VibeMemBench`（链接待补）。完整参考见数据集说明卡
[`benchmark/README.md`](benchmark/README.md)。

### 数量

111 个 target，覆盖 90 个仓库、9 种语言，由 3634 条已完成的历史轨迹支撑。

| 语言 | Target 数 | 历史轨迹数 |
| --- | --- | --- |
| Python | 56 | 2043 |
| JavaScript | 16 | 503 |
| Go | 12 | 263 |
| Rust | 10 | 185 |
| TypeScript | 5 | 383 |
| Kotlin | 4 | 170 |
| Swift | 4 | 47 |
| Java | 2 | 34 |
| PHP | 2 | 6 |
| **合计** | **111** | **3634** |

target 覆盖 bug 修复、功能请求、接口变更与配置类工作，其中 bug 修复占不到三分之一。

<p align="center">
  <img src="assets/target_profile.png" alt="语言与任务类型分布" width="90%">
</p>
<p align="center"><em>111 个 target 的语言（左）与主要任务类型（右）分布。</em></p>

### Target：`benchmark/test.parquet`

一行一个 target，SWE-bench 格式，共 16 列。发布版对每组字段都施加了可见性边界。

| 分组 | 字段 | 可见性 |
| --- | --- | --- |
| 任务与执行 | `instance_id`、`repo`、`base_commit`、`problem_statement`、`image_name`、`language`、`install_config` | solver 可见 |
| 溯源信息 | `pr_description`、`created_at`、`interface`、`license`、`meta` | 默认不可见 |
| 仅评估器 | `patch`、`test_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS` | 绝不传给 solver |

当 `FAIL_TO_PASS` 与 `PASS_TO_PASS` 里的每个测试都通过时，该次运行判为已解决。

### 历史：`benchmark/history/`

```
benchmark/history/<language>/<repo>/<instance_id>/<instance_id>.traj.json
```

每个 `.traj.json` 是同仓库里某个更早 issue 的一条完整 agent 轨迹，正是记忆系统用来构建经验的
原材料。

### 时间因果规则

一个 target 只能拿到同仓库里**发生在它之前**的 issue 的历史。时序来自 `instance_id` 的尾号，
号小表示更早。

```
把历史 H 喂给 target T  当且仅当  repo(H) == repo(T)  且  number(H) < number(T)
```

这一点很关键，因为历史目录里也躺着 target 自己的轨迹。框架强制执行这条过滤，所以 target 永远
无法检索到它正被考核的那道题的答案。

### 隐私

轨迹数据已就泄漏密钥与个人身份做过审计，不含任何真实的 API key、token、密码或操作者身份。看起来
敏感的值都是公开文档示例或开源仓库元数据。数据可按原样安全共享。

## 评测管线

框架分三个阶段运行。每个阶段读取上一个阶段写出的产物，因此你可以逐阶段单独运行。

| 阶段 | 脚本 | 做什么 | 产出 |
| --- | --- | --- | --- |
| **1. 建记忆** | [`run_memory.py`](scripts/run_memory.py) | 给每个 target 喂其仓库里时间上更早的历史（`add`），再用 problem statement 检索（`search`）。 | `outputs/memory/<method>/<lang>/<iid>/memory.txt` 与 `memory.json` |
| **2. 跑 agent** | [`run_agent.py`](scripts/run_agent.py) | 把检索到的记忆注入后运行 mini-swe-agent，每个 target 跑 4 个 seed。 | `outputs/agent_runs/<lang>/<iid>/<method>/seed<N>/` |
| **3. 打分** | [`run_eval.py`](scripts/run_eval.py) | 用 vendor 的 SWE-rebench 评估器给每个 seed 运行打分。 | `outputs/agent_runs/eval_summary.csv` |

一个 target 的两次运行之间只有记忆条件会变。agent、工具、沙箱镜像、预算和 seed 全部固定。

阶段 2 每次运行只声明一种 method：

| method | 含义 |
| --- | --- |
| `base` | 完全不注入经验块，即对应的「关记忆」基线。 |
| `distractor` | 注入一段固定的、不含任何软件内容的日常句子，走与真实记忆相同的注入规则。用来对照「随便注入点文本也有用」。 |
| `<name>` | 一个记忆系统，如 `demo` 或 `mymem`。它的每条 instance 记忆从阶段 1 写出的 `memory.txt` 读取。 |

用内置 demo 跑一遍最小闭环：

```bash
# 阶段 1：为 demo 系统建记忆并检索
python scripts/run_memory.py --method demo --filter-lang python --limit 3

# 阶段 2：注入该记忆运行 agent（再跑一次 base 作对比）
python scripts/run_agent.py --method demo
python scripts/run_agent.py --method base

# 阶段 3：给所有运行打分
python scripts/run_eval.py
```

### 并发与磁盘

每个 target 都有自己的 Docker 镜像，tag 里带着它的 PR 号与 commit，所以 111 个 target 需要
111 个不同的多 G 镜像，而且**一个都不复用**。不管的话一轮完整运行会把它们全攒在磁盘上。
[`docker_pool.py`](scripts/docker_pool.py) 让拉取领先于执行，并给磁盘常驻数设上限。阶段 2
与阶段 3 共用它，参数名一致。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--pull-workers` | 3 | 同时跑几个 `docker pull`，领先于执行 |
| `--image-workers` | 1 | 同时有几个不同镜像在跑任务 |
| `--seed-workers` | 4 | 一个 target 的几个 seed 同时跑。它们共用镜像但各起单独容器 |
| `--image-pool-size` | 10 | 磁盘常驻镜像上限。设 `0` 则一个也不删 |
| `--pull-timeout` | 1800 | 单次拉取的秒数。agent 自带的 120 秒默认值在最大的镜像上会超时 |

并发容器数的峰值是 `--image-workers × --seed-workers`，所以默认是一个镜像带它的四个
seed 并行跑。

有一笔开销要权衡。淘汰在**单阶段内**是白赚的，因为没有 target 需要其他 target 的镜像；
但**跨阶段不是**：阶段 3 要给同一批 target 打分，需要同一批镜像，所以开了上限的跑法
会把镜像下两遍。磁盘宽裕的话，给阶段 2 传 `--image-pool-size 0`，跑完紧接着打分，
就能省掉第二遍。

## 接入你自己的记忆系统

整个接入面只有两个文件：[`memory_interface.py`](scripts/memory_interface.py) 及其规范
[`CONTRACT_zh.md`](scripts/CONTRACT_zh.md)（[English](scripts/CONTRACT.md)）。你只需实现 `add`
与 `search`。检索结果如何注入、如何打分都由平台固定，与你无关。

```python
class MemorySystem(abc.ABC):
    def add(self, session: HistorySession, *, scope: MemoryScope) -> None: ...
    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> List[RetrievedMemory]: ...
    def count(self) -> Optional[int]: ...   # 可选
```

### 第一步：加一个文件

一个 method 就是 `scripts/memory_systems/` 下的一个文件，**文件名就是 method 名**。想让
`--method mymem` 能用，就新建 `scripts/memory_systems/mymem.py`。仓库里其他任何地方都不用改，
你永远不需要动 harness。

```python
# scripts/memory_systems/mymem.py
from memory_interface import HistorySession, MemoryScope, MemorySystem, RetrievedMemory


class MyMemorySystem(MemorySystem):
    def __init__(self) -> None:
        # harness 为每个 target 新建一个实例，不传参。
        # 需要配置就读环境变量，或写你自己的默认值。
        self._store = {}

    def add(self, session: HistorySession, *, scope: MemoryScope) -> None:
        # 只能写在 scope.user_id 命名空间内。函数返回即代表已可被检索到。
        self._store.setdefault(scope.user_id, []).append(session)

    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> list[RetrievedMemory]:
        # 只在 scope.user_id 内检索，自己排好序，最多返回 top_k 条。
        return [RetrievedMemory(id="m1", content="...", score=1.0)]


# harness 只认这个名字。
MEMORY_SYSTEM = MyMemorySystem
```

建议拷 [`memory_systems/demo.py`](scripts/memory_systems/demo.py) 当起点，它本身能跑，并且演示了
怎么正确读一条轨迹（把 assistant 的命令与紧随其后的 tool 输出配对）。

### 第二步：确认已被识别

```bash
cd scripts && python -c "from memory_systems import available_methods; print(available_methods())"
# ['demo', 'mymem']
```

如果名字没出现，或者 `MEMORY_SYSTEM` 缺失、不是 `MemorySystem` 的子类，harness 会直接报错并
告诉你怎么改。`base` 与 `distractor` 是阶段 2 的保留名，不会到这里找文件。

### 第三步：跑三个阶段

```bash
python scripts/run_memory.py --method mymem     # 产出 outputs/memory/mymem/...
python scripts/run_agent.py  --method mymem     # 注入它，每个 target 4 个 seed
python scripts/run_agent.py  --method base     # 用作对照的基线
python scripts/run_eval.py                     # 两边一起打分
```

建议先小规模跑：阶段 1 加 `--filter-lang python --limit 3`，阶段 2 加 `--dry-run`。dry-run
不碰 Docker 也不发 API 请求，但会把完整的命令和完整的注入内容都写出来。

### 两条由 harness 为你强制保证的规则

- **因果过滤**。一个 target 只会收到同仓库里**更早**的 issue 历史，时序键是 `instance_id`
  的尾号。你**不可能**收到未来的轨迹，所以不需要自己过滤。两个编号以
  `session.meta["issue_number"]` 与 `session.meta["target_issue_number"]` 给出，可用来加权 recency。
- **金标红线**。target 的 `patch`、`test_patch`、`pr_description` 绝不会出现在喂给你的数据里，
  也不得出现在你返回的结果里。harness 在写盘前会校验这一点。

内置的 `EchoMemorySystem`（`--method demo`）是一个可运行的教学样例，检索用朴素 Jaccard，
不调任何模型。它不是一个有竞争力的记忆系统。

## 主要结果

### 冻结的已验证经验迁移

Resolved 是 444 次「target × seed」运行（111 个 target，四个 seed）中声明测试通过的占比。95% CI
给出「相对关记忆行的成对 Resolved 差值」的 percentile bootstrap 区间，单位为百分点。4/4 列统计在
每个 seed 上都通过的 target 数。括号里是相对关记忆行的变化。

| Solver | 条件 | Resolved | 95% CI | 4/4 | 输入 / 次 | 输出 / 次 | 步数 / 次 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-pro | 关记忆 | 67.1% | — | 49 | 1.67M | 19.1k | 50.3 |
| deepseek-v4-pro | 冻结的已验证经验 | 67.1% (+0.0) | [-5.63, +5.86] | 51 (+2) | 1.49M (-0.18M) | 18.9k (-0.2k) | 47.6 (-2.7) |
| deepseek-v4-pro | 无关记忆对照 | 63.7% (-3.4) | [-8.11, +1.35] | 46 (-3) | 1.51M (-0.16M) | 18.1k (-1.0k) | 47.1 (-3.2) |
| glm-5 | 关记忆 | 55.0% | — | 35 | 2.45M | 15.4k | 77.1 |
| glm-5 | 冻结的已验证经验 | 57.4% (+2.4) | [-2.25, +7.21] | 34 (-1) | 1.93M (-0.52M) | 13.2k (-2.2k) | 66.2 (-10.9) |
| glm-5 | 无关记忆对照 | 54.3% (-0.7) | [-5.63, +4.05] | 36 (+1) | 2.13M (-0.32M) | 14.5k (-0.9k) | 69.3 (-7.8) |
| glm-5.2 | 关记忆 | 76.8% | — | 67 | 4.06M | 45.9k | 68.2 |
| glm-5.2 | 冻结的已验证经验 | 80.4% (+3.6) | [-0.90, +8.11] | 74 (+7) | 3.40M (-0.67M) | 39.3k (-6.6k) | 61.0 (-7.2) |
| kimi-k2.7-code | 关记忆 | 69.8% | — | 56 | 1.56M | 16.2k | 56.0 |
| kimi-k2.7-code | 冻结的已验证经验 | 74.3% (+4.5) | [-0.23, +9.46] | 62 (+6) | 1.57M (+0.01M) | 16.4k (+0.2k) | 53.8 (-2.2) |
| kimi-k2.7-code | 无关记忆对照 | 69.1% (-0.7) | [-5.18, +3.83] | 57 (+1) | 1.62M (+0.06M) | 16.6k (+0.4k) | 55.9 (-0.1) |
| qwen3.8-max | 关记忆 | 79.1% | — | 68 | 4.03M | 55.2k | 52.9 |
| qwen3.8-max | 冻结的已验证经验 | 80.2% (+1.1) | [-2.93, +5.41] | 70 (+2) | 4.00M (-0.03M) | 60.3k (+5.1k) | 50.6 (-2.3) |

无关记忆对照保持注入规则不变，但注入一段不含软件内容的干扰文本。在跑该对照的三个 solver 上，它都
低于关记忆基线，说明「光注入文本」复现不出任何增益。glm-5.2 与 qwen3.8-max 没有对照行。

### 现有记忆系统的检索

下面这四个系统是论文评测的对象，**本仓库并不提供它们**，`scripts/memory_systems/` 下也没有
它们任何一个。想复现这几行，需要自己给每个系统写一个适配，做法和接你自己的系统一样。

定义同上。括号里是相对上面同 solver 关记忆行的变化。

| Solver | 系统 | Resolved | 95% CI | 输入 / 次 | 输出 / 次 | 步数 / 次 |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-pro | Mem0 | 62.8% (-4.3) | [-9.68, +1.35] | 1.75M (+0.08M) | 20.3k (+1.2k) | 51.2 (+0.9) |
| deepseek-v4-pro | SimpleMem | 66.9% (-0.2) | [-4.95, +4.50] | 1.54M (-0.13M) | 18.8k (-0.3k) | 48.7 (-1.6) |
| deepseek-v4-pro | MemoryOS | 64.6% (-2.5) | [-8.11, +3.15] | 1.70M (+0.03M) | 19.4k (+0.3k) | 50.9 (+0.6) |
| deepseek-v4-pro | A-MEM | 63.1% (-4.0) | [-10.14, +2.03] | 1.71M (+0.04M) | 20.0k (+0.9k) | 50.0 (-0.3) |
| glm-5 | Mem0 | 49.5% (-5.5) | [-10.59, -0.45] | 2.17M (-0.28M) | 14.7k (-0.7k) | 71.7 (-5.4) |
| glm-5 | SimpleMem | 53.2% (-1.8) | [-6.76, +2.93] | 2.02M (-0.43M) | 13.3k (-2.1k) | 70.2 (-6.9) |
| glm-5 | MemoryOS | 57.0% (+2.0) | [-3.38, +7.43] | 1.96M (-0.49M) | 13.3k (-2.1k) | 67.2 (-9.9) |
| glm-5 | A-MEM | 52.5% (-2.5) | [-8.56, +3.38] | 2.01M (-0.44M) | 14.3k (-1.1k) | 66.5 (-10.6) |
| kimi-k2.7-code | Mem0 | 68.0% (-1.8) | [-6.53, +2.70] | 1.61M (+0.05M) | 17.6k (+1.4k) | 57.0 (+1.0) |
| kimi-k2.7-code | SimpleMem | 69.6% (-0.2) | [-4.95, +4.50] | 1.62M (+0.06M) | 17.2k (+1.0k) | 56.1 (+0.1) |
| kimi-k2.7-code | MemoryOS | 66.0% (-3.8) | [-8.56, +1.13] | 1.60M (+0.04M) | 16.7k (+0.5k) | 53.9 (-2.1) |
| kimi-k2.7-code | A-MEM | 65.8% (-4.0) | [-9.68, +1.35] | 1.66M (+0.10M) | 16.4k (+0.2k) | 55.0 (-1.0) |

十二个组合里有十一个持平或低于对应的关记忆基线。唯一的例外是 glm-5 上的 MemoryOS，提升 2.0 个
百分点，但仍不及冻结的已验证经验。

## 致谢

VibeMemBench 的仓库任务构建于 [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench)，
固定 agent 采用 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)。agent 设计见
[SWE-agent 论文](https://arxiv.org/abs/2405.15793)。接入契约遵循
[Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/api-guide) 的职责边界。

## 论文

VibeMemBench 的论文已发布在 arXiv 上。

- arXiv：[2609.23570](https://arxiv.org/abs/2609.23570)

## 引用

```bibtex
@article{fan2026vibemembench,
  title   = {VibeMemBench: Evaluating Memory Systems for Coding Agents on Real Repository Coding Tasks},
  author  = {Fan, Liyang and Shi, Yingcheng and Li, Yongbin and Sun, Chenghao and Chen, Xin and Xu, Xander and Wei, Hu and Ni, Shiwen and Yang, Min and Ye, Jieping},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv:2609.23570}
}
```
