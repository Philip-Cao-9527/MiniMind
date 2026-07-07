# MiniMind 项目级 Agent 指令

本文件只约束当前个人复现仓库 MiniMind。若与全局偏好冲突，以更具体、更贴近当前仓库真实状态的规则为准。

## 0. 执行环境前置规则

- 默认操作系统是 Windows 11 + WSL2 Ubuntu；从仓库根目录执行时，项目路径记作 `.`，默认 shell 是 Linux Bash。
- 执行命令、测试或脚本前，先用本轮真实命令确认当前目录、Git 状态、Python 解释器和虚拟环境；不要沿用历史假设。
- Windows PowerShell、Linux Bash、macOS zsh/bash 的命令语法、路径分隔符、环境变量写法、虚拟环境路径不同。MiniMind 当前默认使用 Linux Bash；不要把 Bash heredoc、`source`、`/` 路径习惯或 shell 语法直接复制到 PowerShell。
- 本项目使用 Python 和 PyTorch。优先使用 direnv 激活后的 `python`；在非交互命令未自动加载 direnv 时，使用 `direnv exec . python ...` 或显式 `./.venv/bin/python ...`。
- 当前 [.envrc](.envrc) 只设置项目本地虚拟环境路径和 `PATH_add "$VIRTUAL_ENV/bin"`。不要在 [.envrc](.envrc) 中写入 token、密码、私有代理地址或其他敏感信息。
- 跨平台文件读写必须显式考虑 UTF-8。涉及 Python 时，可按项目需要启用 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1` 或 `python -X utf8`。
- 修改文本文件前确认原文件编码和换行风格；发现乱码时先判断是读取方式问题还是文件内容损坏，再继续修改。

## 1. 语言与表达

- 默认使用简体中文回复。
- 代码、命令、报错、路径、字段名、域名、库名和专有名词可以保留英文原文。
- 不要把未验证内容写成已验证事实，不要把推测写成事实。
- 面向用户的说明要自然、清楚、可执行；不要写成调试日志、机器翻译或内部变量说明。

## 2. 项目定位

- 本项目类型：围绕上游 MiniMind 的个人 LLM 源码理解、关键模块手写复现、最小实验验证与面试复盘仓库。
- 项目主目标：看懂当前 MiniMind 主线源码，解释模型、数据、训练、推理之间的调用关系，手写并对照复现关键模块，用最小实验验证真实行为。
- 核心边界：本仓库不是上游 MiniMind 的机械镜像，不把上游 README、教程话术、示例成果或未运行实验写成个人成果。
- 资料优先级：当前本地仓库事实优先；其次是个人 GitHub 远端；再核验上游 `jingyaogong/minimind`；`learn-minimind` 只用于学习顺序、理解辅助和面试追问准备。
- 当前仓库根目录真实可见内容以本轮命令为准。当前已存在的长期项目文件包括 [README.md](README.md)、[.gitignore](.gitignore)、[.envrc](.envrc)、[AGENTS.md](AGENTS.md)、[docs/minimind-roadmap.md](docs/minimind-roadmap.md)、[docs/minimind-source-guide.md](docs/minimind-source-guide.md) 和 [docs/experiment-tiny-pretrain-one-batch-2026-07-04.md](docs/experiment-tiny-pretrain-one-batch-2026-07-04.md)。MiniMind 专用 personal skills 已移动到 `../../.codex/skills/minimind-*`，例如 [minimind-code-reviewer/SKILL.md](../../.codex/skills/minimind-code-reviewer/SKILL.md)、[minimind-knowledge-explainer/SKILL.md](../../.codex/skills/minimind-knowledge-explainer/SKILL.md)、[minimind-plan-mode-planner/SKILL.md](../../.codex/skills/minimind-plan-mode-planner/SKILL.md)、[minimind-prompt-creator/SKILL.md](../../.codex/skills/minimind-prompt-creator/SKILL.md)、[minimind-skill-creator/SKILL.md](../../.codex/skills/minimind-skill-creator/SKILL.md)、[minimind-web-search/SKILL.md](../../.codex/skills/minimind-web-search/SKILL.md)。阶段性报告统一从 `docs/` 中按任务相关性读取，不固定写入本规则。
- 当前 `model/`、`trainer/`、`dataset/` 下已有本地源码文件，`experiments/` 下已有极小实验材料，`scripts/`、`tests/` 目录也已出现；这些内容当前仍有未提交改动或未跟踪文件。引用它们时必须先确认本轮 Git 状态、文件内容和验证记录，不要把上游源码或未验证实验默认写成本仓库已完成基线。

## 3. 顶层代码生成约束

- 默认做最小必要改动，但“最小”不是盲目保守。若面向真实用户需求、解决反复多次仍未修好的 bug，或为了贴合真实工程实践，允许进行更大范围重构。
- 大改前必须说明根因、必要性、影响范围、回归方案和可回滚边界。
- 代码生成必须以真实仓库结构、真实调用链、真实测试入口为依据，不要凭记忆或参考项目路径编造实现。
- 禁止为了显得完整而堆空壳、堆模板、堆不可验证约束。
- 禁止一次性生成超长单文件代码。复杂逻辑应按职责边界拆成模块、函数或配置，保证可审查、可测试、可回滚。
- 不要复制粘贴大段近似逻辑；出现重复分支时，优先抽取明确 helper、配置表或领域模块。
- 新增文件、目录、测试和报告命名要体现职责与场景，避免只有时间戳、缩写或模糊命名。
- 涉及模型、数据、训练或推理行为时，必须区分源码阅读结论、个人手写实现、真实实验结果和后续计划。

## 4. 修改前必读

开始任何修改前，至少先读与本轮任务直接相关的最小上下文：

- 项目总览：[README.md](README.md)。
- 项目规则：[AGENTS.md](AGENTS.md)。
- Git 忽略规则：[.gitignore](.gitignore)。
- 本地环境入口：[.envrc](.envrc)。
- 后续推进规划：[docs/minimind-roadmap.md](docs/minimind-roadmap.md)。
- 项目文档与报告：阅读 `docs/` 下与本轮任务相关的 Markdown 文档；修复报告是阶段性证据，不要把某个具体报告固定为长期必读项。
- MiniMind 专用 personal skills：优先读取本轮相关的 `../../.codex/skills/minimind-*` 下对应 `SKILL.md`、`references/` 和 `scripts/`；这些 skill 服务本项目，但存放在个人 Codex 全局目录，便于从 `/home/harry` 工作区直接调用。
- 上游参考源码：优先使用已同步的 `../../references/minimind`，并确认其 Git 状态和远端提交；本地引用缺失或过期时再联网核验。
- 学习辅助材料：`../../references/learn-minimind` 只作为学习路线与追问准备辅助，不作为个人实验结果来源。
- 本轮相关入口、调用链、配置、脚本：例如上游 [model/model_minimind.py](../../references/minimind/model/model_minimind.py)、[dataset/lm_dataset.py](../../references/minimind/dataset/lm_dataset.py)、[trainer/train_pretrain.py](../../references/minimind/trainer/train_pretrain.py)、[trainer/train_full_sft.py](../../references/minimind/trainer/train_full_sft.py)、[trainer/trainer_utils.py](../../references/minimind/trainer/trainer_utils.py)、[eval_llm.py](../../references/minimind/eval_llm.py)、[scripts/chat_api.py](../../references/minimind/scripts/chat_api.py)、[scripts/serve_openai_api.py](../../references/minimind/scripts/serve_openai_api.py) 等；只有本仓库实际存在或已明确从上游同步后才能按本仓库文件引用。

原则：先理解入口、调用链、数据结构、依赖顺序和验证方式，再动手改文件。不要用过期文件清单替代当前仓库事实。

## 5. 文档与报告目录规则

- [README.md](README.md) 面向项目展示、长期目标和当前边界，不写未验证实验成果。
- `docs/` 面向源码理解、模块拆解、实验记录、问题复盘、修复报告和阶段计划。当前不区分 `docs/experiments/` 与 `docs/reports/` 子目录，新增 Markdown 文档默认直接放在 `docs/` 下，并按主题命名。
- 实验记录和修复报告也放在 `docs/` 下，并通过文件名区分用途，例如 `experiment-*.md`、`fix-report-*.md`、`source-map.md`、`call-chain-*.md`。
- 新增文档前先确认是否已有合适位置；不要把一次性过程记录塞进长期用户文档，也不要把长期规则藏在临时报告里。
- 生成文档时，本仓库内已存在的具体文件必须使用可跳转 Markdown 相对链接；仓库外但位于本机的具体参考文件也必须使用相对链接，并按当前 Markdown 文件所在目录计算目标路径。例如 `docs/` 下链接到上游引用时使用 `../../../references/...`。
- 具体证据优先链接到真实文件名；能定位到行号时使用 `[文件名](相对路径#L行号)`，不要使用 `[文件名](相对路径:行号)`，也不要把本机绝对路径写进 Markdown 链接目标。
- 不要将文件夹、目录通配符或拟新增路径作为 Markdown 超链接。目录路径、`*.py` 这类通配符、尚不存在的计划文件一律用代码格式；如果证据来自某个目录，应链接到该目录下的具体文件，或保留目录代码路径并说明需要后续展开。
- 拟新增文件必须用代码格式标记为“拟新增”；未运行、未生成或尚未确认存在的材料不能写成可跳转证据链接。

## 6. 修复报告规则

修复报告只用于记录核心代码、可执行能力、评测闭环或项目行为的实质变化。纯文档修改、README 更新、说明文字修正、技能 / 指令文件调整、格式整理、注释修正，默认不触发修复报告，也不触发版本号变更。

只有本次改动涉及以下任一类型，完成后才需要在 `docs/` 下新增或更新修复报告；用户明确要求生成报告时除外：

- 功能修复或行为变更。
- 核心代码、运行入口、任务契约、数据结构、接口协议或可执行能力变化。
- 测试、评测、验证闭环、发布流程或真实服务调用方式变化。
- 影响项目架构分层、目录结构、用户可见行为、权限 / 隐私 / 审核风险的变化。
- 用户明确要求生成修复报告。

报告命名规则可以保留占位符，生成报告时必须替换为本轮实际目录、版本、日期和主题：

```text
{{报告目录}}/fix-report-v{{版本号}}-{{日期}}.md
{{报告目录}}/fix-report-v{{版本号}}-{{日期}}-{{主题slug}}.md
```

若用户指定报告路径，可按指定路径生成；不要把某个历史报告写成长期固定路径。

报告至少覆盖：

1. 本轮问题 / 目标与范围。
2. 改动文件清单。
3. 关键修复内容。
4. 验收方式 / 手测步骤 / 自动化测试情况。
5. 版本同步清单。
6. 风险与备注。
7. 结论。

生成报告时必须使用可跳转的 Markdown 相对路径交叉引用。链接优先落到具体文件名，能定位到行号时必须使用 `[文件名](相对路径#L行号)` 范式；不要把 `:行号` 写进链接目标里。不要只写文件夹名代替关键证据，也不要使用当前 IDE 无法跳转的绝对路径，必须强制使用相对路径。

如果本轮不触发修复报告，也要在最终总结中明确说明“不触发报告”的原因。若用户明确要求生成报告，即使是文档或指令文件治理，也必须按用户指定路径生成。

## 7. 版本号演进规则

- 当前已知版本：[README.md](README.md) 徽章标注 `v0.0.1`。
- 当前仓库尚无独立版本文件。创建正式版本机制前，不要编造 `VERSION`、`pyproject.toml` 或其他不存在的版本来源。
- 版本占位符用于升版和报告生成时统一替换：`{{当前版本}}` 当前对应 `v0.0.1`；`{{版本文件}}` 当前对应 [README.md](README.md)，因为尚无独立版本文件；`{{README版本位置}}` 当前对应 [README.md](README.md) 顶部徽章。
- 版本起点：以 [README.md](README.md) 中当前可核验版本为准。
- 其他版本位置：提交前用 `rg -n "v[0-9]+\\.[0-9]+\\.[0-9]+|version" README.md .` 核验，不要静默跳过。

版本含义：

- `PATCH`：小范围 bug fix、小范围行为修正、局部测试改进或不改变核心能力的兼容性修复。
- `MINOR`：新增可用能力、新增较完整工作流、增加用户可见功能或扩展可执行能力。
- `MAJOR`：明显改变核心调用链、接口契约、数据格式、发布方式或旧用法需要迁移。

默认规则：

- 用户没有明确要求 bump 时，默认在当前版本继续修改。
- 纯文档、说明文字、技能 / 指令文件、格式整理、注释修正默认不修改版本号。
- 不要为了显得进展大而随意升版；版本号必须对应真实改动。
- 每次修改项目版本号时，必须同步检查所有写有当前版本号的位置，包括 [README.md](README.md)、修复报告文件名、用户可见版本显示、包管理配置或其他项目实际版本位置。
- 如果某些版本位置历史上不一致，必须说明处理方式，不能静默跳过。

## 8. MiniMind 保护逻辑与错误处理原则

允许必要的安全边界，但不要为了“看起来更稳”而加入没有依据的保护逻辑。本节只服务 MiniMind 的模型、数据、训练、推理、文档和项目级 skill 工作。

禁止无依据新增的典型保护逻辑包括但不限于：固定超时、长度截断、条数上限、重试上限、静默降级、隐藏兜底、broad try/catch 后吞异常、失败后伪造成功或返回空结果冒充正常。

MiniMind 中常见保护边界包括：

- 中断条件：训练 OOM、NaN、数据格式错误、checkpoint 不匹配、权重缺失、设备不可用。
- 容量边界：RTX 5060 Laptop 约 8GB 显存、batch size、max sequence length、梯度累积、checkpoint 和数据文件大小。
- 输入输出限制：JSONL schema、tokenizer 特殊 token、labels 中 `-100` 的含义、padding 与 attention mask 的一致性。
- 重试 / 轮询策略：下载模型或数据失败时先保留错误证据，不盲目切换 CUDA、驱动或 PyTorch。
- 降级 / 回退策略：从 GPU smoke test 降级到 CPU 静态检查时必须说明未验证 CUDA 路径。
- 异常捕获策略：训练、推理和数据解析错误应显式暴露关键上下文，不吞异常。
- 默认值策略：训练参数、权重路径、数据路径必须来自当前脚本或文档，不照搬上游多卡 3090 / A100 参数。

只有存在以下依据之一时，才允许新增保护逻辑：

1. 用户明确要求。
2. 协议、平台、运行环境或第三方服务存在客观限制。
3. MiniMind 当前代码或项目文档已有同类约定，本轮只是沿用并保持一致。
4. 真实故障证据或测试结果证明不加会稳定导致故障、卡死、数据损坏、安全问题或严重体验问题。

新增保护逻辑后，必须说明：

1. 依据是什么。
2. 触发时用户、日志或调用方能看到什么。
3. 可能误伤哪些合法输入、输出、数据规模或慢路径。
4. 如何验证，如何记录风险，后续如何调整。

错误处理应优先显式暴露问题、便于排查，而不是隐藏错误。禁止吞异常、伪造成功、把真实失败包装成正常空结果，除非项目协议明确要求且已经记录依据和可观测方式。

## 9. 测试与验证

- 不要把“代码已写”当作“功能已完成”。
- 当前项目最小环境验证命令：

```bash
git status --short --branch
direnv exec . python -V
direnv exec . python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")
PY
```

- 按本轮实际改动选择验证方式：文档改动至少运行结构校验或人工一致性检查；Python 代码改动至少运行相关脚本、单测或最小 smoke test；训练相关改动必须能闭环单 batch 前向、loss、反向和参数更新。
- 验证证据可以包括本轮命令输出、日志、截图、trace、报告、构建产物、静态检查结果、解析校验结果或人工验证记录。
- 涉及模型、数据、训练、推理时，优先用极小数据、极小 batch、短序列和少步数验证语义；RTX 5060 Laptop 约 8GB 显存下不要直接照搬上游完整训练参数。
- 如需示例，可参考但不限于数据解析、模型前向、训练循环、checkpoint 恢复、推理生成、文档校验等场景；示例不代表所有任务默认必须运行。
- 不适用的风险段落应删除，不要保留空标题。
- 如果验证不可用，必须说明不可用的命令、原因、替代验证和仍未验证的边界。
- 历史测试结果不能写成本轮验证结果；本轮验证必须有本轮命令、日志、截图、trace、报告或明确的静态检查证据。

## 10. 项目专项约束

### 上游同步与来源标注

- 适用条件：同步、复制、改写或解释上游 MiniMind 代码时启用。
- 必须确认：当前本地仓库、个人远端、上游远端和 `../../references/minimind` 的 Git 状态。
- 禁止事项：不要把上游已有实现写成个人原创成果；不要把未同步到本仓库的上游文件写成本仓库事实。
- 验证方式：使用 `git status --short --branch`、`git remote -v`、`git rev-parse HEAD`、必要时 `git fetch --prune`。
- 证据要求：在总结中列出提交哈希、文件路径和未验证边界。

### 模型与张量语义

- 适用条件：修改或解释 embedding、RMSNorm、Attention、RoPE、MLP、MoE、KV Cache、loss 或生成逻辑时启用。
- 必须确认：输入输出 shape、dtype、device、mask 语义、labels shift、`ignore_index=-100`、KV Cache 开关和停止条件。
- 禁止事项：不要只用“能跑通”代替语义验证；不要用上游 README 的效果描述代替本机实验。
- 验证方式：优先写最小张量级测试，再做单 batch 前向、loss、反向和参数更新闭环。
- 证据要求：测试文件、命令输出、关键张量 shape 和结论必须可追溯。

### 数据、tokenizer 与 mask

- 适用条件：处理上游或本仓库的 `dataset/`、JSONL、tokenizer、chat template、labels、loss mask、padding 或截断时启用。
- 必须确认：样本字段、特殊 token、assistant 区间、padding token、attention mask 与 labels 中 `-100` 的关系。
- 禁止事项：不要把预训练数据、SFT 数据、DPO/RLAIF 数据的标签规则混为一谈。
- 验证方式：用一到两个最小样本打印 token、label、mask 对齐关系，并记录哪些 token 参与 loss。
- 证据要求：最小样本、解析脚本或测试输出保存到 `docs/` 下的实验记录或测试日志摘要。

### 训练与显存边界

- 适用条件：运行训练、调整 batch size、max sequence length、dtype、梯度累积、checkpoint 或恢复训练时启用。
- 必须确认：RTX 5060 Laptop GPU 约 8GB 显存、PyTorch CUDA 可用性、当前数据大小、权重路径和保存路径。
- 禁止事项：不要照搬上游多卡 3090、A100 或完整训练参数；不要承诺未经实际运行验证的吞吐、loss、模型效果或性能提升。
- 验证方式：先做极小 smoke test，再逐步增加 batch、序列长度或训练步数；OOM 后记录参数组合与错误，不盲目重装环境。
- 证据要求：命令、配置、显存现象、loss 摘要、checkpoint 路径和未验证边界。

### 推理与生成

- 适用条件：修改或验证上游或本仓库的 `eval_llm.py`、聊天脚本、OpenAI API 服务、采样参数、KV Cache 或停止条件时启用。
- 必须确认：权重来源、tokenizer 来源、prompt 模板、`max_new_tokens`、temperature、top-p/top-k、EOS 与 history 设置。
- 禁止事项：不要把一次随机输出当作模型能力证明；不要把无权重环境下的静态阅读写成推理验证。
- 验证方式：最小 prompt、固定随机种子或明确随机性说明，对比 cache on/off 或采样参数差异时保存可复现条件。
- 证据要求：命令、prompt、关键参数、输出摘要和未验证边界。

### MiniMind skills 与 Agent 规则

- 适用条件：修改 `../../.codex/skills/minimind-*`、[AGENTS.md](AGENTS.md)、提示词模板、校验脚本或项目协作规则时启用。
- 必须确认：skill 名称、触发名、`SKILL.md`、`references/`、`scripts/` 和 `agents/openai.yaml` 是否一致。
- 禁止事项：不要保留旧模板触发名、旧模板提示语或未解释占位符；除报告命名规则中明确允许的占位符外，不要保留未解释的 `{{...}}`；不要把通用 skill 伪装成 MiniMind 项目级 skill。
- 验证方式：运行现有校验脚本、检索模板残留、检查 Markdown/YAML 可读性。
- 证据要求：列出命令、输出摘要、涉及文件和仍未验证的 UI 调用边界。

## 11. Git 与用户改动

- 不要回滚用户已有改动；遇到不相关 dirty 文件，记录并避开。
- 不要自动提交或 push，除非用户明确要求。
- 不要修改 Git 全局配置。
- 不要提交密钥、token、`.env`、本地虚拟环境、缓存目录、下载权重、大型数据集或无关生成文件。
- [.gitignore](.gitignore) 当前忽略 `.venv/`、`out/`、`checkpoints/`、常见本地数据文件、权重和实验平台缓存；提交前必须复查是否有大文件或敏感文件。
- 高风险删除、迁移、恢复操作必须先只读体检，再说明方案和备份边界。

## 12. 输出与验收格式

最终总结尽量按以下顺序组织：

1. 文件改动清单：逐文件说明改了什么。
2. 运行方法 / 验证方式：列出实际运行命令，不把未运行命令写成已运行。
3. 证据路径：截图、日志、trace、报告、构建产物或命令输出摘要；生成报告时必须使用可跳转 Markdown 相对路径交叉引用，链接优先落到具体文件名，能定位到行号时统一使用 `[文件名](相对路径#L行号)`，不要把 `:行号` 写进链接目标。目录路径、通配符和拟新增文件用代码格式表达，不作为超链接。
4. 问题与修复闭环：说明定位、修复、复测和结果。
5. 版本同步清单：说明是否升级了版本，检查了哪些版本位置。
6. 修复报告路径：说明是否新增 / 更新报告；不触发时说明原因。
7. 最终结论：是否达标，仍有什么边界风险或未验证项。

## 13. 进度播报格式

在执行命令、读写文件、测试页面、查看日志时，使用简洁中文进度块：

> 🧩 步骤：{一句话描述正在做什么}
> 🎯 目的：{为什么要做}
> ▶️ 执行：{命令、页面、文件路径或操作}
> ✅ 结果：{当前状态}
> 🧾 证据：{可验证证据路径}
> 📝 备注：{可选，最多一句}

## 14. 默认协作风格

- 优先给出可直接落地的改法。
- 优先给出能闭环真实问题的可行方案；如果最小方案会导致功能割裂、不符合业务实践或留下反复返工风险，要明确说明并给出更合适的调整范围。
- 发现风险要明确指出，不要模糊带过。
- 不为了省事跳过测试。
- 不把未验证写成已验证。
- 不把推测写成事实；有证据给证据，没证据就明确说明。
