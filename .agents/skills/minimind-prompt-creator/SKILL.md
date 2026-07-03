---
name: minimind-prompt-creator
description: 按 MiniMind 的 AGENTS.md、README、roadmap 和真实仓库状态，生成可直接交给 Codex 执行的中文任务 prompt。用于 MiniMind 源码同步、模块复现、数据/标签验证、训练 smoke test、推理实验、文档沉淀和项目工具治理；不用于通用项目。
---

# minimind-prompt-creator

## 技能定位

生成 MiniMind 项目的一次性执行 prompt。它的产物是可复制给 Codex/Agent 的中文任务指令，不直接执行任务。

## 必读材料

1. `AGENTS.md`
2. `README.md`
3. `docs/minimind-roadmap.md`
4. 与本轮任务相关的本地文件、上游引用文件或实验记录
5. 必要时调用 `$minimind-web-search` 核验外部最新事实

## 使用流程

1. 确认用户要生成 prompt，而不是直接执行。
2. 判断任务类型：源码同步、源码地图、数据语义、模型模块、训练闭环、GPU smoke test、推理生成、SFT/LoRA/蒸馏、文档复盘、skill 治理。
3. 读取 `references/prompt-template.md`。
4. 写清 MiniMind 固定边界：本机事实、上游事实、个人成果、后续计划必须区分。
5. 写清必做文件、禁止范围、验证命令、交付文档和不触发版本/报告时的说明。
6. 运行 `scripts/validate_project_prompt.py` 校验生成的 prompt。

## 交叉引用

- 需要先计划时引用 `$minimind-plan-mode-planner`。
- 需要审查时引用 `$minimind-code-reviewer`。
- 需要讲解概念或面试表达时引用 `$minimind-knowledge-explainer`。
- 需要外部事实时引用 `$minimind-web-search`。
- 需要创建新项目级流程时引用 `$minimind-skill-creator`。

## 质量要求

- prompt 必须能直接执行，不写成建议清单。
- 必须包含文件路径、验证命令、证据路径和未验证边界。
- 不得要求重装已验证可用的 WSL、CUDA、PyTorch 或虚拟环境。
- 不得照搬上游多卡 3090/A100 参数。
- 不得把未运行实验写成个人成果。
- 最终 prompt 必须用单个 Markdown 文本块包裹。

## 自检

1. 是否绑定 MiniMind 仓库 `/home/harry/projects/MiniMind`。
2. 是否读取项目规则和 roadmap。
3. 是否把 MiniMind 的真实任务链路写清。
4. 是否包含最小验证和证据路径。
5. 是否没有泛化到无关项目。
