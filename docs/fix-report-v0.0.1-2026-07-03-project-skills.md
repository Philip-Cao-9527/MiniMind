# fix-report-v0.0.1-2026-07-03-project-skills

## 1. 本轮问题 / 目标与范围

本轮目标是治理 MiniMind 仓库的项目级协作规则和项目级 skills，避免通用模板污染项目上下文。

已处理范围：

- 生成 MiniMind 项目级规则文件：[AGENTS.md](../AGENTS.md)。
- 生成项目推进规划：[minimind-roadmap.md](minimind-roadmap.md)。
- 将可在 Codex 中调用的项目级 skills 放到官方 repo-scoped 路径：[.agents/skills](../.agents/skills)。
- 将通用 `code-note-helper` 模板改写为 MiniMind 专用的 6 个项目级 skills。
- 补充项目本地 direnv 入口：[.envrc](../.envrc)。

本轮不包含：

- 不同步上游 MiniMind 源码到本仓库。
- 不运行 MiniMind 训练、推理、pytest 或 GPU smoke test。
- 不提交 Git、不 push。

## 2. 改动文件清单

- [AGENTS.md](../AGENTS.md)：新增 MiniMind 项目级 Agent 指令，明确语言、环境、Git、测试、报告、版本、上游来源和模型训练推理专项约束。
- [.envrc](../.envrc)：新增项目级虚拟环境自动激活配置，只包含 `.venv` 路径和 `PATH_add`。
- [.agents/skills/minimind-code-reviewer/SKILL.md](../.agents/skills/minimind-code-reviewer/SKILL.md)：重写为 MiniMind 专用审查 skill。
- [.agents/skills/minimind-code-reviewer/references/review-template.md](../.agents/skills/minimind-code-reviewer/references/review-template.md)：重写为 MiniMind 专用审查 prompt 模板。
- [.agents/skills/minimind-knowledge-explainer/SKILL.md](../.agents/skills/minimind-knowledge-explainer/SKILL.md)：重写为 MiniMind LLM 源码、训练、推理和面试表达讲解 skill。
- [.agents/skills/minimind-plan-mode-planner/SKILL.md](../.agents/skills/minimind-plan-mode-planner/SKILL.md)：重写为 MiniMind 专用 Plan mode prompt 生成器。
- [.agents/skills/minimind-plan-mode-planner/references/plan-template.md](../.agents/skills/minimind-plan-mode-planner/references/plan-template.md)：重写为 MiniMind 专用计划模板。
- [.agents/skills/minimind-prompt-creator/SKILL.md](../.agents/skills/minimind-prompt-creator/SKILL.md)：重写为 MiniMind 专用执行 prompt 生成器。
- [.agents/skills/minimind-prompt-creator/references/prompt-template.md](../.agents/skills/minimind-prompt-creator/references/prompt-template.md)：重写为 MiniMind 专用执行 prompt 模板。
- [.agents/skills/minimind-prompt-creator/scripts/validate_project_prompt.py](../.agents/skills/minimind-prompt-creator/scripts/validate_project_prompt.py)：更新为 MiniMind 专用 prompt 结构校验。
- [.agents/skills/minimind-skill-creator/SKILL.md](../.agents/skills/minimind-skill-creator/SKILL.md)：重写为 MiniMind 项目级 skill 创建器。
- [.agents/skills/minimind-skill-creator/references/skill-template.md](../.agents/skills/minimind-skill-creator/references/skill-template.md)：重写为 MiniMind 项目级 skill 模板。
- [.agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py](../.agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py)：保留并用于校验讲解类 skill 的关键质量约束。
- [.agents/skills/minimind-web-search/SKILL.md](../.agents/skills/minimind-web-search/SKILL.md)：重写为 MiniMind 外部依据检索 skill。
- 各 skill 的 `agents/openai.yaml`：保留 `$minimind-*` 触发名和 MiniMind 专用展示文案。
- [minimind-roadmap.md](minimind-roadmap.md)：新增后续推进规划，覆盖源码地图、数据语义、模型模块、训练闭环、GPU smoke test、推理、SFT/LoRA/蒸馏和面试复盘。

## 3. 关键修复内容

1. 修正 skill 发现路径  
   将项目级 skills 放在 `.agents/skills/`，而不是项目 `.codex/skills/`。

2. 去掉全局通用模板干扰  
   删除不需要的 `minimind-agents-md-creator` 项目级 skill，因为 [AGENTS.md](../AGENTS.md) 已经生成。

3. 将 6 个 skills 项目化  
   当前保留：

   - `minimind-code-reviewer`
   - `minimind-knowledge-explainer`
   - `minimind-plan-mode-planner`
   - `minimind-prompt-creator`
   - `minimind-skill-creator`
   - `minimind-web-search`

4. 移除泛化表述  
   已清理 “不绑定任何单一项目”、“覆盖后端、前端、数据库、DevOps、产品业务”等不符合 MiniMind 项目级 skill 的表述。

5. 移除空占位符和旧模板残留  
   已清理 `{{...}}`、`生成时根据...`、旧 `$project-*` / `$web-search` 等通用 skill 交叉引用。

6. 固化 MiniMind 事实边界  
   skills 中明确区分本机已验证事实、本地代码事实、上游 MiniMind 事实、Learn MiniMind 学习材料、工程判断和后续计划。

## 4. 验收方式 / 手测步骤 / 自动化测试情况

已运行：

```bash
git status --short --branch
```

结果摘要：

```text
## main...origin/main
?? .agents/
?? .envrc
?? AGENTS.md
?? docs/
```

已运行项目级 skill 结构校验，结果摘要：

```text
通过：6 个 MiniMind 项目级 skill 的 Markdown/YAML 已完成项目化检查。
```

已运行 prompt 模板专项校验：

```bash
/home/harry/projects/MiniMind/.venv/bin/python -X utf8 .agents/skills/minimind-prompt-creator/scripts/validate_project_prompt.py .agents/skills/minimind-prompt-creator/references/prompt-template.md
```

结果：

```text
通过：项目任务 prompt 关键约束检查通过。
```

已运行 knowledge explainer 专项校验：

```bash
/home/harry/projects/MiniMind/.venv/bin/python -X utf8 .agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py .agents/skills/minimind-knowledge-explainer/SKILL.md
```

结果：

```text
通过：minimind-knowledge-explainer 关键质量约束检查通过。
```

已运行 Python 脚本编译校验：

```bash
/home/harry/projects/MiniMind/.venv/bin/python -m py_compile .agents/skills/minimind-prompt-creator/scripts/validate_project_prompt.py .agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py
```

结果：命令退出码为 0。

## 5. 版本同步清单

- 当前 README 版本仍为 `v0.0.1`。
- 本轮是项目规则、项目级 skill 和文档治理，没有修改核心代码、训练入口、推理入口或用户可执行能力。
- 未修改版本号。
- 未创建独立版本文件。

## 6. 风险与备注

- 未验证 Codex UI 在新会话中是否展示全部 `.agents/skills/minimind-*`；需要用户重启或新开 Codex 会话后确认。
- 未运行 MiniMind 训练、推理、pytest、数据下载、模型权重下载或 GPU smoke test。
- 当前 `.agents/`、`.envrc`、`AGENTS.md`、`docs/` 均为未跟踪文件，尚未提交。
- `.envrc` 内容已检查，不包含 token、密码、私有代理地址或敏感信息。
- 本报告记录的是项目治理与工具链修复，不代表 MiniMind 源码复现或训练实验已经完成。

## 7. 结论

本轮 MiniMind 项目级协作基线已经建立：`AGENTS.md`、项目推进规划、6 个项目级 skills 和必要校验脚本均已落地。当前变更建议作为一次项目治理提交纳入 Git，但在提交前仍建议重启 Codex 验证 `$minimind-*` skill 是否能正常出现在选择器中。
