---
name: minimind-skill-creator
description: 为 MiniMind 个人复现项目创建或更新项目级 skill。用于把 MiniMind 中会反复出现的源码拆解、实验记录、训练验证、推理验证、面试复盘或项目治理流程沉淀为 `.agents/skills/minimind-*`，并避免通用模板污染项目上下文。
---

# minimind-skill-creator

## 技能定位

只创建 MiniMind 项目级 skill。新 skill 必须服务 MiniMind 的长期复现流程，不能做成通用开发、通用讲解或跨项目模板。

## 适合创建 skill 的流程

- 某类 MiniMind 任务会反复出现，例如数据标签对齐、单 batch 训练闭环、KV Cache 对照、实验记录整理。
- 每次都需要固定读取顺序、输出结构、验证命令或证据要求。
- 流程比 `AGENTS.md` 更具体，比一次性 prompt 更稳定。

## 不适合创建 skill 的情况

- 只执行一次的开发任务。
- 只是几条长期规则，应该写入 `AGENTS.md`。
- 只是概念讲解，优先用 `$minimind-knowledge-explainer`。
- 只是审查或 prompt 生成，优先用已有项目级 skill。
- 想把所有 MiniMind 规则塞进一个万能 skill。

## 创建规则

- 目录必须位于 `.agents/skills/minimind-*`。
- `SKILL.md` 必须绑定 MiniMind 项目，frontmatter 只包含 `name` 和 `description`。
- `agents/openai.yaml` 必须引用正确 `$skill-name`。
- `references/` 只能放 MiniMind 专用模板或检查清单，不保留空占位符。
- `scripts/` 只放可重复运行的校验脚本，并必须实际运行或说明未运行原因。
- 不创建空目录、空模板或“以后可能用”的文件。

## 使用流程

1. 确认该流程是否值得沉淀成 skill。
2. 读取 `AGENTS.md`、`README.md`、`docs/minimind-roadmap.md` 和相关项目文件。
3. 读取 `references/skill-template.md`。
4. 生成或更新 `.agents/skills/minimind-*`。
5. 检查交叉引用是否使用 `$minimind-*`。
6. 校验 UTF-8、frontmatter、`openai.yaml`、引用文件和脚本。

## 讲解类专项

创建或修改讲解类 skill 时，不得削弱 `$minimind-knowledge-explainer` 中的 MiniMind 绑定、1000 / 3000 字硬门槛、项目事实边界、公式解释、项目落地点和面试追问要求。

运行：

```bash
direnv exec . python -X utf8 .agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py .agents/skills/minimind-knowledge-explainer/SKILL.md
```

## 自检

1. 是否只服务 MiniMind 项目。
2. 是否避免通用模板残留。
3. 是否所有 references 都已项目化。
4. 是否所有交叉引用都使用 `$minimind-*`。
5. 是否运行了必要校验。
