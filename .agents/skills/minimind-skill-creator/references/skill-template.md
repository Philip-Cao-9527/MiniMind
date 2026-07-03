# MiniMind 项目级 skill 模板

本模板用于创建 `.agents/skills/minimind-*`。新 skill 必须服务 MiniMind 长期复现流程，不得做成通用开发模板。

## 最小目录

```text
minimind-task-name/
├── SKILL.md
└── agents/
    └── openai.yaml
```

只有确实需要较长模板或脚本时，才增加：

```text
references/
scripts/
templates/
assets/
```

## SKILL.md 骨架

```markdown
---
name: minimind-task-name
description: 为 MiniMind 个人复现项目处理某个高频任务。写清触发场景、产物和边界，不要泛化到其他项目。
---

# minimind-task-name

## 技能定位

说明它服务 MiniMind 的哪一类流程，例如 tokenizer 标签对齐、单 batch 训练闭环、KV Cache 对照、实验记录整理或面试复盘。

## 必读材料

- `AGENTS.md`
- `README.md`
- `docs/minimind-roadmap.md`
- 与该流程直接相关的本地文件或上游引用文件

## 使用流程

1. 确认任务属于 MiniMind 项目。
2. 读取必读材料。
3. 按本 skill 的固定顺序处理。
4. 运行最小验证。
5. 输出证据路径、未验证边界和下一步。

## 质量要求

- 不把上游成果写成个人成果。
- 不把未运行实验写成已验证。
- 不保留通用模板残留。
- 所有交叉引用使用 `$minimind-*`。
```

## openai.yaml 骨架

```yaml
interface:
  display_name: "MiniMind 任务名称"
  short_description: "一句话说明 MiniMind 专用用途"
  default_prompt: "使用 $minimind-task-name 处理 MiniMind 当前任务。"
```

## 校验

创建或更新后至少运行：

```bash
git status --short --branch
direnv exec . python - <<'PY'
from pathlib import Path
for p in Path('.agents/skills').rglob('*'):
    if p.suffix in {'.md', '.yaml'}:
        text = p.read_text(encoding='utf-8')
        assert '花括号占位' not in text
        assert '跨项目泛化表述' not in text
        assert '无关技术栈罗列' not in text
PY
```

讲解类 skill 还必须运行：

```bash
direnv exec . python -X utf8 .agents/skills/minimind-skill-creator/scripts/validate_knowledge_explainer.py .agents/skills/minimind-knowledge-explainer/SKILL.md
```
