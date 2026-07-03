---
name: minimind-code-reviewer
description: 为 MiniMind 个人复现仓库生成或执行代码审查。用于审查模型、数据、训练、推理、测试、文档、实验记录、项目级 skill 和 AGENTS 规则改动；重点检查事实边界、LLM 张量语义、训练/推理闭环、8GB 显存风险、未验证成果表述和缺失测试。
---

# minimind-code-reviewer

## 技能定位

审查 MiniMind 项目改动。默认以 code-review 立场输出 findings first；如果用户只要求生成审查 prompt，则读取 `references/review-template.md` 生成可复制 prompt。

## 审查重点

- MiniMind 事实边界：本机验证、当前本地代码、个人远端、上游 MiniMind、Learn MiniMind 是否混写。
- 模型语义：shape、dtype、device、RoPE、GQA、causal mask、KV Cache、loss shift 是否有证据。
- 数据语义：tokenizer、chat template、padding、labels、`-100`、loss mask 是否清楚。
- 训练工程：AMP、梯度累积、checkpoint、resume、OOM、NaN、8GB 显存参数是否保守。
- 推理生成：权重来源、采样参数、EOS、history、cache on/off 是否可复现。
- 文档表达：未跑的实验不能写成已完成；教程话术不能写成个人经历。
- 项目工具：`.agents/skills`、`AGENTS.md`、`.envrc`、`.gitignore` 是否与仓库事实一致。

## 必读材料

1. `AGENTS.md`、`README.md`、`.gitignore`、`.envrc`。
2. 本轮 diff 或用户指定范围。
3. 相关源码、测试、实验记录或上游引用文件。
4. 如果涉及上游变化，使用 `$minimind-web-search` 或本地 `/home/harry/references/minimind` 核验。

## 输出要求

- findings first，按严重程度排序。
- 每条 finding 包含：位置、证据、影响、建议修复、最小验证。
- 位置使用 Markdown 相对路径，能定位行号时用 `[文件名](相对路径#L行号)`。
- 如果未发现明确问题，说明已检查范围、剩余风险和测试缺口。
- 不要把风格偏好写成 bug。
- 不要自动修改代码，除非用户明确要求“审查并修复”。

## prompt 生成模式

当用户要求生成审查 prompt 时：

1. 读取 `references/review-template.md`。
2. 把审查范围、对比基准、风险地图和报告路径写清。
3. 要求执行者保存审查报告到 `docs/reports/` 或用户指定路径。
4. 要求执行者区分已验证事实、工程判断和未验证边界。

## 自检

1. 是否审查 MiniMind 专属风险，而不是通用代码风格。
2. 是否检查“未验证写成已验证”的问题。
3. 是否要求最小验证和证据路径。
4. 是否没有回滚用户已有改动。
