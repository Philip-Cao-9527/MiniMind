---
name: minimind-plan-mode-planner
description: 为 MiniMind 个人复现项目生成 Plan mode prompt。用于源码同步、训练实验、模型模块复现、数据语义验证、GPU smoke test、推理/KV Cache、SFT/LoRA/蒸馏等需要先只读探索、提炼关键决策、再执行的任务。
---

# minimind-plan-mode-planner

## 技能定位

生成 MiniMind 专用 Plan mode prompt。它不直接实现，也不替用户拍板；它要求执行者先只读核验本地仓库、个人远端、上游 MiniMind 和硬件边界，再把真正需要用户选择的决策点压缩到 2 到 4 个。

## 适用场景

- 是否同步上游源码、同步哪些目录、如何保留个人复现边界。
- 是否手写模块、复用上游模块或做对照测试。
- 训练参数、smoke test 范围、显存风险和实验记录方式需要先决策。
- 文档、skill、AGENTS、实验记录结构需要先定边界。

## 必读材料

- `AGENTS.md`
- `README.md`
- `docs/minimind-roadmap.md`
- 本轮相关本地文件
- 必要时核验 `/home/harry/references/minimind` 和 `/home/harry/references/learn-minimind`

## 使用流程

1. 确认用户要先计划，不是直接执行。
2. 读取 `references/plan-template.md`。
3. 写明只读阶段允许的命令和禁止的修改。
4. 提炼 MiniMind 项目的关键决策，例如同步策略、验证深度、实验范围、报告路径、可回滚边界。
5. 明确用户选择后才进入实现。

## 质量要求

- 决策点必须影响后续实现，不把文件名、函数名这类可自行判断的细节丢给用户。
- 必须写清本机 8GB 显存、当前源码缺失或上游差异带来的边界。
- 必须要求后续计划包含文件级改动范围、验证命令、证据路径和未验证项。
- 不得把计划写成已经完成的工作。

## 自检

1. 是否绑定 MiniMind 项目。
2. 是否要求只读探索。
3. 是否把关键决策控制在 2 到 4 个。
4. 是否没有替用户拍板。
5. 是否包含用户选择后的验证和交付要求。
