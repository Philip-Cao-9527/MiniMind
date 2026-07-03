# MiniMind 执行 prompt 模板

本模板只用于 `/home/harry/projects/MiniMind`。生成最终 prompt 时，必须把任务目标、文件范围、验证命令和交付物写实，不保留未填写说明。

````text
你要在 `/home/harry/projects/MiniMind` 完成一次 MiniMind 任务。直接执行，不要停留在建议；除非遇到真实阻塞，否则推进到文件修改、验证和总结。

【硬性前置要求】

1. 先阅读：
   - `AGENTS.md`
   - `README.md`
   - `docs/minimind-roadmap.md`
   - 本轮相关源码、文档、测试或上游引用文件
2. 全程使用简体中文。
3. 区分本机已验证事实、本地代码事实、上游事实、学习材料和后续计划。
4. 不要回滚用户已有改动，不要自动 commit/push。
5. 不要建议重装 WSL、CUDA、PyTorch、虚拟环境或 Conda。

【MiniMind 项目边界】

- 本项目是个人 LLM 源码理解、关键模块手写复现和最小实验验证仓库。
- 不把上游 MiniMind 已实现能力写成个人成果。
- 不把 Learn MiniMind 的教程、简历话术或示例成果写成个人经历。
- RTX 5060 Laptop 约 8GB 显存下优先做极小配置和 smoke test。

【本次任务】

在生成最终 prompt 时写清：

1. 本轮目标。
2. 需要读取或修改的真实文件。
3. 明确不做的事项。
4. 交付物路径。
5. 最小验证命令。

【默认验证】

至少包含：

```bash
git status --short --branch
direnv exec . python -V
```

涉及 Markdown / YAML / JSON / Python 时，补充对应 UTF-8 读取、解析或编译校验。涉及模型、数据、训练、推理时，补充最小单测、单 batch 或 smoke test；无法运行时说明原因和未验证边界。

【最终总结】

按顺序说明：

1. 文件改动清单。
2. 验证命令和结果。
3. 证据路径。
4. 版本和报告是否触发；不触发时说明原因。
5. 风险与未验证边界。
6. 下一步最小建议。
````
