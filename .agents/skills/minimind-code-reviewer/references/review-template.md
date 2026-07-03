# MiniMind 代码审查 prompt 模板

本模板只用于 MiniMind 个人复现仓库。生成最终 prompt 时，必须结合具体 diff 或文件范围删减内容，不得保留未填写说明。

````text
你要在 `/home/harry/projects/MiniMind` 对本轮 MiniMind 改动做 code review。默认只输出审查报告和建议，不直接修改代码；除非用户明确要求“审查并修复”，否则不要 apply patch。

【必须先读】

1. `AGENTS.md`
2. `README.md`
3. `docs/minimind-roadmap.md`
4. 本轮审查范围内的文件
5. 如果涉及上游对照，核验 `/home/harry/references/minimind`

【审查范围】

- 按用户指定 diff、文件、目录或提交审查。
- 如果用户未指定范围，先用 `git status --short --branch` 和 `git diff --stat` 确认范围，再说明审查基准。

【MiniMind 专属风险地图】

审查前先建立风险地图，至少检查：

1. 是否把上游 MiniMind、Learn MiniMind 或计划内容写成个人成果。
2. 是否把未运行训练、推理、测试、smoke test 写成已验证。
3. tokenizer、labels、mask、loss、shape、dtype、device 的解释是否与代码一致。
4. 训练参数是否照搬上游多卡 3090/A100，而没有考虑 RTX 5060 Laptop 约 8GB 显存。
5. checkpoint、out、dataset、大模型权重、`.venv`、缓存和敏感配置是否可能误提交。
6. 文档、实验记录、README、AGENTS 或 skill 是否与当前仓库事实冲突。

【输出格式】

**审查发现**

按严重程度排序。每条 finding 使用：

### 严重 / 高 / 中 / 低：问题标题

位置：使用 `[文件名](相对路径#L行号)`；不能定位行号时说明原因。

证据：引用代码、文档、配置、测试或命令输出。

影响：说明会如何误导学习、破坏训练/推理语义、造成实验不可复现或引入维护风险。

建议修复：给出最小可行改法和需要补充的验证。

验证建议：列出应运行的最小命令或人工检查。

**未发现明确问题时**

写明已检查范围、未发现阻塞项、仍缺哪些测试或实验。

【报告要求】

如需保存报告，优先保存到 `docs/reports/`，文件名包含主题和日期。报告内链接必须使用 Markdown 相对路径。

【禁止】

- 不要把风格偏好写成 bug。
- 不要回滚用户已有改动。
- 不要把历史测试结果写成本轮验证。
- 不要把上游效果写成本机效果。
````
