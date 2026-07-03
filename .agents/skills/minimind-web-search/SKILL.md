---
name: minimind-web-search
description: 为 MiniMind 个人复现项目检索外部依据。用于核验上游 jingyaogong/minimind 当前源码、个人仓库远端、learn-minimind 学习材料、MiniMind 相关论文或官方文档、PyTorch/Transformers/API 版本、数据集、benchmark、许可证和工程实践；用户明确要求不联网时不要使用。
---

# minimind-web-search

## 技能定位

只为 MiniMind 项目补充外部事实，不做泛化资料搜索。它解决的问题是：当本地仓库不足以确认事实时，先查可靠来源，再把外部结论转换成 MiniMind 的工程约束、验证步骤和未验证边界。

## 来源优先级

1. 当前本地仓库 `/home/harry/projects/MiniMind`。
2. 个人 GitHub 仓库 `Philip-Cao-9527/MiniMind`。
3. 上游 `jingyaogong/minimind` 的当前 README、源码、release、issue 或 commit。
4. `bcefghj/learn-minimind`，仅用于学习顺序和面试追问准备。
5. 官方文档、论文、PyTorch / Transformers / datasets 文档、benchmark 官方页面。
6. 高质量博客或社区讨论只作补充，不能作为唯一依据。

## 触发场景

- 需要确认上游 MiniMind 最新目录、参数、训练入口、模型配置、README 或 issue。
- 需要确认 PyTorch、Transformers、datasets、tokenizers 等库的当前行为。
- 需要引用论文、官方文档、benchmark 或许可证。
- 需要比较本地仓库、个人远端和上游仓库差异。
- 其他 MiniMind skill 要求外部依据支撑结论。

## 输出要求

- 先说明检索要解决的具体问题。
- 给出使用的关键词和来源类型。
- 按“已确认事实、来源冲突、工程推断、仍需验证边界”分层。
- 给出可点击链接和必要的提交哈希、文件路径或版本号。
- 把结论落到 MiniMind：需要读哪些文件、改哪些文件、跑哪些最小验证。
- 不要把外部仓库做过的事写成用户本机已完成。

## 自检

1. 是否优先使用了一手来源。
2. 是否把来源和结论分层。
3. 是否没有把上游成果写成个人成果。
4. 是否把外部结论转成 MiniMind 的下一步验证。
