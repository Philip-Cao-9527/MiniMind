<div align="center">

# MiniMind

**小参数 LLM · 源码拆解 · 手撕实现 · 训练验证**

一个围绕 [MiniMind](https://github.com/jingyaogong/minimind) 展开的个人学习与复现仓库。

从一串 token 被送入模型，到梯度穿过网络、权重在迭代中更新，再到模型逐 token 生成文本，MiniMind 的训练与推理并不是一条可以一眼看透的流水线。

本项目沿着这条链路逐段拆解数据、模型、训练与生成过程，手撕关键实现，并通过测试与实验确认每个模块的输入、计算和输出是否真正符合预期。

<p>
  <img alt="version" src="https://img.shields.io/badge/version-v0.0.3-4c6ef5">
  <img alt="language" src="https://img.shields.io/badge/language-Python-3776AB">
  <img alt="platform" src="https://img.shields.io/badge/platform-WSL2%20%7C%20Ubuntu%2024.04-5E5CE6">
  <img alt="reference" src="https://img.shields.io/badge/reference-MiniMind-8A2BE2">
</p>

<p>
  <a href="#why">为什么做这个项目</a> ·
  <a href="#focus">重点学习内容</a> ·
  <a href="#workflow">学习与验证方式</a> ·
  <a href="#structure">仓库结构</a> ·
  <a href="#reference">参考与致谢</a>
</p>

</div>

---

<a id="why" name="why"></a>

## 为什么做这个项目

MiniMind 这类小型 LLM 项目，表面上看流程并不复杂：配好环境，准备数据，运行训练脚本。

但真正开始复现后，最容易卡住的并不是如何启动脚本，而是许多默认存在的实现细节并没有被真正理解。比如：

* 一段 token 序列进入模型后，输入、标签和 loss mask 分别如何构造
* Causal Mask、Padding Mask、RoPE 与 KV Cache 各自解决什么问题
* Attention 的张量 shape 在每一步如何变化
* 训练 loss 下降时，模型究竟学到了什么，哪些情况只是表面正常
* 预训练、SFT、LoRA、蒸馏和推理阶段之间，哪些逻辑可以复用，哪些必须严格区分
* 修改模型、数据或生成逻辑后，如何确认没有破坏训练与推理的一致性
* 出现 OOM、NaN、收敛异常时，怎样定位真实原因

---

<a id="focus" name="focus"></a>

## 重点学习内容

### 模型结构

围绕 Decoder-only Transformer 逐步理解和实现核心模块：

* Token Embedding 与位置编码
* RMSNorm
* Causal Self-Attention
* RoPE
* MLP 与 MoE
* KV Cache
* 模型配置、参数初始化与前向计算

### 数据与标签

重点关注语言模型训练中最容易被忽略、但最容易导致语义错误的部分：

* Tokenizer 与特殊 token
* JSONL 数据读取
* 因果语言模型标签构造
* Padding 与 Attention Mask
* Loss Mask
* 序列截断与样本边界
* 预训练数据与指令微调数据的差异

### 训练与微调

围绕训练循环建立完整理解：

* 前向传播与交叉熵损失
* 反向传播与梯度更新
* 梯度累积
* 混合精度训练
* 学习率调度
* Checkpoint 保存与恢复
* 预训练、SFT、LoRA、蒸馏等训练阶段

### 推理与生成

从生成过程理解模型实际如何输出文本：

* Greedy Decoding
* Temperature Sampling
* Top-k 与 Top-p Sampling
* 重复惩罚
* KV Cache 加速
* EOS 终止条件
* 流式输出与生成状态管理

### 验证与复盘

每个关键模块都会尽量形成一条完整闭环：

```text
阅读源码
  -> 明确输入输出与张量语义
  -> 手撕实现
  -> 对照上游逻辑
  -> 编写最小测试
  -> 运行小规模实验
  -> 记录结果、差异与失败原因
```

---

<a id="workflow" name="workflow"></a>

## 学习与验证方式

本项目更关注的是先理解一个模块解决的问题，再独立写出能够解释的版本，最后通过源码对照和最小实验检查行为是否一致。

例如，一个 Attention 模块不会只检查能否前向运行，还会关注：

* 输入输出 shape 是否正确
* dtype 与 device 是否保持一致
* 因果约束是否真正生效
* Padding 是否影响有效 token
* 单 batch 反向传播是否正常
* 修改缓存开关后，生成结果是否与无缓存版本一致
* 边界输入和异常输入是否暴露真实问题

训练相关实验也会尽量保留必要条件，包括代码版本、数据版本、随机种子、主要配置、设备条件、训练步数和评价口径。

---

<a id="structure" name="structure"></a>

## 仓库结构

```text
MiniMind/
├── dataset/            # 数据读取逻辑与本地训练数据
├── model/              # 模型结构与关键模块实现
├── trainer/            # 训练入口、训练循环与恢复训练逻辑
├── scripts/            # 推理、转换与辅助脚本
├── tests/              # 单元测试、边界测试与 smoke test
├── experiments/        # 可执行的小实验：代码、极小输入数据
├── docs/               # 源码理解、实验结果与问题复盘
├── out/                # 可推理模型权重，不提交 Git
├── checkpoints/        # 断点续训状态，不提交 Git
├── README.md
└── requirements.txt    # 环境验证完成后维护
```

目录语义尽量与 MiniMind 上游保持一致。

* `dataset/` 同时承载数据读取逻辑与本地训练数据
* `model/` 存放模型结构源码，不用于保存训练权重
* `trainer/` 存放训练入口、训练循环和训练辅助逻辑
* `out/` 存放可用于推理或后续训练阶段初始化的模型权重
* `checkpoints/` 存放恢复训练所需的完整状态
* `docs/` 只沉淀已经完成的源码理解、实验记录与问题复盘


---

<a id="reference" name="reference"></a>

## 参考与致谢

本项目主要参考以下开源仓库：

* [jingyaogong/minimind](https://github.com/jingyaogong/minimind)  
  MiniMind 主仓库，作为模型实现、训练入口、数据格式与默认行为的主要参考来源。

* [bcefghj/learn-minimind](https://github.com/bcefghj/learn-minimind)  
  MiniMind 学习辅助仓库，用于帮助梳理源码阅读顺序、理解核心模块和准备相关技术问题。