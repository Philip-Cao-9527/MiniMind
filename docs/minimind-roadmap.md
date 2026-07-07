# MiniMind 个人复现后续推进规划

本规划记录当前 MiniMind 个人复现项目的后续推进路径。它只描述下一阶段要做什么、为什么做、如何验证，不把上游 MiniMind 已实现内容、学习教程表述或未来计划写成个人实验成果。

## 当前状态快照

- 本地仓库：`..`（相对本文件的仓库根目录）
- 个人远端：`git@github.com:Philip-Cao-9527/MiniMind.git`
- 当前本地提交：`8093d92bf70a6e7f4052adbac568d8f1c89d35b4`
- 当前个人远端 `origin/main`：`8093d92bf70a6e7f4052adbac568d8f1c89d35b4`
- 上游 MiniMind 本地引用：`../../../references/minimind`（相对本文件）
- 上游 MiniMind 当前引用提交：`512eed0b6556e741d80864f054d45d271459772a`
- Learn MiniMind 本地引用：`../../../references/learn-minimind`（相对本文件）
- Learn MiniMind 当前引用提交：`76fbe5c34c588a9efb924f1399db6e71aad07e81`
- 当前仓库可见的长期项目治理文件包括 [README.md](../README.md)、[.gitignore](../.gitignore)、[AGENTS.md](../AGENTS.md)、[.envrc](../.envrc) 和 [docs/minimind-roadmap.md](minimind-roadmap.md)。阶段性修复报告统一放在 `docs/` 下，按任务相关性读取，不作为长期固定必读文件。
- 本地已验证环境包括 WSL2 Ubuntu、项目 `.venv`、PyTorch `2.7.1+cu128`、CUDA 可用和 RTX 5060 Laptop GPU 可见。
- 本轮尚未验证 MiniMind 训练、推理、pytest、真实数据下载、模型权重下载或 smoke test。

## 推进原则

- 当前本地仓库事实优先；本地缺失时再核验个人远端和上游 MiniMind。
- 上游源码可以同步、拆解、对照，但不能直接写成个人原创成果。
- Learn MiniMind 只用于学习顺序、理解辅助和面试追问准备，不作为个人实验结果来源。
- 每个阶段都必须留下证据：文件路径、命令、输出摘要、实验记录或测试结果。
- RTX 5060 Laptop 约 8GB 显存下优先跑极小配置和 smoke test，不照搬上游多卡 3090、A100 或完整训练参数。

## 阶段 1：建立同步基线

### 阶段目标

明确个人仓库、上游 MiniMind 与学习辅助仓库各自承担的角色，建立后续同步、对照和引用边界。

### 需要阅读、同步、修改或手写的真实文件

- [README.md](../README.md)
- [.gitignore](../.gitignore)
- [AGENTS.md](../AGENTS.md)
- [上游 README.md](../../../references/minimind/README.md)
- [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py)
- [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)
- `../../../references/minimind/trainer/*.py`
- `../../../references/minimind/scripts/*.py`
- [Learn MiniMind L04 项目导览](../../../references/learn-minimind/docs/L04-MiniMind项目导览.md)

### 关键理解点

- 哪些文件是个人仓库真实存在的内容。
- 哪些文件只是上游参考源码，尚未同步到个人仓库。
- 个人复现仓库不是上游仓库镜像，文档必须区分“已实现”“已阅读”“计划实现”。


## 阶段 2：源码地图与调用链拆解

### 阶段目标

建立模型、数据、训练、推理、配置、权重与评估入口之间的源码地图。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py)
- 上游 [model/model_lora.py](../../../references/minimind/model/model_lora.py)
- 上游 [dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)
- 上游 [trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)
- 上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)
- 上游 [trainer/train_lora.py](../../../references/minimind/trainer/train_lora.py)
- 上游 [trainer/train_distillation.py](../../../references/minimind/trainer/train_distillation.py)
- 上游 [trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)
- 上游 [eval_llm.py](../../../references/minimind/eval_llm.py)
- 上游 [scripts/chat_api.py](../../../references/minimind/scripts/chat_api.py)
- 上游 [scripts/serve_openai_api.py](../../../references/minimind/scripts/serve_openai_api.py)
- 上游 [scripts/convert_model.py](../../../references/minimind/scripts/convert_model.py)

### 关键理解点

- `MiniMindConfig` 如何决定 hidden size、层数、heads、GQA、RoPE、MoE 与词表大小。
- `PretrainDataset` 与 `SFTDataset` 如何构造 `input_ids` 和 `labels`。
- 训练入口如何初始化模型、tokenizer、DataLoader、optimizer、AMP、checkpoint 和 resume。
- 推理入口如何加载权重、构造 prompt、调用 `generate` 并处理采样参数。

### 可交付产物

- 拟新增 `docs/source-map.md` 中的“源码入口表”。
- 拟新增 `docs/call-chain-pretrain.md`：预训练调用链。
- 拟新增 `docs/call-chain-inference.md`：推理调用链。

### 必要但最小的验证

- 用 `rg` 和 `sed` 对入口函数、类名、参数默认值做静态核验。
- 暂不要求训练或推理运行，但每个结论必须能指向真实文件。

### 风险与未验证边界

- 静态阅读只能确认代码路径，不能证明本机训练或推理已经成功。
- 如果个人仓库尚未同步源码，不要在文档中用个人仓库路径描述上游文件。

### 完成标准

- 能从一条样本进入 Dataset 开始，解释到模型 forward、loss、backward、optimizer step 和 checkpoint 保存。
- 能从一个 prompt 开始，解释到 tokenizer、generate、KV Cache、采样和 EOS 停止。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/source-map.md`
- 拟新增 `docs/call-chain-pretrain.md`
- 拟新增 `docs/call-chain-inference.md`

## 阶段 3：tokenizer、数据格式、标签与 mask

### 阶段目标

用最小样本验证 tokenizer、JSONL 数据格式、labels、padding、attention mask 和 loss mask 的真实语义。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [model/tokenizer.json](../../../references/minimind/model/tokenizer.json)
- 上游 [model/tokenizer_config.json](../../../references/minimind/model/tokenizer_config.json)
- 上游 [dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)
- 本仓库拟新增 `tests/test_dataset_labels.py`
- 本仓库拟新增 `docs/tokenizer-label-mask.md`
- 本仓库拟新增极小测试数据，例如 `tests/fixtures/pretrain_tiny.jsonl`、`tests/fixtures/sft_tiny.jsonl`

### 关键理解点

- 预训练样本以 `text` 字段为核心，labels 主要复制 `input_ids`，padding token 的 label 变成 `-100`。
- SFT 样本依赖 `apply_chat_template`，只让 assistant 回复区间参与 loss。
- `labels[..., 1:]` 与 `logits[..., :-1, :]` 的 shift 关系。
- `-100` 表示 `F.cross_entropy(..., ignore_index=-100)` 忽略该位置。
- padding、attention mask、labels mask 不能混为一谈。

### 可交付产物

- 一个最小 tokenizer 与 label 对齐测试。
- 一份 token、label、mask 对照表或实验记录。

### 必要但最小的验证

```bash
direnv exec . python -m pytest tests/test_dataset_labels.py -q
```

如果暂未引入 pytest，则先用单脚本打印并保存输出摘要，后续再转成测试。

### 风险与未验证边界

- tokenizer 版本与上游模型权重必须匹配；不同 tokenizer 会导致 token id 和模板边界变化。
- 不要把 DPO、RLAIF、Agent RL 的 mask 规则提前套到 Pretrain 或 SFT。

### 完成标准

- 能用一条极小样本解释每个 token 是否参与 loss。
- 能说清楚 Pretrain 与 SFT 的标签构造差异。

### 应沉淀的文档、测试或实验记录

- `tests/test_dataset_labels.py`
- `tests/fixtures/*.jsonl`
- 拟新增 `docs/tokenizer-label-mask.md`

## 阶段 4：模型核心模块手写对照复现

### 阶段目标

逐个手写并对照验证 Decoder-only Transformer 的关键模块。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py)
- 本仓库拟新增 `model/` 下的个人实现模块，或 `tests/handwritten/` 下的对照实现。
- 本仓库拟新增 `tests/test_rmsnorm.py`
- 本仓库拟新增 `tests/test_rope.py`
- 本仓库拟新增 `tests/test_attention_shapes.py`
- 本仓库拟新增 `tests/test_mlp.py`

### 关键理解点

- RMSNorm 的均方归一化与 dtype 回转。
- RoPE 的 cos/sin 预计算、rotate half 与位置切片。
- GQA 中 q heads、kv heads、repeat kv 的 shape 关系。
- causal mask 与 padding mask 的叠加位置。
- SwiGLU/MLP 的 gate、up、down 投影关系。
- MoE top-k routing、aux loss 与训练时空 expert 分支。

### 可交付产物

- 个人手写模块或对照测试。
- 拟新增 `docs/model-modules.md`：逐模块说明输入、输出、shape、公式和对照结论。

### 必要但最小的验证

```bash
direnv exec . python -m pytest tests/test_rmsnorm.py tests/test_rope.py tests/test_attention_shapes.py tests/test_mlp.py -q
```

### 风险与未验证边界

- 张量 shape 能对上不代表数值完全一致；需要固定随机种子和小张量做数值对照。
- Flash Attention 路径、手写 attention 路径和带 padding mask 路径需要分开验证。

### 完成标准

- 每个核心模块都有最小可运行测试。
- 每个模块都能说明 shape、dtype、device、mask 或 cache 相关边界。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/model-modules.md`
- `tests/test_*.py`

## 阶段 5：单 batch 前向、loss、反向和参数更新闭环

### 阶段目标

在极小配置下验证模型训练最小闭环，而不是直接跑完整训练。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py)
- 上游 [trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)
- 上游 [trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)
- 本仓库拟新增 `tests/test_single_batch_train.py`
- 本仓库拟新增 `docs/single-batch-train.md`

### 关键理解点

- logits 与 labels 的 shift。
- loss 是否有限、是否参与反向。
- optimizer step 后参数是否发生变化。
- AMP、bf16、fp16、GradScaler 的启用条件。
- aux loss 在 Dense 与 MoE 下的差异。

### 可交付产物

- 单 batch 训练测试。
- 一份包含 seed、模型配置、数据、loss、参数变化摘要的实验记录。

### 必要但最小的验证

```bash
direnv exec . python -m pytest tests/test_single_batch_train.py -q
```

### 风险与未验证边界

- 单 batch loss 正常不代表模型可收敛。
- CPU 通过不代表 CUDA、AMP、显存路径已经验证。

### 完成标准

- 极小模型能完成 forward、loss、backward、optimizer step。
- 参数更新前后至少一个可训练参数发生可观测变化。

### 应沉淀的文档、测试或实验记录

- `tests/test_single_batch_train.py`
- 拟新增 `docs/single-batch-train.md`

## 阶段 6：RTX 5060 Laptop 8GB smoke test

### 阶段目标

在当前笔记本 GPU 上验证极小训练和推理路径可运行，记录真实显存边界。

### 需要阅读、同步、修改或手写的真实文件

- 训练入口：`trainer/train_pretrain.py`、`trainer/train_full_sft.py`
- 推理入口：`eval_llm.py`
- 数据：极小 JSONL fixture 或 mini 数据集
- 输出：拟新增 `docs/gpu-smoke-test.md`

### 关键理解点

- batch size、max sequence length、hidden size、layers、dtype 与显存的关系。
- 梯度累积如何替代直接增大 batch。
- checkpoint 与 resume 文件分别保存什么。
- OOM 后应记录参数组合和错误，不重装 CUDA、驱动或 PyTorch。

### 可交付产物

- GPU smoke test 实验记录。
- 一组本机可运行的保守参数。

### 必要但最小的验证

```bash
direnv exec . python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")
PY
```

后续在源码同步后再运行极小训练命令，命令参数必须根据实际文件落地。

### 风险与未验证边界

- 8GB 显存无法保证完整 64M 主线训练参数都可用。
- smoke test 只验证路径和基本稳定性，不代表训练效果。

### 完成标准

- 有一份本机 GPU 可复现命令。
- 有 loss、显存现象、耗时或失败边界记录。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/gpu-smoke-test.md`

## 阶段 7：混合精度、梯度累积、checkpoint 与恢复训练

### 阶段目标

验证训练工程能力：AMP、梯度累积、保存权重、保存 resume 状态和恢复训练。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)
- 上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)
- 上游 [trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)
- 本仓库拟新增 `docs/checkpoint-resume.md`
- 必要时新增 `tests/test_checkpoint_resume.py`

### 关键理解点

- `lm_checkpoint` 保存模型权重、optimizer、scaler、epoch、step、world size 和 wandb id 的逻辑。
- `accumulation_steps` 对 step、loss 缩放和 optimizer step 的影响。
- bf16 与 fp16 下 GradScaler 的不同启用行为。
- DDP 分支在单卡环境下不应误触发。

### 可交付产物

- checkpoint 与 resume 最小实验。
- 恢复前后 step、loss 和参数状态说明。

### 必要但最小的验证

- 先跑极小步数保存 resume。
- 再从 resume 继续跑一小段，确认 step 转换和参数加载正常。

### 风险与未验证边界

- 只在单卡验证不代表多卡 DDP 恢复已验证。
- checkpoint 文件属于大文件或运行产物，不应提交 Git。

### 完成标准

- 能解释普通权重文件与 resume 文件的差异。
- 能完成一次中断后恢复训练的最小闭环。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/checkpoint-resume.md`

## 阶段 8：推理、采样、KV Cache 与停止条件

### 阶段目标

验证最小推理路径，并理解生成过程中 cache、采样参数与停止条件如何影响输出。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py)
- 上游 [eval_llm.py](../../../references/minimind/eval_llm.py)
- 上游 [scripts/chat_api.py](../../../references/minimind/scripts/chat_api.py)
- 上游 [scripts/serve_openai_api.py](../../../references/minimind/scripts/serve_openai_api.py)
- 本仓库拟新增 `docs/inference-generation.md`
- 必要时新增 `tests/test_generation_cache.py`

### 关键理解点

- `generate` 中 `past_key_values` 的增长方式。
- `attention_mask` 随新 token 拼接。
- temperature、top-p、top-k、repetition penalty 的语义。
- EOS、max new tokens 与历史对话长度的停止边界。
- 有无 cache 的结果一致性要在固定条件下讨论。

### 可交付产物

- 一份推理最小实验记录。
- 一个 cache on/off 或短序列生成对照测试。

### 必要但最小的验证

- 固定 prompt、权重、tokenizer、采样参数和 seed。
- 保存输入、参数和输出摘要。

### 风险与未验证边界

- 随机采样输出不能作为模型能力证明。
- 没有权重时只能做静态阅读或构造极小随机模型测试，不能写成真实推理效果。

### 完成标准

- 能解释一次生成中每一步 input、cache、attention mask 和 stop 的变化。
- 能区分预训练权重、SFT 权重、LoRA 权重和随机初始化模型的输出意义。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/inference-generation.md`
- `tests/test_generation_cache.py`

## 阶段 9：SFT、LoRA、蒸馏与评估

### 阶段目标

在完成基础训练与推理闭环后，再推进更高阶训练任务，避免在基础语义未清楚前堆复杂流程。

### 需要阅读、同步、修改或手写的真实文件

- 上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)
- 上游 [trainer/train_lora.py](../../../references/minimind/trainer/train_lora.py)
- 上游 [model/model_lora.py](../../../references/minimind/model/model_lora.py)
- 上游 [trainer/train_distillation.py](../../../references/minimind/trainer/train_distillation.py)
- 上游 [eval_llm.py](../../../references/minimind/eval_llm.py)
- 上游 [scripts/eval_toolcall.py](../../../references/minimind/scripts/eval_toolcall.py)

### 关键理解点

- SFT 只训练 assistant 回复区间。
- LoRA 如何插入线性层、保存和合并权重。
- 蒸馏中 teacher logits、student logits、CE loss、distill loss 和 loss mask 的关系。
- 评估数据、指标和 prompt 模板必须明确，不把单次输出当 benchmark。

### 可交付产物

- 拟新增 `docs/sft-smoke-test.md`
- 拟新增 `docs/lora-smoke-test.md`
- 拟新增 `docs/distillation-reading-notes.md`
- 必要时新增 LoRA 与蒸馏的最小测试。

### 必要但最小的验证

- SFT 与 LoRA 优先用极小样本和极短步数验证路径。
- 蒸馏先完成静态链路与张量语义验证，再决定是否运行。

### 风险与未验证边界

- 高阶训练对显存和权重来源更敏感，必须先确认基础 checkpoint。
- 不承诺效果提升、能力增强或 benchmark 分数，除非本机真实运行并记录。

### 完成标准

- 至少完成 SFT 或 LoRA 的一个最小可复现 smoke test。
- 对蒸馏链路能说明每个 loss 项和 mask 的作用。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/sft-smoke-test.md`
- 拟新增 `docs/lora-smoke-test.md`
- 拟新增 `docs/distillation-reading-notes.md`

## 阶段 10：面试表达与项目复盘材料

### 阶段目标

把真实阅读、手写、测试和实验结果整理成可用于面试和实习展示的材料。

### 需要阅读、同步、修改或手写的真实文件

- 本仓库 [README.md](../README.md)
- 本仓库 `docs/` 下已经完成或与本轮相关的源码拆解、实验记录与阶段性报告；长期规划当前包括 [docs/minimind-roadmap.md](minimind-roadmap.md)
- Learn MiniMind `interview/` 与 `docs/L22-L24`，仅作为追问准备素材
- 拟新增 `docs/interview-review.md`
- 拟新增 `docs/project-retrospective.md`

### 关键理解点

- 面试表达必须基于自己真实做过的实验和文档。
- 可以讲上游 MiniMind 的设计，但要明确“我阅读并复现/验证了哪些部分”。
- STAR 复盘要包含问题、行动、验证、失败边界和下一步，而不是只写结果。

### 可交付产物

- 拟新增 `docs/project-retrospective.md`
- 拟新增 `docs/interview-review.md`
- [README.md](../README.md) 中可验证的项目进展更新。

### 必要但最小的验证

- 每条简历或面试表述都能追溯到一个代码文件、测试、实验记录或文档。
- 对未完成事项明确写成“计划”或“待验证”。

### 风险与未验证边界

- 不要照搬 Learn MiniMind 的简历模板、STAR 示例或示例成果。
- 不要把“读过源码”包装成“训练出模型”。

### 完成标准

- 面试材料中每个技术点都有本仓库证据支撑。
- 能清晰回答：做了什么、为什么这样做、怎么验证、失败边界是什么、下一步怎么推进。

### 应沉淀的文档、测试或实验记录

- 拟新增 `docs/project-retrospective.md`
- 拟新增 `docs/interview-review.md`

## 近期最小执行顺序

1. 提交 [AGENTS.md](../AGENTS.md) 与 [.envrc](../.envrc)，建立项目协作和环境基线。
2. 生成拟新增 `docs/source-map.md`，完成个人仓库与上游源码入口映射。
3. 同步或最小复制 tokenizer、dataset 和模型相关源码时，逐项记录来源与改动边界。
4. 先完成 tokenizer / label / mask 最小测试，再推进模型模块手写复现。
5. 完成单 batch 训练闭环后，再运行 RTX 5060 Laptop GPU smoke test。
6. 基础闭环稳定后，再推进 SFT、LoRA、蒸馏、评估和面试材料。
