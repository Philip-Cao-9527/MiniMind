# 极小预训练闭环实验记录：2026-07-04

本记录对应一次“极小随机初始化单 batch smoke test”。它只验证训练链路能否闭环，不表示已经完成预训练，也不表示模型具备语言能力。

## 目标

验证以下链路在当前 WSL Ubuntu 本地仓库中真实可运行：

```text
JSONL -> tokenizer -> PretrainDataset -> DataLoader -> MiniMindForCausalLM forward -> loss -> backward -> optimizer.step -> 保存模型权重 -> 最小 checkpoint / resume 验证
```

## 代码版本与来源边界

- 个人仓库根目录：`/home/harry/projects/MiniMind`
- 本轮执行时个人仓库提交：`b504462f3b71c7ed0f6c9387f45830c472417547`
- 本轮同步的 MiniMind 核心源码来自本机上游参考仓库：`/home/harry/references/minimind`
- 上游参考提交：`512eed0b6556e741d80864f054d45d271459772a`
- 本轮同步的是最小源码集合，不是整仓复制；上游实现不写作个人原创成果。

本轮直接使用或新增的关键文件：

- 本地模型实现：[model_minimind.py](../model/model_minimind.py)
- 本地 Dataset 实现：[lm_dataset.py](../dataset/lm_dataset.py)
- 本地 tokenizer 配置：[tokenizer_config.json](../model/tokenizer_config.json)
- 本地 tokenizer 文件：[tokenizer.json](../model/tokenizer.json)
- 极小 JSONL 数据：[tiny_pretrain.jsonl](../experiments/tiny_pretrain_one_batch/tiny_pretrain.jsonl)
- 实验脚本：[run_tiny_pretrain_one_batch.py](../experiments/tiny_pretrain_one_batch/run_tiny_pretrain_one_batch.py)
- 运行日志：[run.log](../experiments/tiny_pretrain_one_batch/outputs/run.log)

## 环境

- Python 入口：`direnv exec . python`
- Python：`3.12.3`
- PyTorch：`2.7.1+cu128`
- CUDA：可用，`12.8`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- transformers：`4.57.6`
- datasets：`3.6.0`
- 本轮实际运行设备：CPU

## 实际命令

从仓库根目录执行：

```bash
mkdir -p experiments/tiny_pretrain_one_batch/outputs && direnv exec . python experiments/tiny_pretrain_one_batch/run_tiny_pretrain_one_batch.py --device cpu 2>&1 | tee experiments/tiny_pretrain_one_batch/outputs/run.log
```

## 数据与 tokenizer

极小 JSONL 共 4 条样本，字段为 `text`。本轮没有下载真实预训练数据、SFT 数据或模型权重。

tokenizer 实际信息：

- tokenizer 类：`PreTrainedTokenizerFast`
- BOS：`'<|im_start|>'`，id 为 `1`
- EOS：`'<|im_end|>'`，id 为 `2`
- PAD：`'<|endoftext|>'`，id 为 `0`

第一条样本 `今天天气很好。` 的 `PretrainDataset` 行为：

- `input_ids` 以 BOS 开始，以 EOS 结束，后续使用 PAD token id `0` 补齐到长度 `64`
- `labels` 复制 `input_ids`，但 PAD 位置被替换为 `-100`
- 本轮第一条样本中 `-100` 位置为 `7` 到 `63`
- PAD token id 是输入序列中的真实 token id；`-100` 只用于 labels，供 `cross_entropy(ignore_index=-100)` 忽略 padding 位置

## 模型配置

本轮使用随机初始化模型，不加载任何预训练权重。

- `hidden_size=128`
- `num_hidden_layers=2`
- `max_position_embeddings=64`
- `num_attention_heads=4`
- `num_key_value_heads=2`
- `head_dim=32`
- `vocab_size=6400`
- batch size：`2`
- learning rate：`1e-3`
- seed：`20260704`

说明：当前 `MiniMindConfig` 使用的长度字段是 `max_position_embeddings`，实验脚本中的 `max_seq_len=64` 同步传给 Dataset 和该配置字段。

## 运行结果

本轮成功项：

- `PretrainDataset` 正确读取 4 条 JSONL 样本
- `DataLoader` 生成 batch：`input_ids.shape=(2, 64)`，`labels.shape=(2, 64)`，dtype 均为 `torch.int64`
- forward 成功，`logits.shape=(2, 64, 6400)`
- loss 有限：`8.81688213`
- backward 成功，检查到 24 个参数梯度均为有限值
- `optimizer.step()` 成功
- 观察参数 `model.embed_tokens.weight` 发生变化，最大绝对变化为 `0.0010008216`
- 保存了仅包含模型 `state_dict` 的权重文件
- 保存了包含 model、optimizer、step、seed、config 等状态的 resume checkpoint
- 重新实例化模型和 optimizer 后加载 checkpoint，恢复 step 为 `1`，关键参数一致性为 `True`，optimizer state 条目数为 `24`

生成产物：

- 权重文件：`experiments/tiny_pretrain_one_batch/outputs/tiny_pretrain_state_dict.pth`
- resume checkpoint：`experiments/tiny_pretrain_one_batch/outputs/tiny_pretrain_resume_checkpoint.pth`
- 运行日志：`experiments/tiny_pretrain_one_batch/outputs/run.log`

## 失败与未验证边界

- 调试过程中发现 `MiniMindForCausalLM` 默认 tying embedding 后，`lm_head.weight` 不作为独立 `named_parameters()` 名称出现；最终脚本改为观察真实存在的 `model.embed_tokens.weight`。
- 本轮只在 CPU 上完成闭环；虽然环境中 CUDA 和 RTX 5060 Laptop GPU 可见，但没有运行 GPU 分支。
- 本轮没有运行正式 [train_pretrain.py](../trainer/train_pretrain.py)，没有验证 AMP、DDP、梯度累积、学习率调度、长时间训练或显存边界。
- 本轮没有证明 loss 可持续下降，也没有证明模型具备任何语言能力。
- 本轮没有下载或使用真实大数据、SFT 数据、模型权重。

## 结论

截至本轮命令执行完成，当前个人仓库已经具备一个可重复运行的极小预训练闭环 smoke test。它验证的是最短训练调用链和 checkpoint / resume 机制，不是正式预训练结果。
