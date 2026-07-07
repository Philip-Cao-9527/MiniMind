# MiniMind 源码导览：从预训练到生成闭环

本文档用于梳理当前本机 MiniMind 上游引用仓库的源码结构、关键调用链和学习顺序。它只记录本轮源码阅读结论，不表示当前个人仓库已经完成完整训练、SFT、LoRA、蒸馏、RL 或推理验证。

## 0. 本轮边界与证据

本轮个人仓库根目录是 `/home/harry/projects/MiniMind`。已读取的个人仓库治理与路线文档包括 [AGENTS.md](../AGENTS.md)、[README.md](../README.md)、[.envrc](../.envrc)、[docs/minimind-roadmap.md](minimind-roadmap.md)。当前个人仓库存在既有未提交改动，并且 `model/`、`dataset/`、`trainer/` 等源码目录在 `git status` 中显示为未跟踪；这些文件虽然与本地上游同名文件逐字节一致，但本轮不把它们表述为个人仓库已经完成的正式实现基线。

本轮上游引用仓库真实路径是 `/home/harry/references/minimind`，当前提交为：

```text
512eed0b6556e741d80864f054d45d271459772a
```

本文中的“上游源码事实”均来自这个本地上游引用提交。重点阅读的上游文件包括：

- [上游 README.md](../../../references/minimind/README.md)：训练阶段、数据、权重和 checkpoint 约定。
- [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)：`PretrainDataset`、`SFTDataset`、`DPODataset`、`RLAIFDataset`、`AgentRLDataset`。
- [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py)：`MiniMindConfig`、`MiniMindModel`、`MiniMindForCausalLM`、`forward`、`generate`、KV Cache。
- [上游 model/model_lora.py](../../../references/minimind/model/model_lora.py)：手写 LoRA 注入、保存、加载、合并。
- [上游 model/tokenizer_config.json](../../../references/minimind/model/tokenizer_config.json)：BOS、EOS、PAD、chat template。
- [上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)：预训练入口。
- [上游 trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)：全参数 SFT 入口。
- [上游 trainer/train_lora.py](../../../references/minimind/trainer/train_lora.py)：LoRA 微调入口。
- [上游 trainer/train_distillation.py](../../../references/minimind/trainer/train_distillation.py)：白盒蒸馏入口。
- [上游 trainer/train_dpo.py](../../../references/minimind/trainer/train_dpo.py)：DPO 入口。
- [上游 trainer/train_ppo.py](../../../references/minimind/trainer/train_ppo.py)：PPO 入口。
- [上游 trainer/train_grpo.py](../../../references/minimind/trainer/train_grpo.py)：GRPO / CISPO 入口。
- [上游 trainer/train_agent.py](../../../references/minimind/trainer/train_agent.py)：Agent RL / Tool Use RL 入口。
- [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)：模型初始化、学习率、checkpoint、resume、分布式与 sampler 工具。
- [上游 eval_llm.py](../../../references/minimind/eval_llm.py)：命令行推理入口。
- [上游 scripts/serve_openai_api.py](../../../references/minimind/scripts/serve_openai_api.py)：OpenAI 兼容接口服务。
- [上游 scripts/chat_api.py](../../../references/minimind/scripts/chat_api.py)：OpenAI SDK 调用示例。
- [上游 scripts/convert_model.py](../../../references/minimind/scripts/convert_model.py)：torch 权重、Transformers 权重和 LoRA 合并转换。

未执行内容：本轮没有下载数据、没有运行正式训练、没有加载真实权重推理、没有做 GPU smoke test。文中的性能、效果、收敛、显存建议如果没有本轮命令证据，均标注为工程建议或待后续验证。

## 1. 仓库结构与职责地图

上游引用仓库顶层当前包含 `dataset`、`model`、`trainer`、`scripts`、`images` 以及根目录推理入口 [上游 eval_llm.py](../../../references/minimind/eval_llm.py)。学习主线不是按目录机械阅读，而是按“数据如何变成 loss、loss 如何更新参数、权重如何被推理入口读取”来串起来。

`model/` 的核心文件是 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py)。这里定义模型配置、Decoder-only Transformer 主体、Causal LM 输出头、shifted cross entropy loss 和自定义 `generate`。对初学者来说，它是理解 Decoder-only LLM 的中心文件。`MiniMindConfig` 决定 `hidden_size`、层数、head 数、KV head 数、词表大小、RoPE、MoE 等结构参数；`MiniMindModel` 负责 token embedding、多个 Transformer block、RMSNorm 和 RoPE buffer；`MiniMindForCausalLM` 在底层 Transformer 之后接 `lm_head`，把每个位置的 hidden state 映射成词表维度 logits，并在 `labels` 存在时计算因果语言模型 loss。

`model/` 里的 [上游 model/tokenizer.json](../../../references/minimind/model/tokenizer.json) 与 [上游 model/tokenizer_config.json](../../../references/minimind/model/tokenizer_config.json) 是 tokenizer 词表、特殊 token 和 chat template 的来源。当前上游配置中 `bos_token` 是 `<|im_start|>`，id 为 `1`；`eos_token` 是 `<|im_end|>`，id 为 `2`；`pad_token` 是 `<|endoftext|>`，id 为 `0`。`tokenizer_config.json` 还定义了对话模板如何把 `system/user/assistant/tool` 消息展开成训练和推理文本。

`dataset/` 的核心文件是 [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)。`PretrainDataset` 读取 JSONL 的 `text` 字段，把文本编码成 token 序列，手工加 BOS/EOS，padding 到固定长度，并把 PAD 位置的 label 改成 `-100`。`SFTDataset` 读取 `conversations`，先用 tokenizer 的 chat template 展开消息，再只让 assistant 回复区间参与 loss。`DPODataset`、`RLAIFDataset`、`AgentRLDataset` 是偏好学习、RLAIF 和 Agent RL 的进阶分支，当前阶段先知道职责即可，不建议抢先细读。

`trainer/` 是训练入口和训练工程逻辑。初学者先读 [上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)，它串起 tokenizer、dataset、DataLoader、模型、loss、backward、AdamW、AMP、保存权重和 resume。然后读 [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)，这里的 `init_model` 决定随机初始化还是从 `out` 加载普通权重，`lm_checkpoint` 决定普通权重和 resume checkpoint 的保存/恢复结构。之后再读 [上游 trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)，它与预训练训练循环高度相似，但 dataset 换成 `SFTDataset`，默认从 `pretrain` 权重开始。

`scripts/` 是服务、转换和辅助入口。[上游 scripts/serve_openai_api.py](../../../references/minimind/scripts/serve_openai_api.py) 把模型包装成 OpenAI 兼容 `/v1/chat/completions` 服务；[上游 scripts/chat_api.py](../../../references/minimind/scripts/chat_api.py) 是客户端调用示例；[上游 scripts/convert_model.py](../../../references/minimind/scripts/convert_model.py) 负责在原生 torch 权重和 Transformers 目录之间转换，也包含 LoRA 合并导出。

根目录 [上游 eval_llm.py](../../../references/minimind/eval_llm.py) 是最重要的本地推理入口。它读取 tokenizer 和权重，构造 prompt 或 chat template，调用 `model.generate`，再 decode 新生成 token。它只读取模型状态，不做梯度更新。

上游权重和输出目录约定来自 [上游 README.md](../../../references/minimind/README.md)、[上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py) 与 [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py)：普通可推理权重默认保存到 `out`，如 `pretrain_768.pth`、`full_sft_768.pth`；完整续训状态默认保存到 `checkpoints`，如 `pretrain_768_resume.pth`、`full_sft_768_resume.pth`。本文不把这些文件链接为 Markdown，因为当前个人仓库和上游引用仓库中未确认存在这些具体权重文件。

当前阶段建议先读：

1. [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py) 的 `PretrainDataset.__getitem__`。
2. [上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py) 的参数、`train_epoch` 和主函数。
3. [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py) 的 `init_model`、`lm_checkpoint`、`SkipBatchSampler`。
4. [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py) 的 `MiniMindConfig`、`MiniMindModel.forward`、`MiniMindForCausalLM.forward`、`MiniMindForCausalLM.generate`。
5. [上游 eval_llm.py](../../../references/minimind/eval_llm.py) 的 `init_model` 和 `main`。

当前先跳过：DPO、PPO、GRPO、Agent RL、ToolCall 评测、WebUI 和模型转换的细节。它们都复用主模型和 tokenizer，但会引入偏好对、rollout、reward、KL、value model、工具调用解析等额外变量，容易干扰“先把 Causal LM 闭环看懂”的主线。

## 2. 预训练调用链

预训练主线从 JSONL 样本开始，当前上游 README 示例格式是：

```json
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

真实调用链如下：

```text
JSONL text
-> AutoTokenizer.from_pretrained("../model")
-> PretrainDataset.__getitem__
-> input_ids / labels
-> DataLoader
-> MiniMindConfig
-> init_model -> MiniMindForCausalLM
-> MiniMindForCausalLM.forward
-> MiniMindModel.forward
-> logits
-> logits[..., :-1, :] 与 labels[..., 1:]
-> F.cross_entropy(ignore_index=-100)
-> loss / accumulation_steps
-> scaler.scale(loss).backward()
-> clip_grad_norm_
-> AdamW.step
-> zero_grad
-> 普通权重保存到 out
-> resume checkpoint 保存到 checkpoints
```

入口文件是 [上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)。主函数先解析训练参数：默认 `batch_size=32`、`learning_rate=5e-4`、`dtype=bfloat16`、`accumulation_steps=8`、`hidden_size=768`、`num_hidden_layers=8`、`max_seq_len=340`、`data_path=../dataset/pretrain_t2t_mini.jsonl`、`from_weight=none`、`from_resume=0`。这是上游源码事实，不是本机已验证可承受配置。RTX 5060 Laptop 约 8GB 显存下，后续真实训练前应保守降低 `batch_size`、`max_seq_len`、`num_workers`，并先跑极小 smoke test。

`PretrainDataset.__getitem__` 的核心逻辑是：

```python
tokens = tokenizer(text, add_special_tokens=False, max_length=max_length - 2, truncation=True).input_ids
tokens = [bos_token_id] + tokens + [eos_token_id]
labels = input_ids.clone()
labels[input_ids == pad_token_id] = -100
```

这段代码说明三件事。第一，预训练样本不是由 Dataset 自己创建 tokenizer；tokenizer 在训练入口通过 `init_model` 返回，再传给 Dataset。第二，`add_special_tokens=False` 表示这次编码不让 tokenizer 自动添加 BOS/EOS，因为上游后面手工加了 BOS 和 EOS。第三，`max_length - 2` 给手工 BOS/EOS 预留两个位置；超过长度的文本会被截断。

`DataLoader` 在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L161) 中使用 `batch_sampler=SkipBatchSampler(...)`、`num_workers=args.num_workers`、`pin_memory=True`。这里的 DDP 指 PyTorch 的 `DistributedDataParallel`，也就是“多进程/多 GPU 分布式数据并行训练”。它的基本思路是：启动多个训练进程，每个进程持有一份模型副本，分别处理不同的数据 batch；反向传播后，各进程之间同步梯度，让每份模型副本做一致的参数更新。上游入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L110) 调用 `init_distributed_mode()`，如果环境变量里没有分布式训练需要的 `RANK`，就走普通单进程训练；如果用 `torchrun --nproc_per_node N train_pretrain.py` 这类方式启动，就会初始化 DDP，并在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L154) 用 `DistributedDataParallel` 包住模型。

DDP 会影响 sampler，是因为多进程训练时不能让每个进程都完整遍历同一份数据，否则等于同一批样本被不同 GPU 重复训练。`DistributedSampler` 的职责就是按进程 rank 切分数据索引：比如 2 个进程时，rank 0 处理一部分样本，rank 1 处理另一部分样本；每个 epoch 再通过 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L158) 的 `train_sampler.set_epoch(epoch)` 改变 shuffle 顺序，避免每轮切分顺序固定。没有 DDP 时，训练入口只需要用 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L159) 的 `torch.randperm(len(train_ds)).tolist()` 生成一份随机样本索引；有 DDP 时，则使用 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L136) 的 `DistributedSampler`。这行代码：

```python
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

可以拆成三层理解。`train_sampler or indices` 是“样本索引来源”：DDP 模式下优先用 `DistributedSampler`，非 DDP 模式下用随机打乱后的 `indices` 列表。`args.batch_size` 是每个 micro-batch 包含多少条样本。`skip` 是 resume 时要跳过多少个已经训练过的 batch；没有 resume 时它是 0。随后 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L162) 把这个 `batch_sampler` 交给 `DataLoader`，所以 DataLoader 不再自己决定 batch 怎么切，而是照着 `SkipBatchSampler` 产出的索引列表去取样本。

`SkipBatchSampler` 的实现位于 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L134)。它不是一个“读取数据”的类，它只做一件事：把样本下标组织成一批一批的下标，并在断点续训时跳过前面已经训练过的 batch。真正读取 `input_ids, labels` 的仍然是 Dataset；真正按下标取样本的是 DataLoader。

先把上游源码短片段摆出来，再逐行看它的行为。下面代码来自 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L134)：

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
```

带注释版可以这样读：

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler          # 样本下标来源：DistributedSampler 或随机 indices
        self.batch_size = batch_size    # 每多少个样本下标组成一个 batch
        self.skip_batches = skip_batches  # resume 时要丢弃的已训练 batch 数

    def __iter__(self):
        batch = []      # 临时收集当前 batch 的样本下标
        skipped = 0     # 已经跳过了几个完整 batch
        for idx in self.sampler:        # 逐个遍历样本下标，不读取样本内容
            batch.append(idx)           # 把当前下标放入临时 batch
            if len(batch) == self.batch_size:  # 凑够一个完整 batch
                if skipped < self.skip_batches:
                    skipped += 1        # 这个 batch 属于已训练进度，只计数
                    batch = []          # 丢弃这个 batch 的下标
                    continue            # 继续凑下一个 batch
                yield batch             # 把本 batch 下标交给 DataLoader
                batch = []              # 清空，开始收集下一个 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch                 # 处理最后不足 batch_size 的尾 batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
```

逐行解释如下：

- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L135)：`sampler` 保存样本下标来源，可以是 DDP 的 `DistributedSampler`，也可以是普通 Python list `indices`。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L137)：`batch_size` 决定几个样本下标合成一个 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L138)：`skip_batches` 决定本轮开头丢弃几个完整 batch，用于 resume。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L141)：`batch = []` 是临时篮子，先收集样本下标。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L142)：`skipped = 0` 记录已经跳过了几个 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L143)：`for idx in self.sampler` 逐个拿样本下标，不拿样本内容。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L144)：`batch.append(idx)` 把当前样本下标放进临时 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L145)：只有 `len(batch) == batch_size` 时，才说明凑够了一个完整 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L146)：如果 `skipped < skip_batches`，这个完整 batch 属于上次已经训练过的进度。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L147)：`skipped += 1` 只增加“已跳过 batch 数”，不更新模型。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L148)：`batch = []` 清空这个被跳过的 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L149)：`continue` 回到循环开头，继续收集下一个 batch。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L150)：如果不需要跳过，`yield batch` 把这一批下标交给 DataLoader。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L152)：遍历结束后，如果还剩一个不满 `batch_size` 的尾 batch，并且已经完成应跳过的 batch，也把尾 batch 交出去。
- [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L156)：`__len__` 用向上取整计算总 batch 数，再减去 `skip_batches`，让训练入口能估算本轮还有多少 step。

举例：假设随机索引是 `[8, 3, 5, 1, 9, 0, 2]`，`batch_size=2`，没有 resume 时 `skip=0`，产出的 batch 是 `[8,3]`、`[5,1]`、`[9,0]`、`[2]`。如果从 checkpoint 恢复，`start_step=2`，训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L160) 令 `skip=2`，那么前两个完整 batch `[8,3]`、`[5,1]` 会被跳过，本轮从 `[9,0]` 开始继续。这里跳过的是 batch，不是样本，也不是 token；它服务的是断点续训的“数据进度对齐”。如果不跳过，恢复训练会重复消费前两个 batch，相当于同一段数据被多训练一次；如果跳过太多，又会漏掉本来没训练过的数据。

DataLoader 输出的一个 batch 是：

```text
input_ids: LongTensor, shape = [batch_size, max_seq_len]
labels:    LongTensor, shape = [batch_size, max_seq_len]
```

当前预训练链路没有从 Dataset 返回 `attention_mask`，训练时调用也是 `model(input_ids, labels=labels)`，所以 `MiniMindForCausalLM.forward` 收到的 `attention_mask` 默认为 `None`。这意味着训练中的 padding 不通过 attention mask 屏蔽；PAD 位置主要通过 label 中的 `-100` 不贡献 loss。注意这不等于 PAD 完全不影响有效 token 的上下文，这是后续需要进一步用最小样本验证的风险点。

模型初始化在 [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py) 的 `init_model`：

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM(lm_config)
if from_weight != "none":
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
```

当 `from_weight='none'` 时，模型是随机初始化，只能开始从头训练。预训练入口默认就是这个状态。当 `from_weight='pretrain'` 或 `full_sft` 时，会从 `out/<权重名>_<hidden_size>.pth` 读取普通权重，再加载进模型。`strict=False` 允许权重字典和模型结构存在缺失或多余 key 时不直接报错，这给 LoRA、转换或结构变化留了空间，但风险是你可能以为加载成功，实际有部分层没有加载或有权重被忽略；后续实验应打印 missing/unexpected keys 或改为更严格检查。

前向传播在 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py)。`MiniMindForCausalLM.forward` 先调用底层 `MiniMindModel.forward` 得到：

```text
hidden_states: [B, T, hidden_size]
past_key_values: 每层一个 K/V cache，训练默认 use_cache=False
aux_loss: MoE 辅助损失；dense 模型为 0
```

然后 `lm_head` 把 hidden state 投影成：

```text
logits: [B, T, vocab_size]
```

如果传入 `labels`，上游用 shifted cross entropy：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(x.view(-1, vocab_size), y.view(-1), ignore_index=-100)
```

直观理解是：第 `t` 个位置的 logits 用来预测第 `t+1` 个 token。模型输入仍然是一整段序列，训练时可以并行算出所有位置的 logits；但每个位置的监督目标是下一个 token。`ignore_index=-100` 让 PAD 对应的 label 不参与 loss。

这里的 cross entropy 可以先按单个位置理解。假设某个位置的 logits 是 $z_1,\dots,z_V$，$V$ 是词表大小；softmax 先把 logits 转成概率：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

如果这个位置的真实目标 token id 是 $y$，那么这个位置的交叉熵是：

$$
L_{\text{CE, one}} = -\log p_y
$$

MiniMind 的 shifted cross entropy 是把这个公式用到所有有效的“当前位置预测下一个 token”位置上。设 $z_{b,t,v}$ 是第 $b$ 条样本、第 $t$ 个位置、词表 token $v$ 的 logits；设 $y_{b,t+1}$ 是 `labels[..., 1:]` 中对应的目标 token；有效监督位置集合为：

$$
S = \{(b,t)\mid labels[b,t+1] \ne -100,\ 0 \le t < T-1\}
$$

那么 `res.loss` 对应：

$$
L_{\text{CE}} =
-\frac{1}{|S|}
\sum_{(b,t)\in S}
\log
\frac{e^{z_{b,t,y_{b,t+1}}}}{\sum_{v=1}^{V} e^{z_{b,t,v}}}
$$

后文 `6.5 shifted cross entropy loss` 会把 softmax、one-hot 交叉熵、`ignore_index=-100` 和 `loss = res.loss + res.aux_loss` 的关系再展开一遍。

参数更新发生在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L24) 的 `train_epoch`。在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L36) 中，`res = model(input_ids, labels=labels)` 会调用 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L245) 的 `MiniMindForCausalLM.forward`。这个 forward 已经算出 `res.loss`，也就是 shifted cross entropy；如果模型启用 MoE，还会从每层 MoE FFN 中累加 `res.aux_loss`。

训练入口的总 loss 是 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L37) 的：

$$
L_{\text{train}} = L_{\text{CE}} + L_{\text{aux}}
$$

梯度累积时，实际传给 `backward()` 的是：

$$
L_{\text{backward}} = \frac{L_{\text{train}}}{\text{accumulation\_steps}}
$$

其中 $L_{\text{CE}} = \text{res.loss}$，来自 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L251) 的 shifted cross entropy；$L_{\text{aux}} = \text{res.aux\_loss}$，来自 MoE 路由辅助损失。dense 模型没有 MoE 层时，$L_{\text{aux}} = 0$，所以 $L_{\text{train}} = L_{\text{CE}}$。MoE 模型中，单层 MoE 的辅助项源码在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L171)：先统计每个 expert 的实际负载 $load_e$，再乘以路由概率均值 $prob_e$，最后乘以 expert 数 $E$ 和 `router_aux_loss_coef`。可以近似记成：

$$
L_{\text{aux, layer}} = \lambda \cdot E \cdot \sum_{e=1}^{E} load_e \cdot prob_e
$$

$$
L_{\text{aux}} = \sum_{\text{layer}} L_{\text{aux, layer}}
$$

它的作用不是让模型预测下一个 token，而是约束 MoE 路由不要长期偏向少数 expert。对 dense 模型来说这项不存在；对 MoE 模型来说，训练目标就是“语言模型预测损失 + 路由均衡损失”。

`scaler.scale(loss).backward()` 计算梯度；每到 `step % accumulation_steps == 0` 时，先 `scaler.unscale_(optimizer)`，再 `clip_grad_norm_`，然后 `scaler.step(optimizer)`、`scaler.update()`、`optimizer.zero_grad(set_to_none=True)`。真正改变模型参数的是 `optimizer.step`，准确说在 AMP 封装下由 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L46) 的 `scaler.step(optimizer)` 调用。`backward` 只把梯度累积到参数的 `.grad`，不会直接更新参数。

AMP 分支是训练链路的一部分，配置位置在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L119)，使用位置在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L35)。如果设备是 CPU，源码使用 `nullcontext()`；如果设备包含 `cuda`，源码使用 `torch.cuda.amp.autocast(dtype=dtype)`。`dtype` 参数只有 `bfloat16` 和 `float16` 两种分支：`args.dtype == "bfloat16"` 时用 `torch.bfloat16`，否则用 `torch.float16`。[上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L137) 的 `GradScaler(enabled=(args.dtype == "float16"))` 只在 fp16 时启用；bf16 分支不启用 scaler。bf16/fp16 的指数位、尾数位、数值范围和工业实践差异在本文 `6.8` 详细解释；这里先记住它只影响训练 forward/loss/backward 的数值计算方式，不改变 Dataset、labels 或模型结构。

保存逻辑有两类文件。训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L68) 直接把模型 `state_dict` 转成 half CPU tensor 后保存到 `args.save_dir`，例如 `../out/pretrain_768.pth`。随后在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L69) 调用 `lm_checkpoint`；`lm_checkpoint` 的实现位于 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L63)。普通权重是单纯的模型参数字典，适合推理或后续阶段初始化；resume checkpoint 是一个包含 `model`、`optimizer`、`epoch`、`step`、`world_size`、`wandb_id`、`scaler` 等状态的字典，适合中断后继续训练。普通权重不能恢复 optimizer 动量、AMP scaler 和数据跳过进度；resume checkpoint 可以恢复这些状态。

resume 恢复链路是：训练入口设置 `--from_resume 1` 后，[上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L117) 先调用 `lm_checkpoint(..., save_dir="../checkpoints")` 的加载分支；如果找到 `<weight>_<hidden_size>_resume.pth`，[上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L109) 会用 `torch.load(..., map_location="cpu")` 读出 checkpoint。随后训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L143) 恢复 `model`、`optimizer`、`scaler`，并把 `start_epoch`、`start_step` 设为保存值。`SkipBatchSampler` 会跳过已经训练过的 batch。需要注意，checkpoint 中的 `model` 已经保存为 half CPU tensor，加载回训练模型后是否对数值精度、optimizer 状态和后续训练稳定性有影响，属于后续最小 resume smoke test 需要验证的内容。

## 3. 推理 / 生成调用链

推理入口是 [上游 eval_llm.py](../../../references/minimind/eval_llm.py)。它的调用链是：

```text
prompt 或 conversation messages
-> tokenizer.apply_chat_template 或 bos_token + prompt
-> tokenizer(..., return_tensors="pt", truncation=True)
-> init_model 加载 tokenizer 和模型权重
-> model.eval().half().to(device)
-> model.generate
-> forward(input_ids[:, past_len:], past_key_values, use_cache=True)
-> logits[:, -1, :]
-> temperature / repetition_penalty / top-k / top-p
-> sample 或 argmax 得到 next_token
-> 拼接到 input_ids
-> 更新 past_key_values
-> 遇到 eos_token_id 停止
-> tokenizer.decode 新生成 token
```

`eval_llm.py` 的 `init_model` 有两条分支。如果 `args.load_from` 字符串包含 `model`，它使用上游原生 `MiniMindForCausalLM(MiniMindConfig(...))`，再从 `./out/<weight>_<hidden_size>.pth` 加载普通 torch 权重，且 `strict=True`。如果 `load_from` 不是本地原生 `model` 路径，则使用 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载 Transformers 格式模型。加载后统一 `model.half().eval().to(args.device)`。这是推理只读状态，不会更新参数。

prompt 处理有两种。若 `weight` 名称包含 `pretrain`，上游直接用 `tokenizer.bos_token + prompt`；否则用：

```python
tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=...)
```

这会把消息展开成当前 tokenizer 配置中的 `<|im_start|>user...<|im_end|>`、`<|im_start|>assistant\n<think>...` 等模板文本。再由 `tokenizer(..., return_tensors="pt", truncation=True)` 变成 `input_ids` 和 `attention_mask`。与预训练不同，推理链路会把 `attention_mask` 传给 `generate`。

这里一定要区分，不是因为“预训练模型结构”和“SFT 模型结构”不同，而是因为这两类权重在训练时见过的输入分布不同，学到的“接下来该怎么续写”的习惯也不同。上游预训练数据在 [上游 PretrainDataset.__getitem__](../../../references/minimind/dataset/lm_dataset.py#L47) 里只是把普通 `text` 编码后，手工包上 `bos_token` 和 `eos_token`；它的目标很朴素，就是看到一串自然文本后继续预测下一个 token。这里虽然 `bos_token` 的字符串本身也是 `<|im_start|>`，但在预训练阶段，它主要承担“样本起点”的边界作用，不等于模型已经学会完整的聊天协议，更不等于它天然知道 `user`、`assistant`、`tool` 这些角色段落该如何组织。

SFT/后训练就不一样了。上游 [上游 SFTDataset.create_chat_prompt](../../../references/minimind/dataset/lm_dataset.py#L71) 会先把 `conversations` 展开成 chat template 文本，再在 [上游 SFTDataset.generate_labels](../../../references/minimind/dataset/lm_dataset.py#L88) 里只让 assistant 回复区间参与 loss。换句话说，SFT 权重学到的不是“见到任意纯文本就继续写”，而是“见到一段符合模板的多轮消息，并且最后已经出现 assistant 起始前缀后，我应该站在 assistant 这个角色继续往下补全”。这相当于把“你现在该回答了”这个信号，编码进了 prompt 结构本身。

所以推理时要尽量复现各自训练时的输入形态。给预训练权重喂 `bos_token + prompt`，是在模拟它最熟悉的“单段文本续写”场景；给 SFT/后训练权重喂 chat template，是在模拟它训练时真正优化过的“用户消息 -> assistant 回答”场景。如果把这两者弄反，会出现明显的分布错位：预训练权重可能把整段聊天模板当成一串生硬的特殊 token 和角色文本来续写，回答风格不稳定；SFT 权重如果只看到裸 prompt，没有 assistant 起始位置信号，就可能不知道应该从哪个角色继续、是否该进入回答区、回复边界该怎么组织。

可以把它记成一句话：推理 prompt 不是随便包装一下文本，而是在尽量对齐“这份权重当年是按什么格式学会生成的”。当前 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L73) 用 `if 'pretrain' in args.weight` 来分支，本质上是一个基于权重命名约定的工程捷径。它默认认为名字里带 `pretrain` 的权重更接近纯续写模型，其余 `full_sft`、`ppo_actor`、`grpo` 等权重都应按聊天模板来驱动。这个判断在上游默认命名里是成立的，但如果后续你自己改了权重命名，就要自己保证“权重名称”和“实际训练阶段”仍然语义一致，否则推理入口可能会走错分支。

`MiniMindForCausalLM.generate` 在 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py) 中自定义实现。关键参数含义如下：

- `max_new_tokens`：最多追加多少个新 token。增大会增加最长生成时间和 KV Cache 显存占用；不等于模型已经真正具备长文本能力。
- `temperature`：logits 除以温度后采样。越高越随机；太低更接近贪心；为 0 在当前实现中会除零，因此不应传 0。
- `top_k`：只保留概率最高的 k 个候选；越小越保守，越大候选越多。
- `top_p`：按累计概率保留候选；越小越保守，越大越开放。
- `repetition_penalty`：对已出现 token 的 logits 做惩罚；大于 1 会降低重复倾向，但过大可能损害连贯性。
- `eos_token_id`：如果新 token 等于 EOS，则将该序列标为完成；全部完成后 `break`。
- `use_cache`：为 True 时复用每层历史 K/V，推理速度更快，但会随生成长度占用更多显存。

生成循环每步先计算 `past_len`。如果已有 cache，只把 `input_ids[:, past_len:]` 送入 forward，也就是通常只送入最新 token；如果没有 cache，则送入当前完整序列。`forward` 返回 `outputs.past_key_values` 后，`generate` 把它保存给下一步。每层 cache 的形态来自 attention 内部：未 repeat 前，K/V 形状近似为 `[B, total_seq_len, num_key_value_heads, head_dim]`；当新 token 到来时，当前步新 K/V 与历史 `past_key_value` 在序列维 `dim=1` 拼接，所以 `total_seq_len` 每步增加 1。

EOS 停止发生在采样之后。当前实现先生成 `next_token`，再执行：

```text
finished |= next_token == eos_token_id
if finished.all(): break
```

如果一个 batch 中某条序列已经完成，后续会用 EOS 填充它的 next token，直到所有序列完成或达到 `max_new_tokens`。最后 `eval_llm.py` 用 `tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)` 只解码 prompt 之后的新 token。

推理与训练共享的模型逻辑是：token embedding、Transformer block、RoPE、attention、MLP、RMSNorm、`lm_head`、`forward` 产出 logits。训练额外需要 `labels`、shifted CE、`backward`、optimizer、checkpoint；推理额外需要 chat template、采样策略、KV Cache、EOS 停止和 decode。

## 4. Decoder-only 与 Encoder-only 的衔接

你已有 Encoder-only Transformer 基础，可以这样迁移理解 Decoder-only LLM：

| 维度 | Encoder-only | Decoder-only Causal LM |
| --- | --- | --- |
| attention 可见范围 | 通常双向可见，位置 `t` 可以看左右上下文 | 因果可见，位置 `t` 只能看自己和历史位置 |
| 训练目标 | 常见是分类、MLM、句向量、token 标注等 | next-token prediction，预测下一个 token |
| 输出形式 | 每个 token 或 `[CLS]` 表征常接任务头 | 每个位置输出词表 logits |
| 是否天然生成 | 不天然逐 token 自回归生成 | 天然按“上文 -> 下一个 token”生成 |
| mask 重点 | padding mask、任务 mask | causal mask、padding mask、loss mask、KV Cache |

causal mask 限制的是 attention 中当前位置能读取哪些 key/value。对于长度为 `T` 的序列，位置 `t` 不能看 `t+1...T-1` 的 token，否则训练时模型会偷看答案。上游 attention 的非 flash 路径中用上三角 `-inf` 加到 attention scores 上，softmax 后未来位置概率接近 0。

训练时能并行计算全序列 loss，是因为真实完整序列已经给定。模型一次 forward 得到每个位置的 logits，但每个位置的 attention 都被 causal mask 限制，只能看过去。因此它没有偷看未来，却可以利用矩阵并行把所有位置的预测同时算出来。推理时不同：第 `t+1` 个 token 还不存在，必须先采样出第 `t+1` 个 token，再把它追加到上下文中生成第 `t+2` 个 token，所以推理仍需逐 token 自回归。

`MiniMindForCausalLM` 是“底层 Transformer + 语言模型头 + 生成能力”的封装。底层 `MiniMindModel` 只把 `input_ids` 变成 hidden states；`lm_head` 把 hidden states 投影到 `vocab_size`，得到每个候选 token 的 logits；`forward` 负责训练时可选计算 loss；`generate` 负责推理时循环采样。它继承 `PreTrainedModel` 和 `GenerationMixin`，但当前上游重写了 `generate`，所以推理主循环以本文件实现为准。

`MiniMindConfig` 决定模型结构。`hidden_size` 越大，embedding、attention、MLP、lm_head 参数和激活显存越大；`num_hidden_layers` 越多，计算更深，KV Cache 层数也更多；`num_attention_heads` 与 `num_key_value_heads` 决定 GQA 形态；`vocab_size=6400` 决定 embedding 和输出层大小；`max_position_embeddings` 决定 RoPE buffer 长度；`use_moe=True` 会把 FFN 换成 MoE 路由专家并引入 `aux_loss`。

随机初始化模型、加载预训练权重、加载 SFT 权重、加载 resume checkpoint 的含义不同：

- 随机初始化：只有结构，没有语言知识，适合预训练从 0 开始。
- 加载预训练权重：模型已有基础语言建模能力，可继续 SFT 或推理基座续写。
- 加载 SFT 权重：模型已适配对话模板和 assistant 风格，适合 `eval_llm.py --weight full_sft` 这类聊天推理。
- 加载 resume checkpoint：恢复训练状态，不只是模型参数，还包括 optimizer、scaler、epoch、step 等，用于中断续训。

`state_dict` 是 PyTorch 模块参数和 buffer 的字典。`torch.load` 从磁盘反序列化权重或 checkpoint。`load_state_dict` 把字典加载进模型。`strict=True` 要求 key 严格匹配；`strict=False` 允许缺失或多余 key。`strict=False` 的风险是结构不一致时仍然继续运行，可能导致某些层保持随机初始化或某些权重被忽略。普通权重文件通常就是模型 `state_dict`；resume checkpoint 是包含模型和训练器状态的复合字典。

## 5. tokenizer、BOS、EOS、PAD、labels 与 `-100`

tokenizer 是文本和 token id 的双向映射器。它负责把字符串拆成 token id，也负责把生成的 token id decode 回字符串。Dataset 内不一定 import tokenizer，因为 tokenizer 通常由训练入口统一加载后传入 Dataset；这样能保证训练、推理、SFT、LoRA 等阶段使用同一套词表和特殊 token。

当前上游 tokenizer 事实来自 [上游 model/tokenizer_config.json](../../../references/minimind/model/tokenizer_config.json)，本轮也用 `AutoTokenizer.from_pretrained('/home/harry/references/minimind/model')` 做了静态核验：

```text
BOS: <|im_start|>, id = 1
EOS: <|im_end|>, id = 2
PAD: <|endoftext|>, id = 0
```

`input_ids` 是模型输入序列。`labels` 是监督目标序列。预训练中二者一开始 clone，是因为 Causal LM 的目标就是预测同一条序列的下一个 token；但 clone 后二者职责立刻分开：`input_ids` 仍保留 PAD 作为输入占位，`labels` 中 PAD 位置被改成 `-100`，表示这些位置不参与 loss。

BOS 是输入边界信号，让模型知道一段文本或一轮消息开始了。EOS 是结束信号，训练时它是一个需要被预测出来的目标；推理时它也是生成停止条件。PAD 是 batch 对齐用的填充 token，使不同长度样本可以组成同样长度的张量。PAD 可以存在于 `input_ids`，但不应该贡献 loss。`-100` 不是 token id，不来自 tokenizer，也不会出现在正常 tokenizer 生成的 `input_ids` 中；它只是 PyTorch `F.cross_entropy(ignore_index=-100)` 使用的忽略标记。

padding、attention mask、causal mask、loss mask 解决的是四个不同问题：

- padding：把短序列补齐到固定长度，便于组成 batch。
- attention mask：告诉 attention 哪些输入 token 是有效 token，哪些是 PAD，不应被看见。
- causal mask：告诉 Decoder-only attention 不能看未来 token。
- loss mask：告诉 loss 哪些 label 应参与监督，哪些应忽略。MiniMind 预训练链路通过 label 中的 `-100` 实现 loss mask。

当前上游预训练 `PretrainDataset` 不返回 `attention_mask`，训练入口也没有传 attention mask。推理链路会从 tokenizer 得到 `attention_mask` 并传给 `generate`。这两条链路不能混为一谈。

下面用短文本 `你好`、`max_length=8` 展示预训练样本。当前 tokenizer 将 `你好` 编为一个 token id `1968`；`convert_ids_to_tokens` 的原始 token 字符串显示为字节级片段，本文用 decode 后语义解释。

```text
原始文本：你好
原始 token ids：1968
手工加边界：BOS, 你好, EOS
padding 到 8：BOS, 你好, EOS, PAD, PAD, PAD, PAD, PAD
```

| 位置 | input_ids | token 语义 | labels | 是否是有效输入 | 该 label 是否参与 loss |
| --- | ---: | --- | ---: | --- | --- |
| 0 | 1 | BOS `<|im_start|>` | 1 | 是 | 是，但 shift 后不会作为被预测目标 |
| 1 | 1968 | `你好` | 1968 | 是 | 是 |
| 2 | 2 | EOS `<|im_end|>` | 2 | 是 | 是 |
| 3 | 0 | PAD `<|endoftext|>` | -100 | 否 | 否 |
| 4 | 0 | PAD | -100 | 否 | 否 |
| 5 | 0 | PAD | -100 | 否 | 否 |
| 6 | 0 | PAD | -100 | 否 | 否 |
| 7 | 0 | PAD | -100 | 否 | 否 |

shift 后真正比较的是：

| logits 位置 | 当前输入 token | 预测目标 `labels[..., 1:]` | 是否参与 loss |
| ---: | --- | --- | --- |
| 0 | BOS | `你好` | 是 |
| 1 | `你好` | EOS | 是 |
| 2 | EOS | `-100`，对应 EOS 后的 PAD | 否 |
| 3 | PAD | `-100` | 否 |
| 4 | PAD | `-100` | 否 |
| 5 | PAD | `-100` | 否 |
| 6 | PAD | `-100` | 否 |

这张表展示了三条最容易混淆的关系：

```text
BOS -> 第一个真实 token
最后一个真实 token -> EOS
EOS 后的 PAD 对应 label -> -100
```

注意 `labels[0] = BOS` 仍存在，但由于 loss 使用 `labels[..., 1:]`，它不会作为预测目标。第 0 个 logits 对齐的是 `labels[1]`，即第一个真实 token。

## 6. 关键代码块与参数解释

### 6.1 `PretrainDataset.__getitem__`

`PretrainDataset` 位于 [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py)。它的输入是一条 JSON 样本，要求存在 `text` 字段；输出是 `(input_ids, labels)` 两个 `torch.long` 张量。

- `add_special_tokens=False`：不自动加特殊 token，因为上游手工加 BOS/EOS。改成 True 可能导致边界 token 重复，训练语义改变。
- `max_length=self.max_length - 2`：给 BOS/EOS 留位置。改大增加序列长度、显存和计算；改小会更早截断上下文。
- `truncation=True`：超长文本截断。如果关掉，长度可能超过 `max_length`，后续 padding 长度为负或 batch shape 不一致。
- `labels[input_ids == pad_token_id] = -100`：只屏蔽 PAD 位置的 loss，不屏蔽真实 token 和 EOS。

工程建议：8GB 显存下后续实验先把 `max_seq_len` 控制在很短范围，例如 64 或 128，并把 batch size 降低到 1 或 2 做语义 smoke test；这不是上游默认，而是本机资源保守策略。

### 6.2 模型初始化与权重加载

[上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py) 的 `init_model` 负责同时加载 tokenizer 和模型：

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM(lm_config)
```

`tokenizer_path='../model'` 指向上游 `model` 目录中的 tokenizer 文件。`MiniMindForCausalLM(lm_config)` 先构造模型结构。若 `from_weight != 'none'`，再按 `save_dir`、`from_weight`、`hidden_size`、`use_moe` 拼出权重路径，并 `torch.load`、`load_state_dict(strict=False)`。

`from_weight` 会影响训练语义：预训练默认 `none` 是从 0 学；SFT 默认 `pretrain` 是在预训练基座上对齐对话；LoRA 默认 `full_sft` 是在 SFT 模型上加低秩适配；蒸馏默认学生和教师都从 `full_sft` 权重加载。

### 6.3 `MiniMindConfig`

`MiniMindConfig` 位于 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py)。核心参数：

- `hidden_size`：隐藏维度。增大会显著增加参数、激活和显存。
- `num_hidden_layers`：层数。增大会增加计算深度和 KV Cache 层数。
- `vocab_size`：默认 6400。决定 embedding 和 `lm_head` 输出维度。
- `num_attention_heads`、`num_key_value_heads`、`head_dim`：决定注意力头和 GQA 结构。
- `max_position_embeddings`、`rope_theta`、`inference_rope_scaling`：决定 RoPE 位置编码范围和外推配置。
- `use_moe`、`num_experts`、`num_experts_per_tok`：决定是否使用 MoE FFN 以及每个 token 路由专家数。

源码事实是训练入口只显式传 `hidden_size`、`num_hidden_layers`、`use_moe`，其余使用 `MiniMindConfig` 默认值。推理 API 服务中曾传入 `max_seq_len=args.max_seq_len`，但 `MiniMindConfig` 当前读取的是 `max_position_embeddings` 而不是 `max_seq_len`；这个参数是否只是无效额外字段，需要后续进一步阅读 Transformers `PretrainedConfig` 行为或做最小测试验证。

### 6.4 模型 `forward`

这一节讲的是模型前向计算，训练和推理都会走到这里。预训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L36) 调用 `model(input_ids, labels=labels)`；推理生成在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L265) 的 `generate` 内部调用 `self.forward(...)`。二者共享 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L245) 的 `MiniMindForCausalLM.forward`。

`MiniMindModel.forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False)` 位于 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L209)。它的输入核心 shape 是 `[B, T]`。它先在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L214) 用 `embed_tokens(input_ids)` 得到 `[B, T, hidden_size]`，再根据 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L213) 的 `past_key_values` 计算 `start_pos`，从 RoPE buffer 取 `[start_pos:start_pos + T]` 的 cos/sin。每层 block 都执行 RMSNorm、self attention、残差、MLP/MoE、残差。最终在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L232) 返回 norm 后的 hidden states、每层 present K/V 和 MoE aux loss。

`MiniMindForCausalLM.forward` 在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L248) 接收 hidden states 后通过 `lm_head` 得到 logits。训练时 logits shape 通常是 `[B, T, vocab_size]`。`logits_to_keep` 默认 0；`slice(-0, None)` 在 Python 中等价于全量 slice，所以默认保留所有时间步 logits。推理中如果未来传入非 0，可能只保留尾部 logits 以省显存，但当前 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L257) 的 `generate` 没有显式传这个参数。

### 6.5 shifted cross entropy loss

这一节讲的是 `res.loss` 的来源。它发生在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L250)，也就是 `MiniMindForCausalLM.forward` 内部；训练入口 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L36) 只负责把 `labels` 传进去，不需要自己再 shift。

上游 loss 是标准 Causal LM shift：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

先从最小概念开始。模型在某个位置会输出一排 logits，记为 $z_1, z_2, \dots, z_V$，其中 $V$ 是词表大小。logits 还不是概率，可以是任意实数。softmax 会把它们转成概率：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

这里 $p_i$ 表示“模型认为下一个 token 是词表中第 $i$ 个 token 的概率”。所有概率加起来等于 1：

$$
\sum_{i=1}^{V} p_i = 1
$$

如果真实答案 token id 是 $y$，那么模型希望 $p_y$ 越大越好。交叉熵对单个位置的定义是：

$$
L_{\text{CE, one}} = -\log p_y
$$

这条公式的直观含义是：如果模型给正确答案很高概率，比如 $p_y=0.9$，那么 $-\log(0.9)$ 很小；如果模型给正确答案很低概率，比如 $p_y=0.01$，那么 $-\log(0.01)$ 很大。loss 越大，说明模型越不相信正确答案。

如果用 one-hot 标签写得更完整，设 $q_i$ 是真实分布，正确 token 位置 $q_y=1$，其他 token 位置 $q_i=0$，那么交叉熵是：

$$
H(q,p) = -\sum_{i=1}^{V} q_i \log p_i
$$

因为 one-hot 里只有 $q_y=1$，所以上式会退化成：

$$
H(q,p) = -\log p_y
$$

MiniMind 的 shifted CE 是把这个“单个位置的交叉熵”应用到 batch 里所有有效位置。设 $z_{b,t,v}$ 是第 $b$ 条样本、第 $t$ 个输入位置、词表 token $v$ 的 logits；设 $y_{b,t+1}$ 是 shift 后的目标 token id。有效监督位置集合是：

$$
S = \{(b,t)\mid labels[b,t+1] \ne -100,\ 0 \le t < T-1\}
$$

对每个有效位置，先计算正确 token 的概率：

$$
p_{b,t,y_{b,t+1}} =
\frac{e^{z_{b,t,y_{b,t+1}}}}{\sum_{v=1}^{V} e^{z_{b,t,v}}}
$$

再对所有有效位置求平均：

$$
L_{\text{CE}} =
-\frac{1}{|S|}
\sum_{(b,t)\in S}
\log p_{b,t,y_{b,t+1}}
$$

也就是“用位置 $t$ 的 logits 预测位置 $t+1$ 的 label”。如果某个 `labels[b,t+1] = -100`，这个位置不进入集合 $S$，不会贡献 loss，也不会进入平均分母。PyTorch 的 `F.cross_entropy(..., ignore_index=-100)` 默认就是对未忽略位置做 mean。

它和前文 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L37) 的 `loss = res.loss + res.aux_loss` 的关系是：

$$
\text{res.loss} = L_{\text{CE}}
$$

$$
\text{res.aux\_loss} = L_{\text{aux}}
$$

$$
L_{\text{train}} = L_{\text{CE}} + L_{\text{aux}}
$$

$$
L_{\text{backward}} = \frac{L_{\text{CE}} + L_{\text{aux}}}{\text{accumulation\_steps}}
$$

所以 `6.5` 解释的是语言模型预测本身的 loss；训练入口再把 MoE 的辅助 loss 加进来。如果当前是 dense 模型，$L_{\text{aux}} = 0$，这两者看起来就像同一个 loss。如果当前是 MoE 模型，日志中的 `loss` 会大于或等于纯 `logits_loss`，因为它多加了路由辅助项。

改错 shift 会造成 off-by-one：比如拿 `logits[..., 1:, :]` 对 `labels[..., 1:]`，就变成用同位置预测同位置 token，语义错误。

`ignore_index=-100` 只影响 loss，不影响模型 forward。也就是说 PAD token 仍可能进入 embedding 和 attention。当前预训练链路的 [上游 PretrainDataset](../../../references/minimind/dataset/lm_dataset.py#L55) 只返回 `input_ids, labels`，训练调用 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L36) 也只传 `input_ids, labels`，没有传 `attention_mask`；这一点后续应做最小样本对照验证。

### 6.6 DataLoader 参数

这一节讲的是预训练链路中 Dataset 之后、模型 forward 之前的 batch 组织。预训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L135) 创建 `PretrainDataset`，再在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L161) 创建 `SkipBatchSampler`，最后构造：

```text
DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
```

`batch_sampler` 由 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L134) 的 `SkipBatchSampler` 提供，负责 batch 组装和 resume 时跳过已训练 batch。`num_workers` 越大，数据加载并行度越高，但会占用更多 CPU/内存；WSL 和小数据 smoke test 下不一定越大越好。`pin_memory=True` 可能加快 CPU 到 CUDA 拷贝，但也会增加 pinned memory 使用。8GB 显存主要受模型、激活、batch、seq_len 影响，`num_workers` 主要影响主机内存和加载行为。

### 6.7 AdamW、梯度累积和裁剪

这一节讲的是训练链路中 `loss.backward()` 之后、参数真正更新之前和更新时的代码。上游预训练 optimizer 在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L138) 创建：

```python
optim.AdamW(model.parameters(), lr=args.learning_rate)
```

训练中每 step 先在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L31) 用 `get_lr` 动态设置 cosine 学习率。`learning_rate` 过大可能发散，过小训练慢。`accumulation_steps` 会把多个 micro-batch 的梯度累积后再 step，可以在显存有限时模拟更大的有效 batch；但它增加一次参数更新前的计算步数。

`grad_clip=1.0` 对应 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L44) 的：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

它做的是“按全模型梯度范数整体缩放”。假设模型所有参数拼起来是一个大向量 $\theta$，反向传播后所有参数的梯度拼起来是：

$$
g = \nabla_\theta L
$$

梯度范数通常使用 L2 norm：

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$

如果当前梯度范数不超过阈值 $c$，也就是 $\|g\|_2 \le c$，梯度保持不变：

$$
g_{\text{clipped}} = g
$$

如果当前梯度范数超过阈值 $c$，也就是 $\|g\|_2 > c$，就按同一个比例把所有梯度缩小：

$$
g_{\text{clipped}} = g \cdot \frac{c}{\|g\|_2}
$$

在当前上游默认里，$c=1.0$。例如某一步反向传播后全模型梯度范数是 5，那么梯度会整体乘以 $\frac{1}{5}$；如果梯度范数是 0.4，就不缩放。它不改变 loss，也不改变 forward 的 logits；它只改变 optimizer step 前实际拿去更新参数的梯度。

为什么要裁剪：训练偶尔会遇到异常大的梯度。如果不裁剪，AdamW 这一步可能把参数推得太远，导致 loss 突然升高、出现 NaN，或者训练不稳定。裁剪相当于给“单次更新幅度”加一个刹车。

阈值大小的影响：

- 阈值太大：大多数时候不会触发裁剪，等于几乎没保护；遇到梯度爆炸时仍可能一步把训练推坏。
- 阈值适中：只在梯度异常大时触发，平时不干扰训练；这是常见目标。
- 阈值太小：很多正常梯度也被缩小，参数每步更新变小，训练会变慢，甚至学不到足够信号。

所以 `grad_clip=1.0` 不是“loss 最大为 1”，也不是“梯度每个元素最大为 1”。它限制的是所有参数梯度合起来的整体 L2 范数。后续如果做 smoke test，可以记录裁剪前的梯度范数，判断这个阈值是经常触发、偶尔触发，还是几乎不触发。

### 6.8 AMP / bf16 / fp16 / GradScaler

这一节讲的是训练链路，不是推理链路。AMP/autocast 出现在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L119) 的混合精度设置，并在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L35) 包住 forward 和 loss 计算。

当前源码事实：

- CPU：不使用 autocast。
- CUDA + `dtype=bfloat16`：使用 bf16 autocast，不启用 GradScaler。
- CUDA + 非 `bfloat16`：按源码进入 fp16 autocast，并启用 GradScaler。

bf16 和 fp16 都是 16 bit 浮点数，目的都是让矩阵乘法、激活和梯度占用更少显存，并利用 GPU Tensor Core 加速。核心区别是指数位和尾数位分配不同：

```text
fp32: 1 bit 符号 + 8 bit 指数 + 23 bit 尾数
fp16: 1 bit 符号 + 5 bit 指数 + 10 bit 尾数
bf16: 1 bit 符号 + 8 bit 指数 + 7 bit 尾数
```

指数位决定“能表示多大或多小的数”，尾数位决定“这个数附近能表示得多细”。bf16 保留了和 fp32 一样的 8 bit 指数，所以数值范围接近 fp32，不容易因为梯度或激活过大/过小而 overflow/underflow；代价是尾数只有 7 bit，单个数的精细程度比 fp16 低。fp16 有 10 bit 尾数，局部精度比 bf16 细一点，但只有 5 bit 指数，动态范围窄，训练中更容易出现梯度下溢或溢出。

`GradScaler` 的作用主要是服务 fp16 训练：先把 loss 乘上一个 scale，让很小的梯度不要在 fp16 里下溢成 0；反向传播后再在 optimizer step 前 unscale 回来。所以上游在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L137) 写的是 `GradScaler(enabled=(args.dtype == 'float16'))`。bf16 动态范围足够大，通常不需要 loss scaling，所以源码在 bf16 分支不启用 GradScaler。

工程实践上，近几年在支持 bf16 的数据中心 GPU 和新一代消费级 GPU 上，LLM 训练更常优先使用 bf16，因为它的稳定性更接近 fp32，调参和排查 NaN/overflow 的成本更低；fp16 仍然大量存在于较旧硬件、历史训练代码、部分推理和对吞吐/兼容性有特定要求的场景。对本项目来说，源码默认 `dtype=bfloat16` 是一个上游选择；但“当前 RTX 5060 Laptop + 本地 PyTorch/CUDA 组合是否稳定、是否最快”，仍要以后续 smoke test 为准，不能仅凭源码默认值下结论。

### 6.9 checkpoint 保存与 resume 恢复

这一节讲的是训练链路的保存与恢复，主要代码在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L61) 和 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L63)。普通权重保存用于推理和阶段初始化。resume checkpoint 保存用于继续训练。区别可以记为：

```text
普通权重：model.state_dict()
resume：model + optimizer + scaler + epoch + step + world_size + wandb_id + 其他状态
```

普通权重的写入有两处：训练入口 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L68) 会直接 `torch.save({k: v.half().cpu() ...}, ckp)`；`lm_checkpoint` 在 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L75) 也会保存一个模型 `state_dict` 到 `<weight>_<hidden_size>.pth`。这个文件只知道“模型参数现在是多少”，不知道 AdamW 的动量、GradScaler 状态、当前 epoch/step 和已经消费到哪个 batch。

resume checkpoint 在 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L85) 构造为 `resume_data`，里面包含 `model`、`optimizer`、`epoch`、`step`、`world_size`、`wandb_id`，训练入口还通过 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L69) 传入 `scaler=scaler`，因此保存时也会写入 scaler 状态。恢复时，训练入口在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L143) 依次恢复 model、optimizer、scaler、epoch、step。

`from_weight` 控制普通权重加载，代码在 [上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py#L123)：它适合“从某个阶段权重开始训练或推理”。`from_resume` 控制是否查找 resume checkpoint，代码在 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L117)：它适合“上一次训练中断后继续”。二者不是一回事。后续如果要验证 resume，最小闭环应至少检查：训练若干 step 保存、重启后从同一 step 继续、optimizer state 存在、`SkipBatchSampler` 的 step skip 生效。

### 6.10 `generate`、采样和 KV Cache

这一节讲的是推理链路，不是预训练链路。入口是 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L82) 的 `model.generate(...)`，实际生成循环在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L257)。训练时不会调用这个 `generate`；训练只调用 forward、loss、backward 和 optimizer。推理阶段虽然也会调用 `MiniMindForCausalLM.forward`，但它被 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L256) 的 `@torch.inference_mode()` 包住，只读模型参数，不构建反向图，也不会产生梯度更新。

`generate` 的输入已经是 tokenizer 处理后的 prompt token。CLI 推理在 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L73) 区分预训练权重和 SFT 权重：预训练权重用 `bos_token + prompt`，SFT/后训练权重用 chat template。随后 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L78) 得到 `input_ids` 和 `attention_mask`，再传入 `generate`。因此 `generate` 不负责理解原始字符串，也不负责套聊天模板；它只接收形如 `[batch, seq_len]` 的 token id 序列，并在这个序列后面逐个追加新 token。

可以把自回归生成记成下面这个关系：

$$
\begin{aligned}
\text{已有 token} &: x_1, x_2, \ldots, x_t \\
\text{最后位置 logits} &: \mathrm{logits}_t = f(x_1, x_2, \ldots, x_t)_{\text{last}} \\
\text{采样下一个 token} &: x_{t+1} \sim \mathrm{softmax}(\mathrm{logits}_t / T) \\
\text{下一轮输入} &: x_1, x_2, \ldots, x_t, x_{t+1}
\end{aligned}
$$

这里的关键是“最后一个位置预测下一个 token”。模型一次 forward 会对当前输入中的每个位置都给出 logits，但推理生成只需要最后一个有效位置的分布。具体代码是 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L265) 先 forward，得到当前可用 token 的 logits；[上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L267) 取 `outputs.logits[:, -1, :] / temperature`。如果 prompt 是 `BOS, 你, 好`，第一次生成取的是 `好` 这个位置对应的 logits，用它预测第一个新 token；新 token 拼回 `input_ids` 后，下一轮再用这个新 token 所在位置的 logits 预测再下一个 token。


采样参数都发生在 logits 变成 `next_token` 之前。这里的“采样”不是从训练数据里抽样本，而是在推理时从“下一个 token 的概率分布”里选出一个 token。模型 forward 后给出的 `outputs.logits[:, -1, :]` 还不是概率，它只是词表里每个 token 的原始分数。假设词表大小是 6400，那么最后一个位置的 logits 形状就是 `[batch, 6400]`；每一列对应一个候选 token，例如“我”“你”“的”“。”或者某个英文子词。分数越大，表示模型越倾向于下一个位置输出这个 token。

logits 要先经过 softmax 才会变成概率分布：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

这里 $z_i$ 是第 $i$ 个 token 的 logit，$p_i$ 是第 $i$ 个 token 被选中的概率。softmax 做了两件事：第一，把任意实数分数变成正数；第二，让所有候选 token 的概率加起来等于 1。这样模型就不再只是说“哪个 token 分数高”，而是给出一个“下一个 token 候选表”。
更完整地写，采样可以表示为：

$$
\mathbf{p}=\mathrm{softmax}(\mathbf{z}), \qquad x_{t+1} \sim \mathrm{Categorical}(\mathbf{p})
$$

这里 $\mathbf{z}$ 是词表里所有 token 的 logits 向量，$\mathbf{p}$ 是 softmax 后的概率向量，$x_{t+1}$ 是从这个概率分布里抽出来的下一个 token。`Categorical` 可以理解成“按每个 token 的概率大小抽一次”。

换一个更接近真实推理的例子。假设现在不是训练，而是在运行 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L82) 的推理。用户输入：

```text
请用一句话解释 KV Cache 的作用。
```

这里再把“为什么要分成两条路”代入这个具体问题里看一遍。若当前加载的是 SFT/后训练权重，上游会在 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L75) 先把这句用户问题包成 chat template。模型真正看到的不是一句孤零零的“请用一句话解释 KV Cache 的作用。”，而更接近“前面是 user 角色发言，现在轮到 assistant 开始回答”的结构化上下文。因为它训练时就是在这种模板里学习 assistant 段落该怎么续写，所以这一步是在给它一个熟悉的起跑姿势。

若当前加载的是预训练权重，上游则在 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L73) 只在前面加 `bos_token`。这时模型更像是在做“看到一段文本后继续往下写”的普通续写，不额外假设自己已经被放进 `user/assistant` 对话协议里。然后 tokenizer 在 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L78) 把文本变成 `input_ids`。模型第一次 forward 后，假设最后位置正在准备生成回答的第一个 token，softmax 后候选分布示意如下。注意：下面数字是为了讲清采样机制构造的例子，不是本机实测输出，也不是上游固定结果；真实概率会随权重、prompt、tokenizer 和采样参数变化。

| token | 概率 |
| --- | --- |
| `缓存` | 0.40 |
| `减少` | 0.25 |
| `保存` | 0.15 |
| `训练` | 0.12 |
| `天气` | 0.08 |

这张表就是这一轮“采样”的对象。它表达的是：模型认为回答开头接 `缓存` 的可能性最大，接 `减少` 也比较合理，接 `保存` 也有机会；`训练` 可能和上下文有一点关系但不是最贴切；`天气` 基本跑题，所以概率很低。生成不是一次性把“KV Cache 可以减少重复计算……”整句话都选出来，而是每一轮只从当前这张表里选 1 个 token。

假设这一轮真的采到了 `减少`，代码就会在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L280) 把 `减少` 这个 token 拼到 `input_ids` 末尾。此时序列可以粗略理解成：

```text
用户问题 + 已生成回答开头：减少
```

下一轮模型再基于“用户问题 + 减少”重新 forward，得到新的最后位置 logits。新的概率表可能变成：

| token | 概率 |
| --- | --- |
| `重复` | 0.45 |
| `推理` | 0.20 |
| `计算` | 0.18 |
| `显存` | 0.10 |
| `天气` | 0.07 |

如果第二轮又采到 `重复`，回答开头就变成“减少重复”。第三轮再根据“用户问题 + 减少重复”继续算下一张概率表，可能更倾向于接 `计算`。这样一步一步，最后才形成“减少重复计算”这类连续文本。所以采样是推理阶段的逐 token 决策：每一步只决定下一个 token，后面的句子还没固定，要等前面的 token 采出来后再继续算。

```python
 generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
```

如果 `do_sample=False`，代码走 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L278) 的 `argmax`，第一轮会永远选概率最大的 `缓存`。这叫贪心解码，不是真正随机采样。它稳定、可复现、保守，但容易每次都走最显眼的路，输出可能变得单调。

如果 `do_sample=True`，代码走 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L278) 的 `torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)`。这才是这里说的采样：不是必然选概率最大的 token，而是按概率抽一次。上面这个例子里，`缓存` 有 40% 概率被抽中，`减少` 有 25% 概率被抽中，`天气` 也仍有 8% 概率被抽中。可以把它想象成一个不均匀转盘：每个 token 占的扇区大小就是它的概率，指针转到哪里，下一个 token 就是谁。概率大的 token 更容易中，但不是 100% 中。

MiniMind 这段代码里的采样顺序是：

1. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L267) 先取最后位置 logits，并除以 `temperature`。
2. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L268) 如果启用 `repetition_penalty`，先惩罚已经出现过的 token。
3. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L271) 如果 `top_k > 0`，只保留分数最高的 k 个 token。
4. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L273) 如果 `top_p < 1.0`，再按累计概率保留一批候选 token。
5. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L278) 最后才 softmax，并用 `torch.multinomial` 或 `argmax` 得到 `next_token`。

每个参数的作用可以拆开看：

- `temperature`：在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L267) 除 logits。公式是：

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

`T` 就是 temperature。`T` 越大，logits 之间的差距被压小，softmax 后的概率更平均，低分 token 也更容易被抽中，输出更发散、更有随机性。`T` 越小，logits 之间的差距被放大，最高分 token 的概率会更接近 1，输出更保守、更像贪心。当前实现直接做除法，所以不应传 0。直观记法：temperature 像“胆子大小”，温度高更敢选冷门词，温度低更只选稳妥词。

- `repetition_penalty`：在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L268) 惩罚已经出现过的 token。代码会先找出当前 `input_ids` 里已经出现过的 token，再调整这些 token 的 logits。这样做不是禁止重复，而是降低重复概率。例如模型已经连续输出很多次“的”，那“的”这个 token 仍然可能被选中，但分数会被压低，下一步更有机会选别的 token。它解决的是生成中常见的“循环复读”倾向。

- `top_k`：在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L271) 只保留分数最高的 k 个 token，其余置为 `-inf`。置为 `-inf` 的意思是 softmax 后概率变成 0，不会被采样。比如词表有 6400 个 token，`top_k=50` 时，每一步最多只从分数最高的 50 个 token 里抽。它像先做一个硬筛选：太离谱、分数太低的候选直接淘汰。

- `top_p`：在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L273) 按概率从高到低排序，然后累计概率，只保留累计概率不超过阈值附近的一组 token。它和 `top_k` 的区别是：`top_k` 固定保留 k 个，`top_p` 不固定保留几个，而是看概率分布本身。如果模型很确定，前几个 token 概率加起来就超过 `top_p`，候选集合会很小；如果模型不确定，很多 token 概率都差不多，候选集合会变大。它像动态筛选：模型有把握时少给选择，模型没把握时多给选择。

- `do_sample`：在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L278) 决定最后一步是 `torch.multinomial` 随机采样，还是 `argmax` 贪心选择。前面的 `temperature`、`top_k`、`top_p` 都是在改“候选分布”；`do_sample` 决定到底是从这个分布里随机抽，还是直接拿最大值。也就是说，`do_sample=False` 时，很多“随机性”的效果会消失，因为最终不会按概率抽，而是选当前分数最高的 token。

还是用上面“解释 KV Cache”的第一轮概率表来理解。如果 `top_k=3`，候选只剩 `缓存`、`减少`、`保存`，`训练` 和 `天气` 的概率会被清零；然后再在前三个里重新归一化并采样。如果 `top_p=0.85`，按概率从大到小累加：`缓存` 是 0.40，`缓存 + 减少` 是 0.65，`缓存 + 减少 + 保存` 是 0.80，再加上 `训练` 到 0.92，已经超过 0.85，所以候选大致保留到 `训练` 附近；如果 `top_p=0.60`，候选会更窄，大致只保留 `缓存` 和 `减少` 附近。MiniMind 当前代码会先做 `top_k` 再做 `top_p`，所以两者同时打开时，实际候选集合会被连续筛两次。

所以，采样可以用一句话记住：模型先把“下一个 token 可能是谁”变成一个概率表，`temperature`、`repetition_penalty`、`top_k`、`top_p` 负责修改这张概率表，`do_sample` 决定是按概率抽一个，还是直接选概率最大的那个。采样只影响本次生成选择哪个 `next_token`，不会更新模型权重，也不会改变 tokenizer 或训练数据。

KV Cache 是推理加速状态，也是大模型生成里非常重要的概念。先看普通 attention：每一层都会把 hidden states 投影成 Q、K、V。对某个位置来说，Q 表示“当前 token 想查什么信息”，K 表示“历史 token 提供什么索引”，V 表示“历史 token 真正携带什么内容”。attention 的核心关系可以写成：

$$
\mathrm{Attention}(Q, K, V)
= \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + \mathrm{mask}\right)V
$$

训练时通常把一整段序列并行送入模型，所有位置的 Q/K/V 一次算完，再用 causal mask 保证第 `i` 个位置看不到未来 token。推理生成不同：它是一个 token 一个 token 地往后接。假设 prompt 长度是 `n`，已经生成了 `m` 个 token，如果每一轮都把完整的 `n + m` 个 token 重新送进模型，那么前面所有历史 token 的 K/V 会被反复计算。历史 token 没变，模型参数也没变，这些历史 K/V 每轮重算就是浪费。

KV Cache 的做法是：第一次 forward 处理完整 prompt，并把每一层算出的 K/V 保存下来；之后每生成一个新 token，只为这个新 token 计算新的 Q/K/V，再把新的 K/V 接到历史 K/V 后面。MiniMind 的 Attention 在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L120) 做的正是这件事：如果 `past_key_value` 不为空，就把历史 `past_key_value[0]` 和当前 `xk` 沿序列维 `dim=1` 拼接，把历史 `past_key_value[1]` 和当前 `xv` 也沿序列维拼接。随后 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L122) 在 `use_cache=True` 时把拼接后的 `(xk, xv)` 作为本层新的 cache 返回。

这条链路在 `generate` 里对应三步：

1. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L261) 初始化 `past_key_values`，第一次通常是 `None`。
2. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L264) 从第一层 cache 的 K 张量长度读出 `past_len`。
3. [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L265) 只把 `input_ids[:, past_len:]` 送进 forward。第一次 `past_len=0`，所以送入完整 prompt；第二次 cache 已经覆盖 prompt，`input_ids[:, past_len:]` 通常只剩刚追加的 1 个新 token。

用一个具体例子看更直观。假设 prompt 长度是 4，生成 3 个 token，且 `use_cache=True`：

- 第 1 轮：cache 为空，forward 输入 4 个 prompt token，用最后位置 logits 采样第 5 个 token，返回长度为 4 的每层 K/V cache。
- 第 2 轮：`input_ids` 已变成 5 个 token，`past_len=4`，只 forward 第 5 个 token，用它的 logits 采样第 6 个 token，cache 长度变成 5。
- 第 3 轮：`input_ids` 已变成 6 个 token，`past_len=5`，只 forward 第 6 个 token，用它的 logits 采样第 7 个 token，cache 长度变成 6。

注意这个例子里 `input_ids` 最终已经包含 7 个 token，但如果循环正好因为 `max_new_tokens=3` 停止，最后一次刚采样出来的第 7 个 token 还没有进入下一轮 forward，所以返回的 cache 只覆盖到第 6 个 token。正常 decode 文本不受影响，因为输出依据是 `input_ids`；只有调试 `return_kv` 时才需要注意这个时序。

如果 `use_cache=False`，`past_key_values` 每轮都是 `None`，`past_len` 就一直等于 0，每轮都会把从 prompt 到当前生成结果的完整 `input_ids` 重新 forward。这样显存里不需要长期保存所有层的历史 K/V，但计算量会明显增加，长 prompt 或长输出时速度会变慢。KV Cache 的本质不是让 attention 不看历史，而是把“历史 K/V 的计算结果”存下来；新 token 的 Q 仍然会和完整历史 K/V 做 attention，所以模型仍然能利用上下文。

MiniMind 当前实现还要注意 shape。进入 Attention 时，当前 token 的投影形状大致是：

$$
\begin{aligned}
x_q &: [\mathrm{batch},\ \mathrm{current\_seq\_len},\ \mathrm{num\_attention\_heads},\ \mathrm{head\_dim}] \\
x_k &: [\mathrm{batch},\ \mathrm{current\_seq\_len},\ \mathrm{num\_key\_value\_heads},\ \mathrm{head\_dim}] \\
x_v &: [\mathrm{batch},\ \mathrm{current\_seq\_len},\ \mathrm{num\_key\_value\_heads},\ \mathrm{head\_dim}]
\end{aligned}
$$

因为 MiniMind 配置里有 GQA，`num_key_value_heads` 可以小于 `num_attention_heads`。cache 保存的是拼接后的 K/V，形状接近：

$$
\begin{aligned}
\text{每层 K cache} &: [\mathrm{batch},\ \mathrm{total\_seq\_len},\ \mathrm{num\_key\_value\_heads},\ \mathrm{head\_dim}] \\
\text{每层 V cache} &: [\mathrm{batch},\ \mathrm{total\_seq\_len},\ \mathrm{num\_key\_value\_heads},\ \mathrm{head\_dim}] \\
\text{全部层 cache} &: \mathrm{num\_hidden\_layers}\ \text{个}\ (K, V)
\end{aligned}
$$

随后 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L124) 才对 K/V 做 `repeat_kv`，把较少的 KV heads 扩展到 Q heads 对应的数量参与 attention 计算。也就是说，cache 里保存的是较省显存的 KV heads，而不是已经重复到所有 attention heads 的版本。粗略估算 KV Cache 元素量可以记成：

$$
\mathrm{cache\_elements}
\approx 2 \times \mathrm{num\_hidden\_layers} \times \mathrm{batch} \times \mathrm{total\_seq\_len} \times \mathrm{num\_key\_value\_heads} \times \mathrm{head\_dim}
$$

$$
\mathrm{cache\_memory}
\approx \mathrm{cache\_elements} \times \text{每个元素字节数}
$$

这里的 `2` 是 K 和 V 两份；`total_seq_len` 是 prompt token 加已生成 token 的总长度；bf16/fp16 通常每个元素 2 字节，fp32 通常 4 字节。这个公式解释了为什么 KV Cache 会加速推理但占用显存：输出越长、batch 越大、层数越多、KV heads 越多，cache 就越大。对本项目的 RTX 5060 Laptop 约 8GB 显存环境来说，长上下文、较大 batch、较大的 `max_new_tokens` 都可能让 KV Cache 成为显存压力来源；这只是从源码和公式得到的工程边界，具体能跑到多长仍要以后续 smoke test 为准。

KV Cache 还必须和位置编码配合。MiniMind 使用 RoPE，位置不能因为每轮只送 1 个 token 就从 0 重新开始。`MiniMindModel.forward` 在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L210) 根据 cache 长度得到 `start_pos`，再在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L216) 切出 `freqs_cos[start_pos:start_pos + seq_length]` 和 `freqs_sin[start_pos:start_pos + seq_length]`。所以第二轮只送入第 5 个 token 时，它拿到的是第 5 个绝对位置的 RoPE，而不是位置 0 的 RoPE。这一点很关键：KV Cache 省掉的是历史 token 的重复计算，不是把序列位置重置。

`attention_mask` 也会随生成增长。[上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L266) 每轮在 mask 末尾拼一个 1，表示新生成 token 是有效 token。这样如果原 prompt 中存在 padding，attention 计算仍能通过 mask 避免关注无效位置；如果 prompt 没有 padding，mask 通常全是 1，第一次 forward 可能走 PyTorch 的 `scaled_dot_product_attention` 快路径，后续带 cache 的短步生成则按当前源码条件走普通 attention 分支。

EOS 停止在采样之后。`next_token` 被拼回 `input_ids` 的代码是 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L280)；cache 更新在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L281)；如果 `next_token == eos_token_id`，就在 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L284) 标记完成，全部完成后 [上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L285) break。如果某条样本已经完成，后续会被强制填 EOS，避免继续生成普通 token。最后 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L88) 只 decode prompt 之后的新 token。所有这些操作都不改变模型参数，只改变本次生成过程中的 token 序列、attention mask 和 cache 状态。

这一节最容易混淆的点可以集中记成四句话：

- `generate` 是推理循环，不是训练循环；它不做 backward，也不更新权重。
- `outputs.logits[:, -1, :]` 表示用当前最后一个位置预测下一个 token，不是把所有位置都 decode 成输出。
- KV Cache 保存的是每一层历史 K/V，不保存 Q；下一轮新 token 的 Q 仍会去看完整历史 K/V。
- KV Cache 用显存换速度；它减少重复计算，但上下文越长，cache 占用越大。

## 7. 预训练与推理的参数更新边界

本节把训练链路和推理链路中“改 token”“改参数”“只读状态”的位置分开。这样看代码时不容易把 `generate` 误认为训练的一部分，也不容易把 `backward` 误认为已经更新了参数。

会改变 token 序列的位置：

- 预训练数据阶段：[上游 PretrainDataset.__getitem__](../../../references/minimind/dataset/lm_dataset.py#L47) 把文本 token 前后加 BOS/EOS，尾部加 PAD。
- SFT 数据阶段：[上游 SFTDataset.create_chat_prompt](../../../references/minimind/dataset/lm_dataset.py#L71) 把消息用 chat template 展开成带角色和特殊标签的文本。
- 推理 prompt 阶段：[上游 eval_llm.py](../../../references/minimind/eval_llm.py#L78) 用 tokenizer 把字符串变成 token id。
- 推理生成阶段：[上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L280) 每步把采样得到的 `next_token` 追加到 `input_ids`。
- 推理输出阶段：[上游 eval_llm.py](../../../references/minimind/eval_llm.py#L88) 用 `tokenizer.decode` 把生成 token ids 变回文本。

会改变模型参数的位置：

- 训练反向阶段：[上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L40) 的 `loss.backward()` 只产生或累积梯度，不直接改参数。
- 训练更新阶段：[上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py#L46) 的 `scaler.step(optimizer)` / `optimizer.step()` 真正更新参数。
- LoRA 训练阶段：[上游 train_lora.py](../../../references/minimind/trainer/train_lora.py#L140) 只让包含 `lora` 的参数 `requires_grad=True`，并在 [上游 train_lora.py](../../../references/minimind/trainer/train_lora.py#L152) 只把 `lora_params` 交给 AdamW。

只读取状态、不更新参数的位置：

- 推理加载阶段：[上游 eval_llm.py](../../../references/minimind/eval_llm.py#L12) 的模型加载、[上游 eval_llm.py](../../../references/minimind/eval_llm.py#L30) 的 `model.half().eval().to(...)` 只设置推理状态，不更新参数。
- 推理生成阶段：[上游 model_minimind.py](../../../references/minimind/model/model_minimind.py#L256) 的 `@torch.inference_mode()`、forward、采样、KV Cache 更新都只读模型权重。
- 推理解码阶段：[上游 eval_llm.py](../../../references/minimind/eval_llm.py#L88) 的 `tokenizer.decode` 只是 token id 到字符串的转换。
- 蒸馏训练中，teacher model 在 [上游 train_distillation.py](../../../references/minimind/trainer/train_distillation.py#L207) 加载后，在 [上游 train_distillation.py](../../../references/minimind/trainer/train_distillation.py#L208) `eval()`，并在 [上游 train_distillation.py](../../../references/minimind/trainer/train_distillation.py#L209) `requires_grad_(False)`，只提供分布，不更新 teacher 参数。

训练和推理都使用 `MiniMindForCausalLM.forward` 产出 logits。训练多了 labels、loss、backward、optimizer、checkpoint；推理多了 prompt 模板、采样、cache、EOS 和 decode。

## 8. 进阶分支地图

SFT 分支在 [上游 trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)，主循环几乎复用预训练，但 Dataset 换成 `SFTDataset`，默认 `from_weight='pretrain'`，默认 `learning_rate=1e-5`。它的关键差异是 labels 只让 assistant 回复区间参与 loss，不让 user/system/tool prompt 全量参与监督。

LoRA 分支在 [上游 model/model_lora.py](../../../references/minimind/model/model_lora.py) 和 [上游 trainer/train_lora.py](../../../references/minimind/trainer/train_lora.py)。`apply_lora` 给输入输出维度相同的 Linear 层增加低秩分支，并 monkey patch forward；训练时冻结非 LoRA 参数，只把 `lora` 参数交给 AdamW。它适合后续理解参数高效微调，但当前不应先于 Causal LM 主线。

蒸馏分支在 [上游 trainer/train_distillation.py](../../../references/minimind/trainer/train_distillation.py)。它加载学生和教师模型，用监督 CE 加教师分布 KL 训练学生。这里会引入 `teacher_logits`、`temperature`、`alpha` 和 mask 后 KL，建议在看懂 SFT labels 后再读。

DPO 分支在 [上游 trainer/train_dpo.py](../../../references/minimind/trainer/train_dpo.py)，数据来自 `DPODataset` 的 chosen/rejected 对，并同时使用 policy model 和 reference model。PPO/GRPO/RLAIF 分支在 [上游 trainer/train_ppo.py](../../../references/minimind/trainer/train_ppo.py)、[上游 trainer/train_grpo.py](../../../references/minimind/trainer/train_grpo.py)、[上游 trainer/rollout_engine.py](../../../references/minimind/trainer/rollout_engine.py)，会引入在线采样、reward、KL、优势估计等概念。Agent RL 分支在 [上游 trainer/train_agent.py](../../../references/minimind/trainer/train_agent.py)，还会引入多轮工具调用和环境反馈。当前阶段只需知道它们是后训练分支，不要混入预训练主线。

Tool Use 推理主要落在 tokenizer chat template、[上游 scripts/serve_openai_api.py](../../../references/minimind/scripts/serve_openai_api.py)、[上游 scripts/chat_api.py](../../../references/minimind/scripts/chat_api.py) 和上游工具评测脚本 [上游 scripts/eval_toolcall.py](../../../references/minimind/scripts/eval_toolcall.py)。它复用 `generate`，但会额外解析 `<tool_call>` 和 `<tool_response>`。

## 9. 面向 Encoder-only 基础的记忆点

可以把 Decoder-only Causal LM 记成三个对齐：

```text
输入对齐：input_ids[t] 作为当前位置输入
可见性对齐：位置 t 只能看 <= t 的 token
监督对齐：logits[t] 预测 labels[t + 1]
```

如果这三个对齐都成立，模型就能训练 next-token prediction。训练时一次性喂完整序列，是为了并行；推理时逐 token 生成，是因为未来 token 必须由模型自己先生成出来。

把 `-100` 和 PAD 分清也很重要：

```text
PAD token id = 0，是词表中的一个输入 id
-100 不是 token id，只是 loss 忽略标记
input_ids 可以含 PAD
labels 可以含 -100
tokenizer 正常不会生成 -100
```

把 EOS 的双重身份分清：

```text
训练中：EOS 是模型应该学会预测的结束 token
推理中：EOS 是 generate 停止条件
```

## 10. 后续阅读顺序与最小验证

第一步精读 [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py) 的 `PretrainDataset.__getitem__`。读完要能回答：一条 `{"text": ...}` 如何变成 `input_ids` 和 `labels`；BOS/EOS/PAD 在哪里加入；`-100` 在哪里产生；预训练链路有没有 attention mask。

第二步精读 [上游 trainer/train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py) 的主函数和 `train_epoch`。读完要能回答：tokenizer 从哪里来；DataLoader 如何组 batch；什么时候 forward；什么时候 backward；什么时候 step；什么时候保存普通权重和 resume；`from_weight` 与 `from_resume` 差异是什么。

第三步精读 [上游 trainer/trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py) 的 `init_model` 和 `lm_checkpoint`。读完要能回答：随机初始化、普通权重加载、resume 加载分别发生在哪里；`strict=False` 有什么风险；resume 文件里到底保存了什么；`SkipBatchSampler` 如何跳过已训练 batch。

第四步精读 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py) 的 `MiniMindConfig`、`MiniMindModel.forward`、`Attention.forward`、`MiniMindForCausalLM.forward`。读完要能回答：`input_ids` 到 `hidden_states` 到 `logits` 的 shape；causal mask 在哪里；RoPE 在哪里；KV Cache 的 K/V 如何拼接；loss 的 shift 为什么是 `[:-1]` 对 `[1:]`。

第五步精读 [上游 model/model_minimind.py](../../../references/minimind/model/model_minimind.py) 的 `generate` 和 [上游 eval_llm.py](../../../references/minimind/eval_llm.py)。读完要能回答：prompt 如何变成 token；chat template 何时使用；普通权重如何加载；每步如何采样；cache 如何增长；EOS 如何停止；decode 为什么只取 prompt 之后的新 token。

第六步再读 [上游 trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py) 和 [上游 dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py) 的 `SFTDataset`。读完要能回答：SFT 为什么不是所有 token 都参与 loss；assistant 区间如何识别；`apply_chat_template` 如何改变输入序列。

暂时跳过：LoRA、蒸馏、DPO、PPO、GRPO、Agent RL、OpenAI API 服务和模型转换。它们不是不重要，而是依赖你先把 Causal LM 的 token、label、loss、checkpoint、generate 主线看懂。

后续最小实验建议按三层区分：

1. “源码已经看懂”：能用文件、类、函数、参数和 shape 解释调用链，但没有运行。
2. “已手写对照”：在个人仓库写了最小模块或最小样本，对照上游输出 token/label/shape。
3. “已实际运行验证”：有本轮命令、输出、日志或测试证明，例如单 batch forward/loss/backward/step、checkpoint 保存恢复、最小 prompt 推理。

下一步最小验证不应直接跑正式训练，而应先做：

- 用一条极小 JSONL 样本打印 `input_ids`、`labels`、shift 目标和 loss mask。
- 用极小 `MiniMindConfig(hidden_size=..., num_hidden_layers=...)` 做单 batch forward，确认 logits shape 和 loss 有限。
- 再做一次 backward 和 AdamW step，确认至少一个参数发生变化。
- 最后再验证保存普通权重和 resume checkpoint 的文件结构。

## 11. 本轮自检清单

- 上游路径均标注为“上游”或“上游引用”，没有把上游实现写成个人仓库已完成实现。
- 本文没有声明已完成训练、推理、SFT、LoRA、蒸馏、DPO、PPO、GRPO、Agent RL 或性能验证。
- BOS、EOS、PAD、`input_ids`、`labels`、PAD token id 与 `-100` 已明确区分。
- 已分别说明预训练调用链和推理 / 生成调用链。
- 重要结论已尽量落到当前本地上游引用中的具体文件、类、函数和参数。
- Markdown 链接均指向具体文件；目录、通配符和不存在的权重文件使用代码格式。
- 待后续验证项已明确标出，没有把推测写成事实。
