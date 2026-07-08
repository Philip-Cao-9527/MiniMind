# 从零讲明白 MiniMind 这次 SFT NaN、zero-supervision 与 resume 语义问题

## 事实边界说明

### 本机已验证事实

- 本轮已实际读取 [AGENTS.md](../../AGENTS.md)、[README.md](../../README.md)、[code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md)、[fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md)、[dataset/lm_dataset.py](../../dataset/lm_dataset.py#L58)、[trainer/train_full_sft.py](../../trainer/train_full_sft.py#L24)、[model/model_minimind.py](../../model/model_minimind.py#L245)、[tests/diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L1)、[trainer/trainer_utils.py](../../trainer/trainer_utils.py#L63)、[eval_llm.py](../../eval_llm.py#L41)、归档 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md)、归档 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log) 和归档 [full-sft-current-run.env](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-current-run.env)。
- 本轮确认：当前本地 `SFTDataset` 已经是修复后的“先生成完整 labels，再裁到尾部窗口”的实现，见 [lm_dataset.py](../../dataset/lm_dataset.py#L106) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L128)。
- 本轮确认：当前本地 `train_full_sft.py` 已经加入 `valid_label_tokens == 0` 直接跳过、非有限 loss 立即失败、checkpoint 延后到有效 optimizer update 边界的逻辑，见 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。
- 本轮确认：当前本地模型 loss 仍然是 `F.cross_entropy(..., ignore_index=-100)`，没有在模型层额外兜底，见 [model_minimind.py](../../model/model_minimind.py#L249) 到 [model_minimind.py](../../model/model_minimind.py#L253)。
- 本轮确认：当前本地推理入口 `eval_llm.py` 在 `weight=full_sft` 时直接从 `out/full_sft_*.pth` 这类普通权重加载推理，不会读取 `resume checkpoint`，见 [eval_llm.py](../../eval_llm.py#L49) 到 [eval_llm.py](../../eval_llm.py#L67)。
- 本轮确认：归档 README 明确记录了这次失败 / 中断 run 的真实启动参数，其中 `batch_size=1`、`max_seq_len=384`、`accumulation_steps=6`、`save_interval=5000`、`from_weight=pretrain`、`from_resume=0`，见 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L9) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L23)。
- 本轮确认：归档日志确实在 `980 / 1060 / 1880 / 7580 / 8220 / 8840` 等 step 记录了 `loss: nan, logits_loss: nan`，并且在 `step=10000` 日志之后出现 `KeyboardInterrupt by user`，堆栈停在 `torch.save(resume_data, resume_tmp)`，见 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L67)、[full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L71)、[full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L112)、[full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L397)、[full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L429)、[full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L460) 和 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L518) 到 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L534)。

### 当前本地代码事实

- `SFTDataset` 的关键语义是：聊天样本先走 chat template，再编码成 `input_ids`，然后只把 assistant 回复区间写进 `labels`，其他位置写成 `-100`，见 [lm_dataset.py](../../dataset/lm_dataset.py#L71) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L128)。
- `MiniMindForCausalLM.forward` 的关键语义是：它仍然做标准因果语言模型（causal language model）训练，即当前位置预测下一个 token；SFT 并没有改掉这个数学目标，而是改了哪些位置参与监督，见 [model_minimind.py](../../model/model_minimind.py#L245) 到 [model_minimind.py](../../model/model_minimind.py#L253)。
- `train_full_sft.py` 的关键语义是：训练循环统计整个 micro-step 的有效监督 token 数 `valid_label_tokens`，如果为 0 就直接 `continue`；如果 loss、aux loss 或 grad norm 非有限，就清梯度并抛错，见 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L152)。
- `lm_checkpoint` 保存的是模型参数、优化器状态、epoch、step、world_size，以及像 `scaler` 这样的额外 `state_dict`；它不保存参数当前 `.grad`，见 [trainer_utils.py](../../trainer/trainer_utils.py#L63) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L116)。
- 当前训练脚本把 `--max_seq_len` 暴露为 SFT 的训练窗口长度，默认值是 `768`，并把它传给 `SFTDataset(max_length=args.max_seq_len)`；这说明它直接决定单条训练样本最多保留多少 token，见 [train_full_sft.py](../../trainer/train_full_sft.py#L227) 和 [train_full_sft.py](../../trainer/train_full_sft.py#L261) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L264)。
- 当前模型结构里另外还有 `max_position_embeddings=32768`，见 [model_minimind.py](../../model/model_minimind.py#L27)。这和脚本默认配置里的 `max_seq_len=768` 不是一回事：前者更接近“模型结构允许的位置编码上限”，后者更接近“训练脚本默认准备拿多长的窗口去训练”；而这次实际中断 run 的真实窗口长度，还要以归档 README 记录的 `max_seq_len=384` 为准。

### 文档 / 日志证据

- 这次事件的审查结论在 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L15) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L123)。
- 审查报告给出的关键证据包括：前 `10000` 个 step 中旧行为产生 `136` 个全 `-100 labels` 样本、你列出的 NaN step 全命中这些 step、CPU toy `CrossEntropy` 在全部 `labels=-100` 时返回 `nan`，见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L27) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L61)。
- 修复报告明确写明：这次 partial full SFT 不允许用于 resume、推理或验收；后续新的 full SFT 必须从 `--from_weight pretrain --from_resume 0` 启动，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L7) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L87)。
- 归档 README 直接给出了这次中断 run 的真实启动参数和禁止 resume / 推理的边界，见 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L9) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L79)。
- 归档原始日志直接支持“多个 NaN step 真实出现过”和“中断发生在 `torch.save(resume_data, resume_tmp)`”这两条结论，见 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L67) 到 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L71) 以及 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L518) 到 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L534)。
- 审查报告和修复报告当前都引用新的归档目录 `../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/`。如果后续再移动归档位置，相关文档中的相对路径也需要一起更新；否则涉及原始日志、README 和 checkpoint 的 Markdown 引用会直接失效。

### 工程判断

- “全 `-100 labels` 是这次 NaN 的主要机制”有较强支持证据，但不能单凭当前材料断言“所有 NaN 都只由这一条机制导致”。
- “旧版 checkpoint 保存点与有效梯度累积边界不一致，会导致 resume 不等价”是成立的工程结论，因为当前保存格式不包含 `.grad`，而恢复逻辑又会按 `step` 跳过已经走过的 micro-step。
- “这次问题影响预训练底座权重”目前没有直接证据支持，因此不能下这个结论。

### 后续需要验证

- 新的 full SFT 从 `pretrain` 干净启动后，长时间 GPU 路径是否稳定。
- 新的 full SFT 完成后，`out/full_sft_768.pth` 的推理质量、KV Cache、EOS 停止和 history 行为是否正常。
- 旧 partial run 是否还有其他独立于 zero-supervision 的 NaN 诱因，目前不能完全排除。


## 详细原理讲解（>=3000字，含公式）

### 1. 先从最基础开始：SFT 到底是什么

SFT 是监督微调（Supervised Fine-Tuning）。“监督”两个字的核心含义是：训练时不只是给模型看输入，还告诉它“正确答案是什么”；“微调”表示不是从随机初始化重新发明一个模型，而是在已有参数基础上继续训练。

如果把一个语言模型想成“会根据上文往后续写的人”，那么预训练像是让它读了很多文章、小说、网页、对话，慢慢学会语言规律；SFT 像是把它招进岗位培训，告诉它：“以后遇到这种 user 提问格式，你要以 assistant 身份这样回答。”两者都还在做“下一个 token 预测”，但监督的重点不一样。

可以把这两个阶段写成两条关系式：

预训练阶段：

$$
\text{文本} \xrightarrow{\text{tokenizer}} \text{input\_ids}
\xrightarrow{\text{labels 近似等于输入右移}}
\mathcal{L}_{\text{pretrain}}
$$

SFT 阶段：

$$
\text{conversations} \xrightarrow{\text{chat template + tokenizer}} \text{input\_ids}
\xrightarrow{\text{仅 assistant 回复正文与结束标记参与 labels}}
\mathcal{L}_{\text{SFT}}
$$

它们的骨架很像，但“哪里算监督”不同。

### 2. SFT 和 pretrain 在目标、数据、labels、loss 上到底差在哪

先给一句最短结论：**SFT 和 pretrain 的数学任务都还是因果语言建模，但 SFT 只在 assistant 回复区间上打分，pretrain 通常在绝大多数正文 token 上打分。**

#### 2.1 数据结构不同

预训练样本常常是这样的：

```json
{"text": "今天天气很好，我们去公园散步。"}
```

SFT 样本常常是这样的：

```json
{
  "conversations": [
    {"role": "user", "content": "1加1等于几？"},
    {"role": "assistant", "content": "1加1等于2。"}
  ]
}
```

在 MiniMind 里，预训练数据由 `PretrainDataset` 处理，见 [lm_dataset.py](../../dataset/lm_dataset.py#L37) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L55)。SFT 数据由 `SFTDataset` 处理，见 [lm_dataset.py](../../dataset/lm_dataset.py#L58) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L134)。

#### 2.2 tokenizer 前面多了一层 chat template

预训练时，`text` 直接编码成 token。SFT 时，先要把消息列表排版成模型认识的对话模板，然后再编码。也就是：

$$
\text{messages} \xrightarrow{\text{apply\_chat\_template}} \text{prompt string}
\xrightarrow{\text{tokenizer}} \text{input\_ids}
$$

当前实现见 [lm_dataset.py](../../dataset/lm_dataset.py#L71) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L86)。

这一步为什么重要？因为模型并不天然理解“这一句是 user，那一句是 assistant”，它只认识 token。chat template 实际上是在告诉模型角色边界和消息边界。

#### 2.3 labels 语义不同

预训练里，labels 通常是：

$$
\text{labels} \approx \text{input\_ids}
$$

只是在 padding 上改成：

$$
\text{labels}[t] = -100
$$

SFT 里则是：

$$
\text{labels}[t] =
\begin{cases}
\text{input\_ids}[t], & t \in \text{assistant supervised span} \\
-100, & \text{otherwise}
\end{cases}
$$

但这里的 `assistant supervised span` 如果只写成一句术语，确实很容易让人不知道“到底是哪几个 token”。MiniMind 当前本地实现里，这个 span 不是“整段 assistant 相关 token 都参与”，而是更细一点：

1. 先把整条样本的 `labels` 全部初始化成 `-100`，见 [lm_dataset.py](../../dataset/lm_dataset.py#L88) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L90)。
2. 再去找一段“assistant 回复开始标记”，也就是 `bos_id = tokenizer(f'{tokenizer.bos_token}assistant\\n', add_special_tokens=False).input_ids`，见 [lm_dataset.py](../../dataset/lm_dataset.py#L65)。
3. 找到这个开始标记以后，`start = i + len(self.bos_id)`，也就是**从 assistant 角色头后面一个位置开始**才算真正要监督的回复内容，见 [lm_dataset.py](../../dataset/lm_dataset.py#L92) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L94)。
4. 然后继续往后找结束标记 `eos_id = tokenizer(f'{tokenizer.eos_token}\\n', add_special_tokens=False).input_ids`，见 [lm_dataset.py](../../dataset/lm_dataset.py#L66) 和 [lm_dataset.py](../../dataset/lm_dataset.py#L95) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L98)。
5. 最后把从 `start` 到 `end + len(eos_id)` 这一段写成真实 token id，见 [lm_dataset.py](../../dataset/lm_dataset.py#L99) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L100)。

这句话翻成人话就是：

- `system` 消息不参与 labels。
- `user` 消息不参与 labels。
- `assistant` 的角色头，也就是类似“现在轮到 assistant 说话了”的那段模板 token，不参与 labels。
- **真正参与 labels 的，是 assistant 角色头后面的回复正文，以及这段回复结尾的结束标记 token 串。**
- padding 仍然不参与 labels。

也就是说，SFT 不是把 user 文本从输入里删掉，而是把 user 位置从监督里删掉；更细一点说，SFT 连 assistant 的“角色头提示词”通常也不直接监督，真正监督的是 assistant 开始回答之后的内容。

可以看一个最小化的伪示例。假设 chat template 展开后，模型看到的是下面这串内容：

```text
[system 提示]
[user 提问：1加1等于几？]
[assistant 角色头]
[assistant 回复：1加1等于2。]
[assistant 结束标记]
```

那么 `input_ids` 的语义大概是：

```text
input_ids =
[system token..., user token..., assistant头 token..., 回复正文 token..., 结束标记 token...]
```

而 `labels` 更接近：

```text
labels =
[-100, -100, ..., -100, -100, -100, 回复正文 token..., 结束标记 token...]
```

请注意，这里前面那一长串 `-100` 不是说模型“没看到”这些 token，而是说模型虽然看到了它们，但老师不拿这些位置直接计分。

如果再具体一点，可以写成一个更直观的小表。下面这只是示意，不代表真实 tokenizer 一定正好切成这些词：

| 位置语义 | `input_ids` 是否包含 | `labels` 是什么 | 是否参与 loss |
| --- | --- | --- | --- |
| `system` 内容 | 包含 | `-100` | 否 |
| `user` 提问 | 包含 | `-100` | 否 |
| `assistant` 角色头 | 包含 | `-100` | 否 |
| assistant 回复正文第 1 个 token | 包含 | 真实 token id | 是 |
| assistant 回复正文后续 token | 包含 | 真实 token id | 是 |
| assistant 结束标记 | 包含 | 真实 token id | 是 |
| padding | 包含 | `-100` | 否 |

这里还有两个初学者特别容易误会的点。

第一，不是“只有最后一轮 assistant 才参与监督”。当前 `generate_labels` 会从左到右扫完整条 `input_ids`，只要遇到一段 assistant 开始标记，再找到对应结束标记，就会把这一段回复正文写进 labels。所以如果一条样本里有多轮 assistant 回复，那么**每一轮 assistant 回复正文都可能参与监督**，见 [lm_dataset.py](../../dataset/lm_dataset.py#L91) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L103)。

第二，不是“assistant 一整段都参与监督”。角色头本身并不直接被监督，因为代码里监督起点是：

$$
\text{start} = i + |\text{bos\_id}|
$$

也就是从 `bos_id` 之后开始。你可以把 `bos_id` 理解成“assistant 这段话的牌子”，而真正要评分的是牌子后面写出来的回答内容。

如果把这段逻辑写成更贴近代码的关系式，可以写成：

$$
\text{assistant supervised span} = [\text{after bos\_id},\ \text{through eos\_id}]
$$

再翻译成人话就是：

> 被监督的不是“看到 assistant 这几个字”这件事，而是“assistant 从这里开始到底说了什么，以及它在哪里结束”。

这样再回头看前面那条公式：

$$
\text{labels}[t] =
\begin{cases}
\text{input\_ids}[t], & t \in \text{assistant supervised span} \\
-100, & \text{otherwise}
\end{cases}
$$

你就可以把它理解成：

- 只要当前位置属于 assistant 回复正文或结束标记，就把真实 token id 抄进 labels。
- 只要当前位置属于 system、user、assistant 角色头、padding，labels 就保持 `-100`。

这也是为什么我前面一直强调：SFT 和 pretrain 的真正差别，不是模型前向公式突然换了一套，而是“老师到底拿哪一段 token 来计分”。

#### 2.4 loss 公式表面相同，监督集合不同

MiniMind 当前模型端 loss 直接定义在 [model_minimind.py](../../model/model_minimind.py#L245) 到 [model_minimind.py](../../model/model_minimind.py#L253)。最关键的三行是：

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

也就是说，代码先做了一个“错位对齐（shift）”，再把所有位置摊平成二维 logits 和一维 labels，最后调用 `F.cross_entropy(..., ignore_index=-100)`。

把它写成公式，就是：

$$
x = \text{logits}[..., :-1, :]
$$

$$
y = \text{labels}[..., 1:]
$$

$$
\mathcal{L} = \text{CrossEntropy}(x, y, \text{ignore\_index}=-100)
$$

这里最好把“为什么要 `[:-1]` 和 `[1:]`”也说透。因为这是标准因果语言模型的 next-token prediction 语义：

- 位置 $t$ 的输出 `logits[..., t, :]`，拿来预测位置 $t+1$ 的真实 token。
- 所以 logits 要去掉最后一个位置，labels 要去掉第一个位置。

如果原始 shape 是：

$$
\text{logits} \in \mathbb{R}^{B \times T \times V}, \quad \text{labels} \in \mathbb{Z}^{B \times T}
$$

那么 shift 之后：

$$
x \in \mathbb{R}^{B \times (T-1) \times V}, \quad y \in \mathbb{Z}^{B \times (T-1)}
$$

再 `view` 以后就是：

$$
x_{\text{flat}} \in \mathbb{R}^{N \times V}, \quad y_{\text{flat}} \in \mathbb{Z}^{N}, \quad N = B(T-1)
$$

这里：

- $B$：batch size。
- $T$：序列长度。
- $V$：词表大小。
- $N$：摊平后的总监督位置数。

更展开一点，可以写成：

$$
\mathcal{L} =
- \frac{1}{|S|}
\sum_{(b,t)\in S}
\log
\frac{\exp(z_{b,t,y_{b,t}})}
{\sum_{v=1}^{V}\exp(z_{b,t,v})}
$$

符号解释：

- $b$：batch 中第几条样本。
- $t$：序列中的位置。
- $V$：词表大小。
- $z_{b,t,v}$：第 $b$ 条样本第 $t$ 个位置对词表第 $v$ 个 token 的 logit。
- $y_{b,t}$：对应位置的真实标签。
- $S$：所有有效监督位置组成的集合，也就是 `y != -100` 的位置。
- $|S|$：有效监督位置的总数。

预训练和 SFT 的关键差别，不是这个公式长得不同，而是集合 $S$ 的构造不同。预训练里 $S$ 很大，通常除了 padding 以外多数位置都在里面；SFT 里 $S$ 只包含 assistant 监督区间。

这里要特别讲清楚 `ignore_index=-100` 到底是什么意思，因为你后面问的 NaN、梯度、backward 都从这里开始。

可以定义一个掩码：

$$
m_{b,t} =
\begin{cases}
1, & y_{b,t} \neq -100 \\
0, & y_{b,t} = -100
\end{cases}
$$

再定义有效监督位置总数：

$$
M = \sum_{b,t} m_{b,t} = |S|
$$

那么当前这条 loss 的更严格写法其实是：

$$
\mathcal{L}
=
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{\sum_{b,t} m_{b,t}}
=
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{M}
$$

其中单位置交叉熵是：

$$
\ell_{b,t}
=
-\log p_{b,t}(y_{b,t})
=
-\log \frac{\exp(z_{b,t,y_{b,t}})}{\sum_{v=1}^{V}\exp(z_{b,t,v})}
$$

这条式子说明三件很关键的事。

第一，**对于单个 `label=-100` 的位置，它不是“loss 返回 0”这么简单，而是这个位置根本不进入平均。** 更准确地说，是这个位置的掩码 $m_{b,t}=0$，所以它对分子贡献 0，对分母也不计数。

第二，只要 batch 里还有至少一个有效标签，也就是：

$$
M > 0
$$

那么所有 `label=-100` 的位置都只是被忽略，它们本身不会把 loss 变成 NaN。

第三，真正危险的是：

$$
M = 0
$$

也就是整个 batch 所有位置都被 ignore。这时公式变成：

$$
\mathcal{L} = \frac{0}{0}
$$

这在数学上是未定义的，不是正常的 0。当前 MiniMind 的 toy 验证已经直接证明，在 `F.cross_entropy(..., ignore_index=-100, reduction="mean")` 这条实现路径下，PyTorch 返回的是 `nan`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。

所以你现在可以把 `-100` 的效果严格地区分成两种情况：

1. **局部 ignore**：只有部分位置是 `-100`。
   这时这些位置被忽略，loss 由剩下的有效位置正常平均，结果通常是有限值。
2. **全局 all ignore**：整个 batch 所有位置都是 `-100`。
   这时有效监督计数 $M=0$，mean loss 变成未定义，当前实现返回 `nan`。

#### 2.5 `max_seq_len` 到底是什么，为什么它在 SFT 里特别重要

如果只用一句话解释，`max_seq_len` 就是：**单条样本在这次训练里最多允许保留多少个 token。**

可以把一条原始聊天样本想成一卷很长的纸带，上面按顺序写着：

```text
system -> user1 -> assistant1 -> user2 -> assistant2 -> ...
```

tokenizer 会把它变成很长的一串 token id。如果这串 token 的总长度记作 $L_{\text{raw}}$，训练允许的窗口长度记作 $L_{\max}$，那么：

$$
L_{\text{kept}} = \min(L_{\text{raw}}, L_{\max})
$$

其中：

- $L_{\text{raw}}$：原始完整样本的 token 长度。
- $L_{\max}$：也就是 `max_seq_len`。
- $L_{\text{kept}}$：最后真正送进训练的长度。

如果 $L_{\text{raw}} \le L_{\max}$，说明这条样本装得下，不需要截断；如果 $L_{\text{raw}} > L_{\max}$，说明它装不下，必须裁窗口。

为什么它在 SFT 里特别重要？因为 SFT 的监督通常集中在最后一轮 assistant 回复，而不是平均散在整条样本上。也就是说，在 pretrain 里，截掉一部分往往只是损失了一部分正文 token；在 SFT 里，截掉的位置如果刚好是最后一轮 assistant 回复，就可能把主要监督整段切掉。所以 `max_seq_len` 在 SFT 里不是“一个普通长度参数”，而是“决定你到底保住了多少训练语义”的关键预算。

对小白最直观的理解是：`max_seq_len` 像一个行李箱容量。

- 行李箱太小：最重要的东西可能塞不进去，被留在外面。
- 行李箱太大：虽然更能装，但会更重、更占地方，搬运也更吃力。

SFT 里最怕的是“行李箱刚好把最重要那件 assistant 回答留在外面”。

它还直接影响显存和速度。若 batch size 记作 $B$，序列长度记作 $L$，隐藏维度记作 $d$，那么很多中间激活大致都和：

$$
O(B \cdot L \cdot d)
$$

同量级增长；而自注意力里常见的打分矩阵还会更依赖序列长度，通常可以粗略记成：

$$
O(L^2)
$$

这不是让你背复杂度考试，而是想让你抓住一个工程直觉：**长度翻倍，开销往往不是“多一点”，而是会明显变重。**

因此 `max_seq_len` 带来的实际影响至少有四个：

1. 它决定单条样本能保留多少上下文。
2. 它决定最后一轮 assistant 监督是否还留在窗口里。
3. 它决定每个 batch 的显存压力和训练速度。
4. 它会影响你能把 `batch_size` 设多大，因为总显存预算是固定的。

在当前 MiniMind 本地代码里，这个值由 [train_full_sft.py](../../trainer/train_full_sft.py#L227) 定义，并在 [train_full_sft.py](../../trainer/train_full_sft.py#L263) 传进 `SFTDataset`。所以后续所有“会不会截断”“裁前面还是裁后面”“会不会产生 zero-supervision”都绕不开它。

结合这次真实中断 run，还能把这件事说得更落地一点：归档 README 记录的实际训练窗口是 `max_seq_len=384`，见 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L11) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L23)。而审查报告里那些命中 NaN 的重点样本，原始 token 长度都在 `569` 到 `764` 之间，明显大于 `384`，见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L27) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L30)。这说明这次事故里，“窗口不够长导致必须截断”不是假设，而是当时真实发生过的训练条件。

### 3. 在 MiniMind 这条真实链路里，SFT 每一环发生了什么变化

你要求按 `tokenizer -> dataset -> model -> loss -> backward -> optimizer -> checkpoint -> resume/inference` 来讲，这个顺序正好最适合初学者。

#### 3.1 tokenizer：从普通文本编码变成“对话模板 + 编码”

SFT 样本先经过 `create_chat_prompt`。对你来说，最重要的理解不是“它调用了一个函数”，而是“模型实际吃到的并不是原始 JSON 结构，而是一整段展开后的对话文本”。这影响后面 assistant supervision span 的定位。

#### 3.2 dataset：labels 只给 assistant，且现在优先保留尾部窗口

当前本地修复版的流程是：

1. 用完整 prompt 得到完整 `input_ids`。
2. 在完整 `input_ids` 上生成完整 `labels`。
3. 如果超长，保留最后 `max_seq_len` 个 token 的窗口。
4. 同步裁剪 `input_ids` 与 `labels`。
5. 不足长度再做 padding。

代码在 [lm_dataset.py](../../dataset/lm_dataset.py#L106) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L128)。

这一步很关键，因为它避免了“先截断，再找监督”的错误顺序。也就是说，`max_seq_len` 这个参数本身不是问题，真正的问题是“当样本长度超过 `max_seq_len` 时，你到底按什么顺序、保留哪一段窗口”。当前修复版的思路是：既然 SFT 最核心的监督往往在尾部最后一轮 assistant 回复，那超过 `max_seq_len` 时，应优先保住尾部的监督区，而不是盲目保住前缀历史。

#### 3.3 model：模型结构本身没有因为 SFT 变掉

当前 `MiniMindForCausalLM` 还是普通 decoder-only Transformer 的因果语言模型头，forward 逻辑没有专门搞一套“SFT 专用网络”，见 [model_minimind.py](../../model/model_minimind.py#L245) 到 [model_minimind.py](../../model/model_minimind.py#L253)。

所以不要把这次问题理解成“模型结构坏了”。这次问题核心在数据监督语义和训练工程边界，不在注意力层或 RoPE 本身。

#### 3.4 loss：数学形式没变，但有效位置集合变得稀疏且更脆弱

SFT 的 `labels` 大量位置本来就应该是 `-100`。这本身不是 bug，反而是正确语义。真正的 bug 是：**所有位置都成了 `-100`。**

可以把“正常 SFT”和“zero-supervision SFT”对比成这样：

正常 SFT：

```text
input_ids:  [user..., assistant..., pad...]
labels:     [-100, -100, ..., token_a, token_b, ..., -100]
```

zero-supervision：

```text
input_ids:  [user..., system..., pad...]
labels:     [-100, -100, -100, -100, -100, ...]
```

前者是“只在 assistant 上打分”；后者是“根本没有题可判”。

#### 3.5 backward：loss 一旦是 NaN，梯度就可能被污染

这一段最好和真实训练代码一起看。当前 `backward` 发生在 [train_full_sft.py](../../trainer/train_full_sft.py#L134) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L147)：

```python
res = model(input_ids, labels=labels)
logits_loss = res.loss
total_loss = logits_loss + aux_loss
loss = total_loss / args.accumulation_steps
scaler.scale(loss).backward()
```

所以这里的逻辑链非常明确：

$$
\text{labels} \rightarrow \mathcal{L} \rightarrow \text{backward} \rightarrow \nabla_\theta \mathcal{L}
$$

参数更新的基本关系式是：

$$
\theta \leftarrow \theta - \eta g
$$

其中：

- $\theta$：模型参数。
- $\eta$：学习率。
- $g = \nabla_\theta \mathcal{L}$：loss 对参数的梯度。

如果只说“loss 是 NaN，所以梯度会坏”，还是太跳了。这里可以更严格地从 softmax + cross entropy 的导数往下推。

先固定一个有效监督位置 $(b,t)$。记该位置 logits 为向量：

$$
z = (z_1, z_2, \ldots, z_V)
$$

softmax 概率为：

$$
p_c = \frac{e^{z_c}}{\sum_{j=1}^{V} e^{z_j}}
$$

如果该位置真实类别是 $y$，单位置交叉熵是：

$$
\ell(z, y) = -\log p_y
$$

它对第 $c$ 个 logit 的导数是经典结果：

$$
\frac{\partial \ell}{\partial z_c} = p_c - \mathbf{1}[c = y]
$$

这里 $\mathbf{1}[c=y]$ 是指示函数，表示“如果 $c$ 就是真实类别，则取 1，否则取 0”。

现在把 `ignore_index=-100` 的 mask 也带进去。前面已经定义过：

$$
m_{b,t} \in \{0,1\}, \quad M = \sum_{b,t} m_{b,t}
$$

总 mean loss 是：

$$
\mathcal{L}
=
\frac{1}{M}\sum_{b,t} m_{b,t}\,\ell_{b,t}
\quad \text{(前提是 } M > 0\text{)}
$$

那么对某个位置某个类别 logit 的导数就是：

$$
\frac{\partial \mathcal{L}}{\partial z_{b,t,c}}
=
\frac{m_{b,t}}{M}
\left(
p_{b,t,c} - \mathbf{1}[c=y_{b,t}]
\right)
$$

这条式子非常关键，它把你关心的“`-100` 到底对梯度有什么严格影响”说清楚了：

1. 如果该位置 `label != -100`，那么 $m_{b,t}=1$。
   这个位置正常贡献梯度。
2. 如果该位置 `label = -100`，那么 $m_{b,t}=0$。
   这个位置对梯度的贡献**严格等于 0**，不是“很小”，也不是“近似没有”，而是公式上直接乘成 0。

所以，当 batch 里“只有一部分位置是 `-100`”时，正确结论是：

> 这些 ignore 位置本身不会产生 NaN，也不会单独产生梯度；它们只是既不计入 loss，也不计入梯度。

真正的问题出在整个 batch 全部被 ignore，也就是：

$$
M = 0
$$

此时前面的 mean loss 写法就不再成立，因为：

$$
\mathcal{L} = \frac{0}{0}
$$

这是未定义值。既然损失本身都未定义，那么“在这个点对参数求导”也就不再是正常实数上的导数问题。你可以把它理解成：

- 对于单个被 ignore 的位置，梯度是严格 0。
- 但对于“所有位置都被 ignore”的整个 batch，最终标量 loss 已经不是一个正常实数，而是未定义点。

这里“未定义点”不要理解成“程序会停在那里，什么都不给你”。数学上说“未定义”，指的是这个表达式不对应一个正常的实数值；而程序实现通常还是会继续算，只是它算出来的不是普通实数，而是特殊浮点值。对当前这条路径来说，最直接的结果就是：

$$
\frac{0}{0} \rightarrow \text{NaN}
$$

也就是 `Not a Number`。所以这里的逻辑不是：

```text
未定义 -> 什么都没有
```

而是：

```text
未定义 -> 当前浮点实现产出 NaN
```

当前 PyTorch 这条实现路径的实际行为，已经由本地 toy 验证给出证据：

```python
toy_loss_all_ignore = F.cross_entropy(
    toy_logits,
    toy_labels_all_ignore,
    ignore_index=-100,
    reduction="mean",
)
```

对应结果是 `loss=nan`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。

这里再补一个术语：我前面说的“非有限值（non-finite）”，指的是**不是正常有限实数的浮点结果**。在 PyTorch 里通常包括两类：

- `NaN`：不是一个数，常见来源就是 `0/0`、`inf - inf`、对非法输入做某些运算等。
- `Inf` / `-Inf`：正无穷或负无穷，常见来源是除以 0、数值溢出等。

而“有限值（finite）”指的是普通实数，比如 `0.5`、`-3.2`、`1024.0`。当前训练脚本里用的检查正是：

```python
torch.isfinite(tensor).all()
```

见 [train_full_sft.py](../../trainer/train_full_sft.py#L39) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L46) 和 [train_full_sft.py](../../trainer/train_full_sft.py#L79) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L87)。也就是说，只要 `loss`、`aux_loss` 或 `grad_norm` 里出现 `NaN`、`Inf`、`-Inf`，当前修复版都会立刻报错，而不会继续训练。

一旦这个 `nan` 标量进入：

```python
scaler.scale(loss).backward()
```

自动求导图里就会继续传播非有限值。更直观地说，链式法则里每一级都会用到上一级的导数：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
=
\sum_{b,t,c}
\frac{\partial \mathcal{L}}{\partial z_{b,t,c}}
\cdot
\frac{\partial z_{b,t,c}}{\partial \theta}
$$

如果 $\frac{\partial \mathcal{L}}{\partial z_{b,t,c}}$ 已经是 NaN 或包含 NaN，那么再乘上后面的 Jacobian、再求和，结果通常仍然会是 NaN。因为数值运算里：

$$
\text{NaN} \times a = \text{NaN}, \quad
\text{NaN} + b = \text{NaN}
$$

于是：

$$
g = \nabla_\theta \mathcal{L}
$$

里的对应分量也会变成 NaN。

如果你更习惯把梯度理解成“损失函数在某一点的导数”，那么这里可以这样看：

- 正常情况下，`backward()` 需要的是一个正常实数 loss，然后在这个实数点附近求导。
- 现在 loss 自己先变成了 `NaN`。
- 那么自动求导系统继续往下传的时候，传递的就不是“某个具体实数的导数”，而是“围绕 NaN 这个非法数值继续做浮点传播”。

这也是为什么我说“自动求导图里继续传播非有限值”。不是说 PyTorch 在这里做了某种神秘的额外数学，而是说：

1. 前向已经产出了 `NaN`。
2. 反向传播的链式法则会用到这个前向结果相关的梯度项。
3. 这些梯度项里一旦有 `NaN`，继续乘和加，通常还是 `NaN`。

接下来如果你不拦截，让优化器真的拿这些 `NaN` 梯度去更新参数，会发生什么？最基本的更新式还是：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta g
$$

假设某个参数分量的梯度已经是 NaN，那么：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \text{NaN} = \text{NaN}
$$

也就是说，这个参数分量本身就会被写坏成 NaN。下一次前向再使用这个参数时，后面的激活、logits、loss 也很可能继续产出 NaN，训练就会进入“坏参数继续制造坏输出”的恶性循环。

如果优化器是 AdamW 这类带状态的优化器，问题还不只在参数本身。它内部还维护一阶矩、二阶矩之类的状态。只要把 NaN 梯度喂进去，这些优化器状态也可能一起变成 NaN。这样就算你后面某一步输入了正常 batch，优化器内部状态也已经脏了，恢复难度会更高。

所以从工程角度看，`NaN loss -> NaN grad` 的危险性不只是“这一批白训了”，而是：

1. 当前这次梯度更新失去数学意义。
2. 参数可能被更新成 NaN。
3. 优化器内部状态也可能被污染。
4. 后续 checkpoint 如果保存下来，就会把坏参数和坏优化器状态一起固化。

这也正是为什么当前修复版要在两个地方做 fail-fast：

- `loss` 非有限时，直接 `zero_grad` 并抛错，见 [train_full_sft.py](../../trainer/train_full_sft.py#L39) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L46)。
- `grad_norm` 非有限时，在真正 `optimizer.step()` 之前再次拦截，见 [train_full_sft.py](../../trainer/train_full_sft.py#L79) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L91)。

你可以把这件事再压缩成一句严格但好记的话：

> 单个 `-100` 位置的梯度是 0；全 batch 都是 `-100` 时，mean loss 的分母变成 0，loss 变成未定义，当前实现返回 NaN，随后 backward 就会把 NaN 传进梯度。

#### 3.6 optimizer：梯度累积让“坏梯度会污染整组更新”

在当前训练脚本里，`loss` 会先除以 `accumulation_steps`，再 backward，见 [train_full_sft.py](../../trainer/train_full_sft.py#L144) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L147)。

累积时的梯度关系是：

$$
g_{\text{acc}} = \sum_{i=1}^{K} \nabla_\theta \left(\frac{\mathcal{L}_i}{K}\right)
$$

其中：

- $K$：`accumulation_steps`。
- $\mathcal{L}_i$：第 $i$ 个 micro-step 的 loss。

如果前 3 个 micro-step 都是正常有限值，第 4 个 micro-step 的 loss 是 NaN，那么：

$$
g_{\text{acc}} = g_1 + g_2 + g_3 + g_4
$$

只要 $g_4$ 里有 NaN，整个 `g_acc` 的对应分量就会变成 NaN。因为数值计算里：

$$
\text{finite} + \text{NaN} = \text{NaN}
$$

这就是为什么 NaN 会“污染训练过程”。污染不是文学表达，而是数值状态真的被传染了。

#### 3.7 checkpoint：如果保存时机不对，会把不完整训练历史固化下来

MiniMind 的 resume checkpoint 保存模型参数、优化器状态、epoch、step 和 scaler 状态，见 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104)。

但它没有保存：

$$
\{\theta.grad\}
$$

也就是所有参数当前还没落地的累积梯度。

这意味着：checkpoint 只能安全地落在“没有待提交梯度”的边界上。否则恢复后会丢失那部分尚未写进参数的训练历史。

#### 3.8 resume / inference：恢复训练和做推理读取的是两类文件

这是很多初学者很容易混淆的点。

- resume 用的是 `checkpoints/*_resume.pth` 这种“训练状态包”。
- inference 用的是 `out/*.pth` 这种“普通模型权重”。

当前 `eval_llm.py` 在 native 模式下是直接从 `out/{weight}_{hidden_size}.pth` 加载，见 [eval_llm.py](../../eval_llm.py#L49) 到 [eval_llm.py](../../eval_llm.py#L60)。

所以“resume checkpoint 有问题”和“推理权重一定有问题”不是同一句话；但如果 partial SFT 过程已经出现 NaN，或者中断点本身就不可信，那么这次 partial full SFT 留下来的这些权重文件、checkpoint 文件和相关训练状态，就都不该被当作可信成果使用。

### 4. 严重 1：旧版前缀截断语义为什么会制造 zero-supervision

#### 4.1 最短结论

旧版语义的问题，不是“截断本身”这三个字，而是“先保留前缀，再去找 assistant 监督”。这样一来，长对话最后一轮 assistant 回复可能整个不在窗口里，labels 自然全变成 `-100`。

换句话说，这里真正出问题的不是“项目里居然有 `max_seq_len` 这种长度限制”，因为任何真实训练都要在显存预算下设长度上限。真正的问题是：当样本长度超过这个上限时，旧版选择了最不适合 SFT 监督语义的保留方式。

#### 4.2 什么叫“前缀截断”

“前缀截断（prefix truncation）”就是：

$$
\text{truncated\_input} = \text{input\_ids}[0:\text{max\_seq\_len}]
$$

也就是只保留开头这段，后面的 token 全部丢掉。

与之相对，当前修复后的“尾部窗口”是：

$$
\text{window\_start} = \max(0, |\text{input\_ids}| - \text{max\_seq\_len})
$$

$$
\text{truncated\_input} = \text{input\_ids}[\text{window\_start}:]
$$

也就是优先保留最后一段。

#### 4.3 “末尾 assistant 回复被裁掉”在 token 和 labels 层面是什么意思

假设完整 token 序列是：

```text
[system][user1][assistant1][user2][assistant2]
```

这里要先补一句，避免误解：`assistant1` 和 `assistant2` 只要都是真实 assistant 回复，那么**它们都属于监督对象**，并不是说 `assistant1` 天生就不重要、`assistant2` 天生就更高级。更准确的说法是：

- 如果窗口足够大，`assistant1` 和 `assistant2` 都应该被保留下来，也都可以参与监督。
- 但如果样本太长，必须在“前面历史”和“后面最新内容”之间做取舍，那么**最后一轮 `assistant2` 往往更值得优先保留**。

原因不是它“身份更高”，而是它通常更接近这条训练样本最新一轮的问答目标。上面这个例子里，`assistant2` 对应的是 `user2` 之后的最新回答；而 `assistant1` 虽然也有监督价值，但它更像已经发生过的一轮历史对话。对 SFT 来说，模型往往更需要学会的是：**看到最新一轮用户输入后，当前应该怎样回答**。如果窗口实在装不下全部历史，那么优先保住最新一轮 user 和对应的 assistant，会更符合这条样本的训练目的。

可以看一个更具体的对话例子。假设原始样本是：

```text
system: 你是一个数学助理，请分步骤回答。
user1: 什么是分数？
assistant1: 分数表示把一个整体分成若干份后，取其中几份。
user2: 那 3/4 和 0.75 为什么相等？
assistant2: 因为 3/4 表示 3 除以 4，计算结果正好是 0.75。
```

如果窗口足够大，这五段内容都应该进入模型，`assistant1` 和 `assistant2` 这两段回复也都应该参与监督。

但如果窗口不够长，必须删掉一部分历史，那么从“这条样本当前最想教模型什么”这个角度看，通常更想保住的是：

```text
user2: 那 3/4 和 0.75 为什么相等？
assistant2: 因为 3/4 表示 3 除以 4，计算结果正好是 0.75。
```

因为这一对 `user2 -> assistant2` 才是这条样本最后一轮、最新的问答目标。`assistant1` 当然不是没用，它仍然是一个真实监督信号；但在“窗口实在装不下全部历史”的情况下，它更像是在为后面的新问题做铺垫，而 `assistant2` 才是模型当前最需要学会继续生成的那段答案。

你可以把它想成老师在课堂上出的连续两道题：

- 第一道题已经讲过一遍，属于前面的上下文。
- 第二道题是刚刚追问的新问题，老师现在更希望你先学会回答这道最新的问题。

所以这里说“优先保留 `assistant2`”，不是说 `assistant1` 不值得学，而是说**在被迫截断时，最新一轮问答通常更接近这条样本当前要保住的监督核心**。

所以这里说“更重要”，不是在说 `assistant1` 不算监督，而是在说“超长截断时的保留优先级”。带着这个限定条件，再看下面的例子就更容易理解了：如果序列太长，旧逻辑保留的是：

```text
[system][user1][assistant1][user2 前半段]
```

最后的 `[assistant2]` 全没了。

在 token 层面，就是：

```text
完整 input_ids 中本来存在 assistant2 对应的一段 token id
-> 前缀截断后，这段 token id 不再存在
```

在 labels 层面，就是：

```text
完整 labels 中本来有一段 assistant2 对应的真实 token id
-> 旧逻辑在截断后的前缀上重新 generate_labels
-> 前缀里找不到 assistant2 span
-> labels 全部保持初始化值 -100
```

#### 4.4 zero-supervision batch 到底是什么

最准确的定义是：进入一次训练 micro-step 时，整个 batch 统计下来没有一个位置的标签是有效监督 token，也就是：

$$
\sum \mathbf{1}[\text{labels} \neq -100] = 0
$$

当前脚本里，这个值正是：

```python
valid_label_tokens = int((labels != -100).sum().item())
```

见 [train_full_sft.py](../../trainer/train_full_sft.py#L110)。

如果 `valid_label_tokens == 0`，这就叫 zero-supervision batch。

#### 4.5 为什么它不是小误差，而是直接破坏训练语义

因为 SFT 的训练语义是：

> 模型读取完整上下文，但只在 assistant 回答区间上被监督。

如果 assistant 区间整个消失，那么：

> 模型仍然在读上下文，但没有任何位置被监督。

这已经不是“监督有点少”了，而是“这一步根本没有监督”。

类比 1：这像老师把题目发给你，但把参考答案那一页整张撕掉。不是“扣分少一点”，而是“根本没法判分”。

类比 2：这像你录了一段老师讲题的视频，但最后真正讲答案的 30 秒被剪掉了。前面铺垫再完整，也不能替代最后那段答案本身。

#### 4.6 在 MiniMind 哪个文件、哪段逻辑里发生

- 当前修复版位置在 [lm_dataset.py](../../dataset/lm_dataset.py#L106) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L128)。
- 旧版错误语义的证据和复现解释在 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L122) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L178) 以及 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L21) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L39)。

#### 4.7 证据

- 旧行为前 `10000` step 中有 `136` 个全 `-100` 样本。
- 你列出的 NaN step 全命中这些 zero-supervision step。
- 修复后同样前 `10000` step 审计中 `zero_supervision_steps_repaired=0`。

这些都在审查报告中明示，见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L27) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L39)。

#### 4.8 修复方案为什么成立

当前修复版做的是：

1. 在完整序列上先确定哪些位置属于 assistant。
2. 再去裁剪窗口。
3. 裁剪时优先保留最后一段。

这个顺序更符合 SFT 的真实目标，因为长对话里最值得保留的，通常正是最后一次 assistant 回答，而不是最前面的历史铺垫。

#### 4.9 最小验证方式

直接运行审查报告里给出的命令：

```bash
./.venv/bin/python tests/diagnose_sft_supervision.py --max_seq_len 384 --steps 10000 --seed 42 --focus_steps 980 1060 1880 2960 4800 7580 8220 8840
```

期待输出：

```text
zero_supervision_steps_original=136
zero_supervision_steps_repaired=0
```

这条命令不是训练命令，而是一个“离线审计脚本”命令。它不会更新模型参数，也不会跑 GPU 训练；它做的是把 SFT 数据集按训练时同样的随机顺序重新走一遍，然后同时对比两套标签生成语义：

1. 旧行为：先截断前 `max_seq_len` 个 token，再在截断后的窗口里找 assistant 监督区间。
2. 修复后行为：先在完整序列上生成 labels，再裁到尾部窗口。

脚本主体在 [tests/diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L1)。它的核心流程可以直接按函数理解：

1. `parse_args()` 读取数据路径、tokenizer 路径、`max_seq_len`、审计步数、重点 step 和输出路径参数，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L21) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L57)。
2. `build_prompt()` 用 `SFTDataset` 的 chat template 逻辑把原始 `conversations` 还原成训练时真正送进 tokenizer 的 prompt，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L174) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L178)。
3. `generate_labels()` 按 assistant 片段生成监督标签，非监督位置填 `-100`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L89) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L109)。
4. `analyze_prompt()` 同时计算完整序列、旧前缀截断窗口、修复后尾部窗口这三种视角下的监督 token 数，并判断最后一段 assistant 回复是否被窗口保留下来，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L122) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L171)。
5. `main()` 先构造输出文件，再用 `setup_seed(args.seed)` 和 `torch.randperm(len(dataset))` 复现训练时的数据打乱顺序，顺序审计前 `steps` 个 micro-step，并把终端输出同步写进 `tests/out/`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L187) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L312)。
6. 脚本最后还做了一个 CPU toy 验证：构造一组 `all labels = -100` 的样例，直接调用 `F.cross_entropy(..., ignore_index=-100, reduction="mean")`，验证它确实会返回 `nan`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。

上面这条命令里每个参数的作用分别是：

- `./.venv/bin/python`：使用项目虚拟环境，确保 `torch`、`transformers`、`datasets` 和本地 tokenizer 版本与项目一致。
- `tests/diagnose_sft_supervision.py`：执行这份监督诊断脚本，而不是执行训练入口。
- `--max_seq_len 384`：按这次真实出问题 run 的窗口长度来审计；也就是模拟“训练时每条样本最多只保留 384 个 token”的行为。
- `--steps 10000`：只审计前 `10000` 个 micro-step，因为当前已知异常证据就落在这个范围内，而且这次历史 run 也是在 `step=10000` 日志之后被用户主动 `SIGINT` 中断，见归档日志 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L518) 到 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L519)。
- `--seed 42`：复现当时训练脚本 `setup_seed(42)` 的打乱顺序；不带这个参数，就不能保证“第 980 step 对应的是同一条样本”。
- `--focus_steps 980 1060 1880 2960 4800 7580 8220 8840`：重点打印这些历史 NaN step 的详细分析，方便把日志里的异常 step 和诊断结果一一对上。
- 默认输出落盘：脚本现在会把整份终端输出同时写入 `tests/out/`，文件名里会带上 `max_seq_len`、`steps`、`seed` 和执行时间戳；如果你想自定义路径，可以额外传 `--output_path`。

如果把这条命令翻成一句更直白的话，它的意思就是：

> “请用和当时 full SFT 一样的 `max_seq_len=384` 与 `seed=42`，按同一批数据顺序重放前 `10000` 个 step，并告诉我旧截断逻辑里哪些 step 会把监督 token 全裁没，以及修复后这些 step 是否恢复正常。”

这里“只审计前 `10000` 个 micro-step”背后的考虑，不是说脚本理论上只能看 `10000` 步，也不是说它只会机械照抄日志；真实原因有三个：

1. 这是这次历史 run 已经实际走到、而且有原始日志证据覆盖到的区间。归档日志显示训练打印到了 `Epoch:[1/2](10000/905718)`，随后就是用户主动 `KeyboardInterrupt`，见 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L518) 到 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L519)。
2. 已知 NaN step 也全部落在这个区间内，也就是 `980 / 1060 / 1880 / 2960 / 4800 / 7580 / 8220 / 8840`。因此前 `10000` 步已经足够覆盖“本次已观测异常”的完整证据面。
3. 这是最小必要审计范围。脚本每审计一步，都要重新取样本、拼 prompt、重新 tokenizer，再比较旧窗口和新窗口逻辑；它虽然不做完整训练，但也不是纯字符串搜索。既然这次要验证的是“已发生的 NaN 是否能被 zero-supervision 解释”，那覆盖到历史异常区间即可，不需要无边界把后面几十万步都扫一遍。

如果后续问题变成“第 `10000` 步之后是否还存在同类风险”，当然可以把 `--steps` 调大；但那属于扩大审计范围，不是当前这条“最小验证”命令的必要前提。

这里还要把 “micro-step” 和 “optimizer step” 分清楚。当前训练循环里，`for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):` 的这个 `step` 就是文中说的 micro-step，见 [train_full_sft.py](../../trainer/train_full_sft.py#L86) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L94)。每进入一次 dataloader、拿到一个 batch、做一次 forward/loss/backward 机会，就记一个 micro-step；它未必会立刻触发一次真实参数更新，因为脚本还要等 `effective_backward_count == args.accumulation_steps` 才会执行 `optimizer.step()`，见 [train_full_sft.py](../../trainer/train_full_sft.py#L144) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L156)。这次历史 run 的归档参数里 `batch_size=1`、`accumulation_steps=6`，见归档 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L9) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L16)。所以在这次具体 run 里，一个 micro-step 可以近似理解成“喂进 1 条样本并完成一次训练循环迭代”，而不是“已经完成了一次参数更新”。

还有一个容易混淆的点：这个诊断脚本既不是“只根据历史日志文本做静态分析”，也不是“重新调用 GPU 把训练跑一遍”。它实际做的是一种离线重放式审计：

1. 它参考历史 run 的真实配置，例如 `max_seq_len=384`、`seed=42` 和已知异常 step。
2. 它重新读取当前数据集和 tokenizer。
3. 它用同样的随机种子和 `torch.randperm(len(dataset))` 重新构造“当时第 N 个 micro-step 会抽到哪条样本”的顺序，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L203) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L204)。
4. 它对这些样本重新做 prompt 构造、tokenizer 编码和标签窗口分析，判断旧逻辑与修复逻辑下的监督 token 数。

也就是说，它确实“重新跑了一遍数据顺序和标签分析”，但没有重新跑模型训练本身。这里不会像 `train_full_sft.py` 那样初始化模型、把 batch 搬到 CUDA、执行真正的 SFT forward/backward；脚本里唯一的 `cross_entropy` 只是一个 CPU toy 验证，用来证明“当 labels 全是 `-100` 时，这条 loss 路径会返回 `nan`”，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。因此更准确的说法是：

> 这份脚本是在 CPU 侧按历史配置重放数据顺序并重新计算监督标签，不是在 GPU 上重新训练模型；它利用历史日志来确定“要重点核对哪些 step”，但结论本身不是只从日志抄出来的，而是重新对数据和标签逻辑做了一次可复现审计。

本机本轮已实际执行这条命令，输出结果与文档预期一致，关键字段如下：

```text
zero_supervision_steps_original: 136
zero_supervision_steps_repaired: 0
focus_steps_hit_zero_supervision: [980, 1060, 1880, 2960, 4800, 7580, 8220, 8840]
all_ignore_labels_mean_loss: nan | isfinite=False
```

这几行分别说明：

1. 旧逻辑下，前 `10000` 个 micro-step 里有 `136` 个 step 出现了“整条样本没有任何有效监督 token”。
2. 修复后逻辑下，同样的前 `10000` 个 step 里，`zero supervision` 降到了 `0`。
3. 你文档里列出的 `980 / 1060 / 1880 / 2960 / 4800 / 7580 / 8220 / 8840` 这些历史 NaN step，全都命中了旧逻辑的 `zero supervision` 集合。
4. 当 labels 全部是 `-100` 时，当前这条 `CrossEntropy(ignore_index=-100, reduction="mean")` 路径在 CPU toy 验证里确实返回 `nan`，这就是训练侧必须在进入 forward/loss 之前先跳过 `valid_label_tokens == 0` batch 的直接原因。

再看重点 step 的详细输出，比如 `step=980` 的摘要是：

```text
step=980 | dataset_index=787784 | raw_token_length=636 | truncated_token_length=384
  full_supervision_tokens=122 | original_supervision_tokens=0 | repaired_supervision_tokens=122
  original_window=(0, 384) | repaired_window=(252, 636)
  last_assistant_span=[612, 636) | last_span_in_original=0 | last_span_in_repaired=1
  lost_due_to_prefix_truncation=1
```

它表达的意思是：

- 这条样本原始长度有 `636` 个 token，但训练窗口只有 `384`。
- 最后一段 assistant 回复落在 `[612, 636)`，也就是序列尾部。
- 旧逻辑只保留前缀窗口 `(0, 384)`，因此最后一次 assistant 回复完全被裁掉，导致 `original_supervision_tokens=0`。
- 修复后保留尾部窗口 `(252, 636)`，最后一段 assistant 回复重新回到窗口里，所以 `repaired_supervision_tokens=122`。

因此，`4.9` 这一段之所以叫“最小验证方式”，是因为它不需要真正启动一次完整 GPU 训练，也不需要等到 loss 再次变成 `nan`；只靠离线重放数据和一个 toy cross entropy 验证，就能把“旧版前缀截断会制造 zero-supervision，而 zero-supervision 会把当前 loss 路径推成 NaN”这条证据链闭环起来。

### 5. 严重 2：为什么 `all labels = -100` 会让交叉熵变成 NaN，并继续污染训练

#### 5.1 最短结论

`CrossEntropy(ignore_index=-100, reduction=mean)` 的“mean”只对有效标签求平均；当有效标签数是 0 时，平均分母为 0，结果就会变成 NaN。NaN 一旦进 backward 和梯度累积，就会把后续梯度、参数乃至 checkpoint 一起污染。

#### 5.2 公式怎么写

把有效监督位置集合记作：

$$
S = \{(b,t)\mid y_{b,t} \neq -100\}
$$

交叉熵可以写成：

$$
\mathcal{L} =
- \frac{1}{|S|}
\sum_{(b,t)\in S}
\log p_\theta(y_{b,t}\mid x_{b,\le t})
$$

其中：

- $|S|$ 是有效标签位置个数。
- $p_\theta(y_{b,t}\mid x_{b,\le t})$ 是模型在当前位置给正确 token 的概率。

当 `labels` 全是 `-100` 时：

$$
S = \varnothing
$$

于是：

$$
|S| = 0
$$

这时损失就变成了“除以 0 的平均”。从数学直觉上你就能看出，这不是一个正常可训练的数字。审查报告里给出的 CPU toy 验证已经表明，这条路径在当前实现里返回的就是 `nan`，见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L47) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L61)。

#### 5.3 为什么 NaN 不只是“这一步 loss 坏了”

因为训练是条流水线：

$$
\text{loss} \rightarrow \text{backward} \rightarrow \text{grad} \rightarrow \text{optimizer.step} \rightarrow \theta \rightarrow \text{checkpoint}
$$

如果 `loss = NaN`，那么：

1. `backward` 看到的是 NaN。
2. 反向传播得到的某些梯度会变成 NaN。
3. `.grad` 一旦含 NaN，和别的正常梯度相加后通常仍是 NaN。
4. `optimizer.step()` 使用 NaN 梯度更新参数，参数也可能变成 NaN。
5. 若此后保存 checkpoint，坏参数和坏优化器状态会一起写入文件。

这就是“污染训练过程”的具体含义。它不是一句抽象的风险话术，而是数值状态在训练图里逐层扩散。

#### 5.4 结合 MiniMind 当前训练循环，污染是怎么发生的

在当前修复后的代码里，已经做了三道防线：

- `valid_label_tokens == 0` 直接跳过，不进入 forward/loss，见 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L132)。
- `logits_loss`、`aux_loss`、`total_loss` 任一非有限，立即 `zero_grad` 并抛 `FloatingPointError`，见 [train_full_sft.py](../../trainer/train_full_sft.py#L39) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L46) 和 [train_full_sft.py](../../trainer/train_full_sft.py#L140) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L142)。
- `optimizer.step()` 前还会检查 `grad_norm` 是否有限，见 [train_full_sft.py](../../trainer/train_full_sft.py#L79) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L91)。

但审查报告描述的是旧版语义。旧版的问题在于：zero-supervision batch 能继续走到 model forward 和 cross entropy。只要那一步产出 NaN，又没有在 backward 前拦住，NaN 就会进入 `.grad`。

#### 5.5 一个很直观的小例子

假设 `accumulation_steps=3`，某一组累积里有 3 个 micro-step。

第 1 步：

```text
loss_1 = 1.2
grad_1 = [0.3, -0.1]
```

第 2 步：

```text
loss_2 = 0.9
grad_2 = [0.2, 0.4]
```

此时累积梯度大致是：

```text
grad_acc = [0.5, 0.3]
```

第 3 步如果是 zero-supervision，旧逻辑没有跳过：

```text
loss_3 = NaN
grad_3 = [NaN, NaN]
```

那么累积梯度变成：

```text
grad_acc = [0.5 + NaN, 0.3 + NaN] = [NaN, NaN]
```

然后 `optimizer.step()` 用这个梯度更新参数：

```text
theta_new = theta_old - lr * grad_acc
```

如果 `grad_acc` 是 NaN，`theta_new` 对应分量也就可能是 NaN。此时再保存 checkpoint，就把坏参数固化了。

#### 5.6 为什么修复方案成立

修复核心不是“给 `CrossEntropy` 打一个补丁”，而是把错误数据在进入 loss 前就挡住：

$$
\text{if } |S| = 0,\ \text{skip batch}
$$

这样最干净，因为 zero-supervision 本质上不是“一个可以继续优化的正常 batch”，而是“根本没有监督信号的 batch”。

#### 5.7 最小验证方式

最小静态验证是阅读 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L152)，确认 zero-supervision 不再进入 forward。

最小数值验证是审查脚本中的 toy cross entropy 验证，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。

### 6. 严重 3：什么叫“checkpoint 保存点与有效梯度累积边界不一致”

#### 6.1 最短结论

如果 checkpoint 保存在“梯度已经累了一半，但参数还没真正更新”的位置，而 checkpoint 又不保存这些未落地梯度，那么恢复训练后虽然会跳过对应 micro-step，实际却少做了一次本应发生的参数更新，这就叫 resume 语义不完整。

#### 6.2 什么是 micro-step

每从 DataLoader 取出一个 micro-batch，做一次：

```text
forward -> loss -> backward
```

这就叫一个 micro-step。它未必会立刻 `optimizer.step()`。

在 MiniMind 当前训练循环里，`step` 这个变量本身就是 micro-step 计数，见 [train_full_sft.py](../../trainer/train_full_sft.py#L102) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L110)：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    last_step = step
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
```

翻成人话就是：DataLoader 每吐出一次 batch，这个 `step` 就加 1。它表示“训练循环已经走到第几个 micro-step”，不是“参数已经更新了几次”。

#### 6.3 什么是 accumulation step

如果设置：

```text
accumulation_steps = K
```

那就表示连续做 $K$ 个 micro-step，把梯度累在 `.grad` 里，等第 $K$ 次之后再执行一次真正的参数更新。

#### 6.4 什么是“有效 optimizer update 边界”

先看当前代码里真正决定“该不该更新参数”的地方，见 [train_full_sft.py](../../trainer/train_full_sft.py#L144) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L161)：

```python
loss = total_loss / args.accumulation_steps

scaler.scale(loss).backward()
effective_backward_count += 1

if effective_backward_count == args.accumulation_steps:
    apply_optimizer_update(epoch, step, last_valid_label_tokens, lr)
    effective_backward_count = 0
    if pending_save_step is not None and is_main_process():
        save_training_state(
            epoch,
            step,
            wandb=wandb,
            reason='deferred_periodic',
            requested_step=pending_save_step,
        )
        pending_save_step = None
```

所以，当前 MiniMind 里真正的“有效 optimizer update 边界”不是简单的：

```text
step % accumulation_steps == 0
```

而是：

$$
\text{effective\_backward\_count} = K
$$

在这个边界上：

- 本轮该累积的梯度已经齐了。
- `optimizer.step()` 会真正执行。
- 参数会真正变化。
- 之后 `optimizer.zero_grad()` 把这组梯度清掉。

这才是“一个完整训练更新”真正完成的时刻。

这里故意用 `effective_backward_count`，而不是直接用 `step % K`，是因为当前代码允许某些 batch 因为 `valid_label_tokens == 0` 被跳过，见 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L132)。跳过的 micro-step 不会 `backward()`，也就不该算进“有效梯度累积边界”。

#### 6.5 为什么保存点如果落在“还有未落地累积梯度”的位置，会让 resume 不完整

因为这时存在一份重要状态：

$$
\{\theta.grad\}_{\text{pending}}
$$

它已经部分承载了训练历史，但还没写进参数。

如果 checkpoint 不保存它，那么恢复时就只拿回：

- 模型参数 $\theta$
- 优化器状态
- scaler 状态
- `epoch`
- `step`

却拿不回：

- 当前已累计但未落地的 `.grad`

这就像记账记到一半，只保存了账户余额和“已经看到了第几张发票”，但没保存桌上那张还没入账的草稿。恢复后你还把前几张发票视为“已经处理过了”，那中间这段钱就永远丢了。

#### 6.6 为什么“不保存 `.grad` 却跳过了对应 micro-step”会产生语义断层

恢复逻辑会根据 `step` 继续往后走。也就是说，前面那些 micro-step 会被认为“已经训练过了”，后续数据采样也不会再重喂一次。

但如果 checkpoint 保存时，这些 micro-step 的梯度还只是累在 `.grad` 里，没有真正 `optimizer.step()`，那恢复后就出现断层：

- 数据进度上，它们被视为“已经走过”。
- 参数更新上，它们却没有完整落地。

这就是“步数看似推进，梯度其实没完整落地”的真实含义。

#### 6.7 一个极小数字例子：`accumulation_steps=6` 时在第 4 个 micro-step 保存会丢什么

假设：

```text
accumulation_steps = 6
```

连续 6 个 micro-step 的梯度分别是：

```text
g1, g2, g3, g4, g5, g6
```

正确、不被中断的训练应该是：

```text
step1: backward(g1/6)
step2: backward(g2/6)
step3: backward(g3/6)
step4: backward(g4/6)
step5: backward(g5/6)
step6: backward(g6/6)
optimizer.step() with (g1+g2+g3+g4+g5+g6)/6
zero_grad()
```

现在假设在第 4 个 micro-step 后保存 checkpoint，而且这个 checkpoint 不保存 `.grad`。

保存瞬间的真实状态其实是：

```text
参数 theta 还没更新
.grad 里累着 (g1+g2+g3+g4)/6
step 记成 4
```

恢复后：

- 加载回来的参数还是保存时那个旧参数 `theta_old`。
- `.grad` 不存在或被清空。
- 训练逻辑会从 `step=5` 继续，前 4 个 micro-step 会被认为“已经过去了”。

于是恢复后的这组更新变成：

```text
step5: backward(g5/6)
step6: backward(g6/6)
epoch 末尾可能触发一次不完整 update，或直接进入下一轮
```

无论哪种细节实现，它都不等价于原本那次应有的：

$$
\frac{g_1 + g_2 + g_3 + g_4 + g_5 + g_6}{6}
$$

因为前 4 步贡献已经没了。模型恢复后少了什么？少的是：

- 本应由 `g1` 到 `g4` 提供的那部分梯度信息。
- 本应在第 6 步边界才落地的一次完整参数更新。

这就是“语义不完整”最具体的解释。

#### 6.8 修复前在代码里到底是怎么保存的，为什么会出问题

修复前的旧逻辑可以直接看上游引用 [train_full_sft.py](../../../../references/minimind/trainer/train_full_sft.py#L27) 到 [train_full_sft.py](../../../../references/minimind/trainer/train_full_sft.py#L70)：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    ...
    scaler.scale(loss).backward()

    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if (step % args.save_interval == 0 or step == iters) and is_main_process():
        ...
        lm_checkpoint(..., epoch=epoch, step=step, ...)
```

这个旧逻辑的关键问题是：**“是否更新参数”** 和 **“是否保存 checkpoint”** 是两套彼此独立的时钟。

- 参数更新时钟：`step % args.accumulation_steps == 0`
- 保存时钟：`step % args.save_interval == 0`

只要这两个周期不对齐，保存就可能发生在“梯度累到一半”的位置。

这次历史 run 的真实参数在归档 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L9) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L16)：

```text
batch_size=1
accumulation_steps=6
save_interval=5000
```

把这三个数字代进去，立刻能看出问题：

```text
5000 % 6 = 2
```

这表示旧逻辑在 `step=5000` 保存时，通常并不处在一次完整更新边界上。更直白地说：

- `step=4998`：刚好做完一次 `optimizer.step()`，梯度清空。
- `step=4999`：开始累计下一轮梯度，第 1/6 份进 `.grad`。
- `step=5000`：累计到第 2/6 份，旧逻辑此时就会直接保存 checkpoint。

也就是保存瞬间更接近下面这个状态：

```text
参数：还是 step=4998 更新后的参数
.grad：里面攒着 step4999 + step5000 贡献的 2/6 梯度
checkpoint.step：记成 5000
```

但 `lm_checkpoint` 保存的内容并不包含参数当前 `.grad`，只保存模型、优化器、epoch、step、scaler 等状态，见 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104)：

```python
resume_data = {
    'model': state_dict,
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'step': step,
    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    'wandb_id': wandb_id
}
for key, value in kwargs.items():
    if value is not None:
        if hasattr(value, 'state_dict'):
            ...
            resume_data[key] = raw_value.state_dict()
```

所以恢复训练时就会出现下面这个断层：

```text
恢复到了 step=5000
但是 step4999 和 step5000 累出来的那 2/6 梯度没被带回来
同时 DataLoader 又会从 step=5001 往后继续
```

这就是“数据进度过去了，但一部分本该参与参数更新的梯度没有真正落地”。

#### 6.9 在 MiniMind 当前代码里，修复之后到底做了什么，为什么成立

修复后的核心思想很简单：**`save_interval` 不再等于“立刻保存”，而只等于“发起一次保存请求”。真正写盘，必须等到没有待提交梯度的时候。**

当前实现见 [train_full_sft.py](../../trainer/train_full_sft.py#L116) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。先看“提出保存请求”的代码：

```python
if step % args.save_interval == 0 and step != iters and is_main_process():
    if effective_backward_count == 0:
        save_training_state(
            epoch,
            step,
            wandb=wandb,
            reason='periodic',
            requested_step=step,
        )
    elif pending_save_step is None:
        pending_save_step = step
        Logger(
            f'Checkpoint save deferred from micro-step {step} '
            f'until next optimizer update boundary'
        )
```

这里语义已经变了：

- 如果 `effective_backward_count == 0`，说明当前没有未落地梯度，可以立刻保存。
- 如果 `effective_backward_count != 0`，说明还有梯度攒在 `.grad` 里，这时不保存，只把“我想在 `step=5000` 保存一次”记到 `pending_save_step`。

然后再看“真正执行延后保存”的位置：

```python
if effective_backward_count == args.accumulation_steps:
    apply_optimizer_update(epoch, step, last_valid_label_tokens, lr)
    effective_backward_count = 0
    if pending_save_step is not None and is_main_process():
        save_training_state(
            epoch,
            step,
            wandb=wandb,
            reason='deferred_periodic',
            requested_step=pending_save_step,
        )
        pending_save_step = None
```

这段代码的实际含义是：

1. 先把这一轮应有的梯度累满。
2. 执行真正的 `optimizer.step()`。
3. 执行 `optimizer.zero_grad()`，把刚才那组梯度清掉。
4. 如果之前有人提过保存请求，现在才真正落盘。

所以修复后的语义变成：

```text
save_interval 命中
  -> 如果没有待提交梯度，立刻保存
  -> 如果还有待提交梯度，先挂起请求
  -> 等到下一次有效 optimizer update 完成后再保存
```

把这次历史 run 的真实参数代进去，理解会更直观：

```text
accumulation_steps = 6
save_interval = 5000
5000 % 6 = 2
```

如果附近没有 `skipped_no_supervision`，那在修复后的脚本里：

- `step=5000`：只会记录 `pending_save_step=5000`，不会立刻写 checkpoint。
- `step=5004`：第 6 个有效 backward 凑齐，执行 `optimizer.step()`。
- `step=5004` 更新完成后：才真正保存 checkpoint，并在日志里保留 `requested_step=5000` 这个信息。

这样保存出来的 checkpoint 对应的是：

```text
前面这 6 个有效 micro-step 的梯度已经全部落进参数
.grad 已经清空
checkpoint.step 指向一个完整更新之后的位置
```

于是恢复训练时就不会再出现“前几个 micro-step 被跳过了，但它们的梯度其实没写进参数”这个断层。

这就是为什么当前修复版**不需要额外序列化 `.grad`，也能保证 resume 语义完整**：因为它根本不在“半截梯度”状态下保存。

#### 6.10 最小验证方式

静态验证：

- 对照阅读旧逻辑 [references/minimind/trainer/train_full_sft.py](../../../../references/minimind/trainer/train_full_sft.py#L27) 到 [train_full_sft.py](../../../../references/minimind/trainer/train_full_sft.py#L70)。
- 对照阅读修复后 [train_full_sft.py](../../trainer/train_full_sft.py#L116) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。
- 阅读 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104)，确认 checkpoint 不保存 `.grad`。

理解验证：

- 自己手算一遍 `accumulation_steps=6, save_interval=5000` 的情况，确认旧逻辑会在 `step=5000` 直接保存，而修复后会把这次保存延后到下一次完整更新边界。
- 再手算一遍上面的“第 4 个 micro-step 保存”例子，确认“保存于半截梯度状态”和“保存于完整 update 之后”在 resume 语义上不是一回事。

### 7. 明确回答：这次问题是不是说明之前预训练权重整个都有问题

#### 7.1 最短回答

不能这样下结论。根据当前文档和代码证据，可以确认有风险的是“这次 partial full SFT 的中断工件与恢复语义”；不能直接推出“pretrain base weight 整个有问题”。

#### 7.2 严格区分四类对象

第一类，`pretrain base weight`：

- 它是新的 full SFT 建议使用的起点，修复报告要求后续从 `--from_weight pretrain` 启动，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L59) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L65)。
- 当前没有证据表明它已经被这次 SFT NaN 事件污染。
- 因此，关于它的保守表述应该是：“当前没有直接证据显示 pretrain 底座损坏。”

第二类，`full SFT partial weight`：

- 它来自这次已经出现 NaN 且中断过的 partial run。
- 修复报告明确说这次 partial full SFT 不允许用于推理、resume 或验收，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L7) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L14)。
- 因此，它是有风险的。

第三类，`resume checkpoint`：

- 它保存了模型、优化器、scaler、epoch、step，但不保存 `.grad`。
- 如果保存点不在有效更新边界上，就会有明确的语义断层风险。
- 因此，它是“当前可以判定有风险”的对象之一。

第四类，训练过程中的临时梯度状态：

- 这是最容易被忽略的部分。
- zero-supervision 导致的 NaN 首先污染的往往不是“已经写死在磁盘里的 base weight”，而是正在内存里累积的 `.grad` 和尚未完成的一次参数更新。
- 这部分状态因为不被 checkpoint 完整保存，所以恰恰最容易在 resume 时丢失或变得不等价。

#### 7.3 哪些是本机已验证事实，哪些只是工程判断

本机已验证事实：

- 当前本地代码确实只在 `out/*.pth` 上做推理，不读 `resume checkpoint`，见 [eval_llm.py](../../eval_llm.py#L49) 到 [eval_llm.py](../../eval_llm.py#L67)。
- 当前 `lm_checkpoint` 确实不保存 `.grad`，见 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L105)。
- 当前 `train_full_sft.py` 已加入 zero-supervision 跳过和延后保存逻辑，见 [train_full_sft.py](../../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。

源码阅读结论：

- SFT 的监督确实只来自 assistant span。
- `all labels = -100` 在当前模型 loss 路径下会走到 `CrossEntropy(ignore_index=-100)`。

文档 / 日志证据：

- 旧行为前 `10000` step 中出现 `136` 个 zero-supervision。
- 这次 partial full SFT 不允许继续使用。

工程判断：

- 新的 full SFT 直接从 pretrain 重启，比试图抢救 partial run 更可靠。
- 旧 partial full SFT 的中间状态和恢复语义不可信。

不能直接下结论的点：

- 不能直接说 pretrain 底座损坏。
- 不能直接说 partial full SFT 的所有字节都必然被 NaN 写坏。
- 不能直接说除了 zero-supervision 之外绝无其他 NaN 诱因。

### 8. 为什么报告建议“新的 full SFT 必须从 `--from_weight pretrain --from_resume 0` 启动”

这个建议不是保守过头，而是基于三层逻辑。

第一层，训练起点要来自可信权重。当前可被当作可信起点的是 pretrain 底座，不是这次 partial SFT 工件。

第二层，resume 语义已经不完整。旧 partial run 在 zero-supervision、NaN 和记忆边界不对齐的风险下中断，再加上 checkpoint 不保存 `.grad`，继续从 `--from_resume 1` 启动没有语义保证。

第三层，推理入口和训练入口的工件含义不同。既然 `eval_llm.py` 用的是普通权重，而修复报告又明确说这次 partial full SFT 不可用于推理，那就不应该再把它混成下一次正式 full SFT 的起点。

简化成一句话就是：

> 这次要重启的不是“从头预训练”，而是“从可信的 pretrain 底座重新做一次干净的 full SFT”；丢弃的是不可信的 partial SFT 中间工件和不完整的 resume 语义，不是整个项目此前所有权重。

## 项目落地点

### 真实文件路径与职责

- 项目规则与边界： [AGENTS.md](../../AGENTS.md)
- 项目定位： [README.md](../../README.md)
- 本次审查结论： [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md)
- 本次中断修复报告： [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md)
- SFT 数据语义： [dataset/lm_dataset.py](../../dataset/lm_dataset.py#L58)
- SFT 训练循环： [trainer/train_full_sft.py](../../trainer/train_full_sft.py#L24)
- 模型 loss 计算： [model/model_minimind.py](../../model/model_minimind.py#L245)
- 诊断脚本： [tests/diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L1)
- checkpoint 保存 / 恢复工具： [trainer/trainer_utils.py](../../trainer/trainer_utils.py#L63)
- 推理入口： [eval_llm.py](../../eval_llm.py#L41)
- SFT 训练窗口长度配置： [trainer/train_full_sft.py](../../trainer/train_full_sft.py#L227)
- 模型结构位置上限配置： [model/model_minimind.py](../../model/model_minimind.py#L27)
- 这次真实中断 run 的归档 README： [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md)
- 这次真实中断 run 的原始日志： [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log)

### 已实现

- 当前本地已实现修复后的尾部窗口裁剪。
- 当前本地已实现 zero-supervision 直接跳过。
- 当前本地已实现 non-finite loss / grad norm fail-fast。
- 当前本地已实现“保存请求延后到有效 optimizer update 边界”。
- 当前本地已经能明确区分“训练窗口 `max_seq_len`”和“模型结构位置上限 `max_position_embeddings`”，但这仍属于源码阅读结论，不是长上下文训练完成结论。

### 正在设计或已形成策略

- 文档层面已经明确：新的 full SFT 必须从 `pretrain` 干净启动，而不是从 partial resume 启动。
- 文档层面已经明确：这次 partial full SFT 不得写成“已完成 SFT”或“可推理权重”。

### 需要验证

- 新的 full SFT GPU 长时稳定性。
- 新的 full SFT 完成后的推理效果与 KV Cache / EOS / history 行为。
- 旧 partial run 是否还存在除 zero-supervision 之外的其他独立 NaN 触发机制，目前仍不能完全排除。

### 哪些内容能写进 README、实验记录或面试材料

可以保守写的：

- 我已经读通了 MiniMind 当前 SFT 的 `tokenizer -> dataset -> model -> loss -> backward -> optimizer -> checkpoint -> inference` 调用链。
- 我能解释为什么 assistant supervision span 决定 SFT 训练语义，为什么 `all labels = -100` 会导致 NaN，为什么 checkpoint 不在累积边界保存会让 resume 不等价。
- 我已经有本地代码与审查报告层面的证据，说明这次 partial SFT 工件不应继续使用。

暂时不能写的：

- 已完成新的 full SFT。
- 已验证新的 full SFT 推理质量达标。
- 已确认旧 partial run 的所有工件都完全可用或完全不可用。
- 已证明 pretrain 底座损坏。

## 面试官 / 评审者可能追问与回答

### 追问 1：SFT 和 pretrain 在监督信号上的本质差别到底是什么

回答：本质差别不是模型数学任务完全不同，而是“哪些 token 被当成老师要打分的位置”。MiniMind 当前模型端依旧是 shifted cross entropy，见 [model_minimind.py](../../model/model_minimind.py#L251) 到 [model_minimind.py](../../model/model_minimind.py#L252)。pretrain 通常让大部分正文 token 参与 loss；SFT 则通过 [SFTDataset.generate_labels](../../dataset/lm_dataset.py#L88) 只让 assistant span 参与 loss。也就是说，pretrain 学的是“广义语言续写”，SFT 学的是“在对话模板里，以 assistant 身份怎样回答”。

### 追问 2：为什么 assistant 回复被截掉会让样本完全失去监督

回答：因为旧版语义是“先前缀截断，再在截断后的 token 里找 assistant supervision span”。如果最后一轮 assistant 回复本来在序列尾部，而尾部整个被裁掉，那么 `generate_labels` 在截断后的序列里就找不到任何 assistant span，最后整条 `labels` 都保持初始化值 `-100`。这不是“监督变少一点”，而是有效监督位置集合 $S$ 直接变成空集。

### 追问 3：为什么 `all labels = -100` 会变成 NaN，而不是单纯返回 0

回答：因为当前使用的是 `CrossEntropy(ignore_index=-100, reduction=mean)`。它的 mean 是对有效监督位置求平均，也就是对 $S=\{y\neq -100\}$ 求平均。当 $S$ 为空时，分母 $|S|=0$，这条计算路径在当前实现的 toy 验证里返回 `nan`，见 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L233) 到 [diagnose_sft_supervision.py](../../tests/diagnose_sft_supervision.py#L247)。这也是为什么训练侧要在进入 forward/loss 前就拦截 zero-supervision。

### 追问 4：为什么 checkpoint 不保存 `.grad` 会影响 resume 语义

回答：因为梯度累积期间，模型参数还没更新，但 `.grad` 已经承载了部分训练历史。如果保存点落在累积中间，checkpoint 却只保存参数、优化器、scaler、epoch、step，不保存 `.grad`，恢复时这些已经做过的 micro-step 会被数据进度逻辑跳过，但它们的梯度贡献却没有被写进参数。这就造成“数据进度继续了，参数轨迹却漏了一段历史”。当前 `lm_checkpoint` 的保存内容可直接在 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104) 看到。

### 追问 5：这次事故到底影响了什么，不影响什么

回答：根据当前证据，可以明确有风险的是这次 partial full SFT 的中间状态、resume 语义和对应 partial 工件；不能直接说 pretrain 底座坏了。修复报告明确要求新的 full SFT 从 `--from_weight pretrain --from_resume 0` 启动，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L59) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L65)。这说明当前更像是“丢弃不可信的 partial SFT 工件，保留可信的 pretrain 底座重来”，而不是“整个项目的底层权重体系都报废”。

### 追问 6：`max_seq_len` 该怎么理解，调大调小分别会怎样

回答：先把它理解成“这次训练给单条样本开的最大窗口长度”，不要先把它理解成“模型智商上限”。在当前本地 SFT 脚本里它默认是 `768`，见 [train_full_sft.py](../../trainer/train_full_sft.py#L227)；而模型结构里还有 `max_position_embeddings=32768`，见 [model_minimind.py](../../model/model_minimind.py#L27)。这两个不是一回事。调小 `max_seq_len`，好处是更省显存、更容易跑起来，坏处是更容易截断上下文，甚至把最后一轮 assistant 回复裁掉；调大 `max_seq_len`，好处是能保留更多上下文和监督，坏处是显存、速度和 batch size 压力都会上升。对这次问题来说，关键不是“768 一定错”或“越大越好”，而是“当样本超出这个窗口时，SFT 必须优先保住真正参与监督的尾部 assistant 区间”。这正是当前修复版相对旧版更合理的地方。
