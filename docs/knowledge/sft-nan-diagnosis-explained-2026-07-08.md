# 从零讲明白 MiniMind 这次 SFT NaN、zero-supervision 与 resume 语义问题

## 事实边界说明

### 本机已验证事实

- 本轮已实际读取 [AGENTS.md](../../AGENTS.md)、[README.md](../../README.md)、[code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md)、[fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md)、[dataset/lm_dataset.py](../../dataset/lm_dataset.py#L58)、[trainer/train_full_sft.py](../../trainer/train_full_sft.py#L24)、[model/model_minimind.py](../../model/model_minimind.py#L245)、[scripts/diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L19)、[trainer/trainer_utils.py](../../trainer/trainer_utils.py#L63)、[eval_llm.py](../../eval_llm.py#L41)、归档 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md)、归档 [full-sft-dense-768-e2-20260708-070010.log](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log) 和归档 [full-sft-current-run.env](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-current-run.env)。
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

## 可直接口述回答（>=1000字）

如果让我面向一个刚开始接触 SFT 的读者，直接口述这次问题，我会先从最基础的一句开始讲：SFT 是“监督微调（Supervised Fine-Tuning）”。它不是换了一种完全不同的模型，也不是把 Transformer 改造成别的东西，而是在一个已经会做“下一个 token 预测”的语言模型上，用“问题长什么样、理想回答长什么样”的样本，再训练一遍，让模型更像一个会按对话格式回答的 assistant。

预训练（pretrain）和 SFT 的共同点是：它们在模型层都还是做“给定前文，预测下一个 token”。MiniMind 里的 loss 公式在模型端并没有因为进入 SFT 阶段而换掉，当前还是 [model_minimind.py](../../model/model_minimind.py#L251) 里的 shifted cross entropy。真正变化的，不是模型的数学骨架，而是监督信号。预训练时，通常整段文本大部分位置都参与 loss，只有 padding 之类无意义位置会被设成 `-100` 忽略；SFT 时，模型仍然会读完整个对话模板，但只会在 assistant 真正回答的那一段上被打分，user、system、padding 这些位置虽然仍在输入里，却不参与 loss。换句话说，预训练更像“海量阅读，让模型会说人话”；SFT 更像“拿着题目和标准答案，教它在对话场景下应该怎么回答”。

在 MiniMind 这条链路里，SFT 对 `tokenizer -> dataset -> model -> loss -> backward -> optimizer -> checkpoint -> resume/inference` 的影响可以逐段看。首先是 tokenizer 和 chat template：聊天数据不是普通文本，而是 `conversations` 结构，`SFTDataset` 会先把多轮消息通过 [create_chat_prompt](../../dataset/lm_dataset.py#L71) 展开成模型真正看到的一段字符串，再编码成 `input_ids`。接着是 dataset：SFT 最大的变化就发生在这里。`generate_labels` 会扫描 token 序列，只把 assistant 回复区间写进 `labels`，其他位置都写成 `-100`，见 [lm_dataset.py](../../dataset/lm_dataset.py#L88) 到 [lm_dataset.py](../../dataset/lm_dataset.py#L104)。再往后是 model 和 loss：模型照样前向传播，照样输出每个位置对词表的打分 `logits`，再通过交叉熵去比较“应该输出的 assistant token”和“模型实际给它的概率”。然后才是 backward、optimizer、checkpoint 和 resume：这些训练工程环节在数学上没有变成另一种训练，但它们对“监督是否存在”“loss 是否有限”“这次更新是否完整落地”非常敏感。一旦监督被错误裁掉，loss 可能直接变成 NaN；一旦保存点不在有效更新边界上，resume 就不再等价。

这里一定要单独把 `max_seq_len` 讲明白，因为它其实是这次问题里最容易让初学者卡住的开关。最通俗的理解是：`max_seq_len` 就像你给单条训练样本准备的一张纸，这张纸最多只能写这么多 token。超过这个长度，后面的内容就必须被裁掉。它不是一句抽象配置，而是直接决定“这次训练模型到底看到了多少上下文、保住了多少 assistant 回复、有没有把关键监督裁掉”。在当前本地脚本源码里，这个参数的默认值是 `768`，见 [train_full_sft.py](../../trainer/train_full_sft.py#L227)；但这次失败 / 中断 run 的真实启动参数不是默认值，而是 `max_seq_len=384`，见归档 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L11) 到 [README.md](../../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md#L23)。这也解释了为什么审查报告的监督诊断是围绕 `384` 这个窗口长度去复现实验顺序的。

`max_seq_len` 太小时，问题是上下文装不下，最后一截会被截断；如果最关键的最后一轮 assistant 回复刚好在被截掉的那一段里，就会直接影响监督语义。`max_seq_len` 太大时，也不是越大越好，因为训练时每条样本都要保留更长的 token、更多的中间激活，显存占用和计算量都会上升，速度也会变慢。对小白来说，可以先记一句非常实用的话：**`max_seq_len` 是“这次训练每条样本最多保留多长窗口”的预算，它太小会丢信息，太大会更慢、更吃显存。**

还有一个非常容易混淆的点是：`max_seq_len` 不等于“模型天生只能理解这么长”。当前模型结构里还有 `max_position_embeddings=32768`，见 [model_minimind.py](../../model/model_minimind.py#L27)。更通俗地说，`max_position_embeddings` 更像“模型这栋楼理论上最多有多少层位置编号”，而 `max_seq_len` 更像“你这次训练开放多少个工位给样本使用”。所以你看到训练脚本默认 `max_seq_len=768`，不能立刻理解成“模型结构最多就 768 token”；对这次具体事故，还要再往下一层看归档 README，那里记录的真实训练窗口是 `384`。

你最关心的第一类问题，是那句“旧版前缀截断语义会把末尾 assistant 回复整体裁掉，直接制造 zero-supervision batch”到底是什么意思。先解释“前缀截断”。它指的是：当一条对话编码后太长，超过 `max_seq_len` 时，旧逻辑保留的是序列最前面的 `max_seq_len` 个 token，也就是“前缀窗口”，而不是最后面的“尾部窗口”。可是在对话数据里，真正需要监督、真正最关键的 token，往往落在最后一轮 assistant 回复的尾部。如果你保留前面、裁掉后面，就可能把最后一轮 assistant 回复整个砍掉。用 token 层面的语言说，就是：原始完整 `input_ids` 里本来存在一段 assistant span，对应的 `labels` 本来应该有一串真实 token id；但旧逻辑是“先截断，再生成 labels”，于是被保留下来的前缀里已经没有那段 assistant 回复了，`generate_labels` 再去扫时，自然找不到任何需要监督的 assistant 区间，最后得到的就是整条样本 `labels` 全部等于 `-100`。这就叫 zero-supervision：模型虽然拿到了一串输入，但老师没有给它任何一个位置的标准答案，整个样本没有一个 token 会参与 loss。

这不是“小误差”，而是直接破坏训练语义。因为 SFT 的本质不是“把所有 token 都喂进去就行”，而是“让模型在 assistant 回答区间上挨批改”。如果最后 assistant 回答整段被裁掉，那这条样本对 SFT 来说就相当于没有老师。这就像老师把题目发给学生，但把参考答案整页撕掉，然后还继续算平均分。它不是“分数有点偏”，而是“这道题根本无法被评分”。审查报告里给出的证据正是这个：前 `10000` 个 step 中旧行为产生了 `136` 个全 `-100 labels` 样本，而且你列出的多个 NaN step 全部命中这些 zero-supervision step，见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L27) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L39)。

第二类问题，是为什么 `labels` 全是 `-100` 会把 `CrossEntropy(ignore_index=-100, reduction=mean)` 变成 NaN。这里的关键点是：交叉熵在 `reduction="mean"` 时，实际上是在所有“有效标签位置”上做平均。如果一条样本或一个 batch 里，一个有效标签都没有，那么平均时的分母就变成 0。可以把它想象成“0 个样本的平均值”，也就是一个没有定义好的结果。在 PyTorch 这条实现路径里，CPU toy 验证已经显示这种情形返回的是 `nan`，见 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L169) 到 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L193)。而且 NaN 不会只停留在“这一步 loss 坏了”。如果旧版训练循环没有在 forward 之前拦住 zero-supervision batch，那么 `loss=nan` 会进入 backward，梯度也可能变成 NaN；一旦 NaN 梯度和前面累积的正常梯度混在一起，整个 `.grad` 就可能被污染；之后 `optimizer.step()` 再根据 NaN 梯度更新参数，参数本身就可能变成 NaN 或非有限值；如果此时再保存 checkpoint，坏状态就被写进 checkpoint 里了。这就是报告里说的“NaN 会直接污染训练过程”。

第三类问题，是为什么 checkpoint 保存点如果不和有效梯度累积边界对齐，resume 语义就不完整。这里先要分清 micro-step 和 optimizer update。每次 DataLoader 吐出一个 micro-batch，训练循环走完一次 forward / backward，这叫一个 micro-step；如果设置了 `accumulation_steps=6`，那通常要连续做 6 个 micro-step，把 6 次梯度累加到参数 `.grad` 里，才会执行一次真正的 `optimizer.step()`。这一次真正改变参数的动作，才是“有效 optimizer update 边界”。如果 checkpoint 恰好保存在第 4 个 micro-step，说明前 4 次 backward 已经发生，梯度已经部分累积到 `.grad` 里，但参数还没有真正更新。问题在于，MiniMind 当前的 resume checkpoint 保存模型、优化器、scaler、epoch、step，却不保存 `.grad`，见 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L105)。于是恢复时，你虽然会从 `step=4` 之后继续，并通过 `SkipBatchSampler` 风格的跳步逻辑跳过前 4 个 micro-step，但那 4 步本来累起来、尚未落地的梯度已经丢了。恢复后的训练轨迹，就不再等价于“不中断地继续训练”。

这也是为什么这次问题不等于“预训练底座权重整个坏了”。更准确的区分应该是：当前有风险的，首先是那次 partial full SFT 过程里的中间状态，包括可能被 NaN 污染的局部梯度、可能不完整的 resume checkpoint、以及不能作为可信推理起点的 partial SFT 权重；而不是已经完成的 pretrain base weight 本身已经被证明损坏。根据当前文档和代码证据，可以确认的只有：新的 full SFT 不应该从这次 partial 工件 resume，而应该从 `--from_weight pretrain --from_resume 0` 干净启动，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L59) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L71)。不能确认的是：“pretrain 底座已经坏掉”或“所有 partial 权重都一定完全无效”这样的更强结论，因为当前没有直接证据支持。

如果把整件事压缩成一句最短总结，那就是：这次问题的核心不是模型结构坏了，而是 SFT 监督信号和训练工程边界同时出了问题。旧版数据裁剪让某些长对话样本失去全部监督，旧版训练循环又没有在进入 loss 之前拦截这些样本，于是 NaN 有机会进入梯度和 checkpoint；同时，旧版保存点不一定落在有效参数更新边界上，resume 又不保存未落地梯度，所以恢复出来的训练轨迹不再等价。也因此，最稳妥的做法不是继续抢救这次 partial run，而是基于可信的 pretrain 权重重新启动一次干净 full SFT。

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
\xrightarrow{\text{仅 assistant 区间参与 labels}}
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

也就是说，SFT 不是把 user 文本从输入里删掉，而是把 user 位置从监督里删掉。

#### 2.4 loss 公式表面相同，监督集合不同

MiniMind 当前模型端 loss 是：

$$
x = \text{logits}[..., :-1, :]
$$

$$
y = \text{labels}[..., 1:]
$$

$$
\mathcal{L} = \text{CrossEntropy}(x, y, \text{ignore\_index}=-100)
$$

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

参数更新的基本关系式是：

$$
\theta \leftarrow \theta - \eta g
$$

其中：

- $\theta$：模型参数。
- $\eta$：学习率。
- $g = \nabla_\theta \mathcal{L}$：loss 对参数的梯度。

如果 $\mathcal{L} = \text{NaN}$，那么 $g$ 通常也会变成 NaN 或包含 NaN。再往下传，就不是“这一步没学到东西”这么简单，而是“修改建议本身已经坏掉”。

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

#### 3.8 resume / inference：恢复训练和做推理读取的是两类工件

这是很多初学者很容易混淆的点。

- resume 用的是 `checkpoints/*_resume.pth` 这种“训练状态包”。
- inference 用的是 `out/*.pth` 这种“普通模型权重”。

当前 `eval_llm.py` 在 native 模式下是直接从 `out/{weight}_{hidden_size}.pth` 加载，见 [eval_llm.py](../../eval_llm.py#L49) 到 [eval_llm.py](../../eval_llm.py#L60)。

所以“resume checkpoint 有问题”和“推理权重一定有问题”不是同一句话；但如果 partial SFT 过程已经出现 NaN 或中断在不可信边界上，那么这次 partial full SFT 工件整体就都不该被当作可信成果使用。

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

其中真正最重要、最需要监督的是最后的 `[assistant2]`。如果序列太长，旧逻辑保留的是：

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
- 旧版错误语义的证据和复现解释在 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L69) 到 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L118) 以及 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L21) 到 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](../code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md#L39)。

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
./.venv/bin/python scripts/diagnose_sft_supervision.py --max_seq_len 384 --steps 10000 --seed 42 --focus_steps 980 1060 1880 2960 4800 7580 8220 8840
```

期待输出：

```text
zero_supervision_steps_original=136
zero_supervision_steps_repaired=0
```

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

最小数值验证是审查脚本中的 toy cross entropy 验证，见 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L169) 到 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L193)。

### 6. 严重 3：什么叫“checkpoint 保存点与有效梯度累积边界不一致”

#### 6.1 最短结论

如果 checkpoint 保存在“梯度已经累了一半，但参数还没真正更新”的位置，而 checkpoint 又不保存这些未落地梯度，那么恢复训练后虽然会跳过对应 micro-step，实际却少做了一次本应发生的参数更新，这就叫 resume 语义不完整。

#### 6.2 什么是 micro-step

每从 DataLoader 取出一个 micro-batch，做一次：

```text
forward -> loss -> backward
```

这就叫一个 micro-step。它未必会立刻 `optimizer.step()`。

#### 6.3 什么是 accumulation step

如果设置：

```text
accumulation_steps = K
```

那就表示连续做 $K$ 个 micro-step，把梯度累在 `.grad` 里，等第 $K$ 次之后再执行一次真正的参数更新。

#### 6.4 什么是“有效 optimizer update 边界”

就是满足下面条件的边界：

$$
\text{effective\_backward\_count} = K
$$

在这个边界上：

- 本轮该累积的梯度已经齐了。
- `optimizer.step()` 会真正执行。
- 参数会真正变化。
- 之后 `optimizer.zero_grad()` 把这组梯度清掉。

这才是“一个完整训练更新”真正完成的时刻。

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

#### 6.8 在 MiniMind 当前代码里，修复为什么成立

当前修复版做法是：`save_interval` 只是“提出保存请求”，真保存要等到下一次有效 optimizer update 边界，见 [train_full_sft.py](../../trainer/train_full_sft.py#L150) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。

这意味着：

- 如果此时没有待提交梯度，马上保存。
- 如果还有待提交梯度，就把这次保存请求挂起到 `pending_save_step`。
- 等下一次 `optimizer.step()` 完成后，再落盘。

这样保存时 `.grad` 已经用完并清空，不需要额外把 `.grad` 序列化，也不会出现“跳过了旧 micro-step，但它们的梯度从未落地”的断层。

#### 6.9 最小验证方式

静态验证：

- 阅读 [train_full_sft.py](../../trainer/train_full_sft.py#L150) 到 [train_full_sft.py](../../trainer/train_full_sft.py#L208)。
- 阅读 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104)，确认 checkpoint 不保存 `.grad`。

理解验证：

- 自己手算一遍上面的 `accumulation_steps=6` 数字例子，确认“保存于第 4 步”和“保存于第 6 步”在 resume 语义上不一样。

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
- 诊断脚本： [scripts/diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L19)
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

回答：因为当前使用的是 `CrossEntropy(ignore_index=-100, reduction=mean)`。它的 mean 是对有效监督位置求平均，也就是对 $S=\{y\neq -100\}$ 求平均。当 $S$ 为空时，分母 $|S|=0$，这条计算路径在当前实现的 toy 验证里返回 `nan`，见 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L169) 到 [diagnose_sft_supervision.py](../../scripts/diagnose_sft_supervision.py#L193)。这也是为什么训练侧要在进入 forward/loss 前就拦截 zero-supervision。

### 追问 4：为什么 checkpoint 不保存 `.grad` 会影响 resume 语义

回答：因为梯度累积期间，模型参数还没更新，但 `.grad` 已经承载了部分训练历史。如果保存点落在累积中间，checkpoint 却只保存参数、优化器、scaler、epoch、step，不保存 `.grad`，恢复时这些已经做过的 micro-step 会被数据进度逻辑跳过，但它们的梯度贡献却没有被写进参数。这就造成“数据进度继续了，参数轨迹却漏了一段历史”。当前 `lm_checkpoint` 的保存内容可直接在 [trainer_utils.py](../../trainer/trainer_utils.py#L85) 到 [trainer_utils.py](../../trainer/trainer_utils.py#L104) 看到。

### 追问 5：这次事故到底影响了什么，不影响什么

回答：根据当前证据，可以明确有风险的是这次 partial full SFT 的中间状态、resume 语义和对应 partial 工件；不能直接说 pretrain 底座坏了。修复报告明确要求新的 full SFT 从 `--from_weight pretrain --from_resume 0` 启动，见 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L59) 到 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](../fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md#L65)。这说明当前更像是“丢弃不可信的 partial SFT 工件，保留可信的 pretrain 底座重来”，而不是“整个项目的底层权重体系都报废”。

### 追问 6：`max_seq_len` 该怎么理解，调大调小分别会怎样

回答：先把它理解成“这次训练给单条样本开的最大窗口长度”，不要先把它理解成“模型智商上限”。在当前本地 SFT 脚本里它默认是 `768`，见 [train_full_sft.py](../../trainer/train_full_sft.py#L227)；而模型结构里还有 `max_position_embeddings=32768`，见 [model_minimind.py](../../model/model_minimind.py#L27)。这两个不是一回事。调小 `max_seq_len`，好处是更省显存、更容易跑起来，坏处是更容易截断上下文，甚至把最后一轮 assistant 回复裁掉；调大 `max_seq_len`，好处是能保留更多上下文和监督，坏处是显存、速度和 batch size 压力都会上升。对这次问题来说，关键不是“768 一定错”或“越大越好”，而是“当样本超出这个窗口时，SFT 必须优先保住真正参与监督的尾部 assistant 区间”。这正是当前修复版相对旧版更合理的地方。
