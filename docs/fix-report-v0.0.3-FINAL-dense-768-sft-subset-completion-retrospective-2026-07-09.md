# Fix Report v0.0.3 FINAL：Dense 768 SFT subset 完成复盘（2026-07-09）

## 1. 问题背景、目标与文档范围

本文只复盘 `v0.0.3` 阶段已经完成的 `Dense 768 SFT subset` 训练链路，不改训练代码、不重跑训练，也不把这次 subset run 冒充为正式 `full SFT` 完成。

本文重点回答五件事：

1. `pretrain_768.pth -> full SFT NaN 诊断 -> subset 计划 -> 旧版 e1 启动后中断 -> 改成 e2 + swanlab -> 20260708-191946 run 完成` 这条链路到底发生了什么。
2. 为什么当前 loss 曲线肉眼看起来非常震荡，但这不等于训练失败。
3. `epoch=2` 和 `batch_size=1` 在这次 subset run 里的真实作用、收益、代价和常见误区。
4. 当前仓库内到底有没有 `eval_llm.py --weight full_sft` 或 `eval_llm.py --weight full_sft_subset` 的真实推理落盘证据。
5. 上游 README 提到的 YaRN / RoPE 长文本外推，当前本地代码是否支持、这次流程是否真的启用过。

边界说明：

- `【本机已验证】` 本文确认这次 subset 训练工件已经落地，但不把“工件存在”扩写成“模型效果已达标”。
- `【本机已验证】` 本文不修改 [README.md](../README.md)。原因是本轮只是补 FINAL 复盘文档，没有新增对外版本能力，也没有把 `v0.0.3` 的项目主声明从 README 徽章改成别的版本机制。
- `【代码/日志支持】` 本文会引用 2026-07-08 已有计划、fix report、code review 和知识讲解文档建立时间线；这些文档用于补齐历史链路，但不替代本轮重新核验过的日志、权重和 checkpoint。

## 2. 证据来源与优先级

本文统一使用五类证据标签：

- `【本机已验证】`：本轮在当前仓库重新执行命令或重新读取文件得到的事实。
- `【代码/日志支持】`：当前代码、日志、manifest、既有复盘文档或结构化元数据可以支持，但本轮没有重新重放全流程。
- `【上游事实】`：来自上游 MiniMind 引用仓库 [README.md](../../../references/minimind/README.md) 与 [eval_llm.py](../../../references/minimind/eval_llm.py) 的事实。
- `【工程判断】`：基于当前证据给出的保守工程解释。
- `【待验证】`：当前仍不能下硬结论的部分。

本轮实际读取和核验的核心材料如下：

- 项目规则与总览：[AGENTS.md](../AGENTS.md)、[README.md](../README.md)、[docs/minimind-roadmap.md](minimind-roadmap.md)
- `v0.0.3` 历史文档：[plan-sft-subset-learning-run-v0.0.3-2026-07-08.md](plan-sft-subset-learning-run-v0.0.3-2026-07-08.md)、[fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md](fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md)、[fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md)、[code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md)、[knowledge/sft-nan-diagnosis-explained-2026-07-08.md](knowledge/sft-nan-diagnosis-explained-2026-07-08.md)、[plan-sft-and-inference-execution-plan-v0.0.3-2026-07-08.md](plan-sft-and-inference-execution-plan-v0.0.3-2026-07-08.md)
- 真实入口代码：[trainer/train_full_sft.py](../trainer/train_full_sft.py)、[trainer/trainer_utils.py](../trainer/trainer_utils.py)、[eval_llm.py](../eval_llm.py)、[model/model_minimind.py](../model/model_minimind.py)、[scripts/start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)
- 真实运行工件：[experiments/logs/full-sft-subset-current-run.env](../experiments/logs/full-sft-subset-current-run.env)、[experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log)、[experiments/logs/full-sft-subset-dense-768-e2-20260708-191946-memory.csv](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946-memory.csv)、[out/full_sft_subset_768.pth](../out/full_sft_subset_768.pth)、[checkpoints/full_sft_subset_768.pth](../checkpoints/full_sft_subset_768.pth)、[checkpoints/full_sft_subset_768_resume.pth](../checkpoints/full_sft_subset_768_resume.pth)、[trainer/swanlog/run-20260708_191956-8en8elmmsjtkpoazyfxxy/files/swanlab-metadata.json](../trainer/swanlog/run-20260708_191956-8en8elmmsjtkpoazyfxxy/files/swanlab-metadata.json)
- 上游引用：[README.md](../../../references/minimind/README.md)、[eval_llm.py](../../../references/minimind/eval_llm.py)
- pretrain 起点材料：[fix-report-v0.0.2-FINAL-dense-768-pretrain-completion-2026-07-08.md](fix-report-v0.0.2-FINAL-dense-768-pretrain-completion-2026-07-08.md)

补充说明：

- `【本机已验证】` 用户给出的 `rg -n "YaRN|inference_rope_scaling|2048|RoPE" README.md eval_llm.py model ../../../references/minimind/README.md ../../../references/minimind/eval_llm.py` 在仓库根目录下会失败，因为 `../../../references/...` 对当前工作目录并不存在。
- `【本机已验证】` 本轮改用等价且实际存在的 `../../references/minimind/...` 路径完成核验；而本文因为位于 `docs/` 目录，最终 Markdown 链接仍使用正确的相对路径 `../../../references/...`。

## 3. Dense 768 SFT subset 最终完成结论

`【本机已验证】` 这次真正完成的 run 是 [experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log) 对应的 `run_id=20260708-191946`。它满足以下完成证据：

- 日志头部明确记录：`save_weight=full_sft_subset`、`from_weight=pretrain`、`from_resume=0`、`epochs=2`、`max_train_samples=100000`。
- 日志中存在 `SFT subset config: original_train_samples=905718, subset_train_samples=100000`。
- 日志末尾到达 `Epoch:[2/2](100000/100000)`。
- 日志末尾存在 `Checkpoint saved at micro-step 100000 (reason=end_of_epoch, requested_step=100000)`。
- 权重与 checkpoint 已真实落盘：
  - [out/full_sft_subset_768.pth](../out/full_sft_subset_768.pth)
  - [checkpoints/full_sft_subset_768.pth](../checkpoints/full_sft_subset_768.pth)
  - [checkpoints/full_sft_subset_768_resume.pth](../checkpoints/full_sft_subset_768_resume.pth)

`【本机已验证】` 本轮执行 `torch.load('checkpoints/full_sft_subset_768_resume.pth', map_location='cpu')` 后，返回的是 `dict`，并且关键字段为：

- `epoch=1`
- `step=100000`
- `has_model=True`
- `has_optimizer=True`
- `has_scaler=True`

`【代码/日志支持】` 这里的 `epoch=1` 是 0 基存储，不表示“只训练了 1 个 epoch”；它与日志里的 `Epoch:[2/2](100000/100000)` 一起解释，表示第二个 epoch 已结束并完成最终保存。

`【本机已验证】` 这次 subset run 没有在日志里出现 `nan`，与 2026-07-08 早些时候那次 full SFT NaN 事件不是同一条完成记录。

`【本机已验证】` 这次 run 只能写成“Dense 768 SFT subset 训练工件完成落地”，不能写成“Dense 768 full SFT 已完成”。

## 4. 本次真实训练配置与真实运行边界

### 4.1 真实入口与调用链

`【本机已验证】` 这次 subset run 的真实主链路是：

`[scripts/start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh) -> [trainer/train_full_sft.py](../trainer/train_full_sft.py) -> [dataset/lm_dataset.py](../dataset/lm_dataset.py) / DataLoader / 日志 / checkpoint / out 权重`

其中：

- 启动脚本负责固定 `save_weight=full_sft_subset`、`from_weight=pretrain`、`from_resume=0`，并生成 [full-sft-subset-current-run.env](../experiments/logs/full-sft-subset-current-run.env)。
- 训练入口负责构造 [SFTDataset](../dataset/lm_dataset.py)、应用 subset、记录 micro-step 日志、按有效 optimizer update 边界保存 checkpoint。
- 训练产物最终写到 `out/` 与 `checkpoints/`，而不是写回正式 `full_sft_768` 路径。

### 4.2 本次 run 的真实参数

`【本机已验证】` 从启动脚本、manifest、训练日志和 swanlab 元数据交叉核对后，这次完成 run 的关键参数如下：

| 参数 | 结论 | 主要证据 |
| --- | --- | --- |
| `save_weight` | `full_sft_subset` | [start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)、[full-sft-subset-current-run.env](../experiments/logs/full-sft-subset-current-run.env)、[full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log) |
| `from_weight` | `pretrain` | 同上 |
| `from_resume` | `0` | 同上 |
| `epochs` | `2` | 同上 |
| `batch_size` | `1` | [start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)、[swanlab-metadata.json](../trainer/swanlog/run-20260708_191956-8en8elmmsjtkpoazyfxxy/files/swanlab-metadata.json) |
| `max_seq_len` | `384` | 同上 |
| `accumulation_steps` | `6` | 同上 |
| `learning_rate` | `1e-5` | 同上 |
| `save_interval` | `10000` | 同上 |
| `max_train_samples` | `100000` | 同上 |
| `train_sample_ratio` | `1.0` | 同上 |
| `train_subset_seed` | `42` | 同上 |
| `train_subset_mode` | `random` | 同上 |
| `use_swanlab` | 开启 | 同上 |

### 4.3 这次 run 的真实边界

`【本机已验证】` 这次虽然传入了 `train_sample_ratio=1.0`，但由于同时传入 `max_train_samples=100000`，最终 subset 仍然只取 `905718` 条原始样本中的 `100000` 条，日志里的 `effective_subset_ratio=0.110410` 已明确写出这一点。

`【本机已验证】` 这次 run 的 checkpoint 保存不是严格卡在 `10000 / 20000 / ...` 整数倍 micro-step，而是会在下一个有效 optimizer update 边界保存，所以日志中真实出现的是：

- `10002 (requested_step=10000)`
- `20004 (requested_step=20000)`
- `40002 (requested_step=40000)`
- `50004 (requested_step=50000)`
- `70002 (requested_step=70000)`
- `80004 (requested_step=80000)`

这与 [trainer/train_full_sft.py](../trainer/train_full_sft.py) 当前的 `pending_save_step` / `effective_backward_count` 语义一致。

`【本机已验证】` [full-sft-subset-dense-768-e2-20260708-191946-memory.csv](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946-memory.csv) 记录了 WSL 宿主内存、进程 RSS、swap 与 checkpoint mtime，但 `gpu_pid_mem_mb` 列当前全是 `[N/A]`。因此本文不会编造这次 run 的 GPU 峰值显存数字。

## 5. `v0.0.3` 过程时间线与关键转折

### 5.1 起点：Dense 768 pretrain 已完成

`【代码/日志支持】` `v0.0.3` 的 SFT 不是从随机初始化开始，而是以 [fix-report-v0.0.2-FINAL-dense-768-pretrain-completion-2026-07-08.md](fix-report-v0.0.2-FINAL-dense-768-pretrain-completion-2026-07-08.md) 记录的 `pretrain_768.pth` 为起点。

### 5.2 第一次 full SFT 发生 NaN 并被中断

`【代码/日志支持】` [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md) 与 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md) 记录了第一次 full SFT 的核心问题：

- 旧版 `SFTDataset` 可能产生全 `-100 labels` 样本。
- `CrossEntropy(ignore_index=-100)` 在该条件下会给出 `nan`。
- 旧版训练侧没有在 forward 前跳过 zero-supervision batch，也没有把 checkpoint 保存严格对齐到有效梯度边界。

### 5.3 随后先修语义，再引入 subset 学习链路

`【代码/日志支持】` [fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md](fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md) 记录了两件关键事：

- 训练入口新增 `max_train_samples / train_sample_ratio / train_subset_seed / train_subset_mode`。
- 启动脚本改成独立的 `full_sft_subset_*` 输出路径，避免污染正式 `full_sft_768` 工件。

### 5.4 旧版 subset 脚本先跑过一次 `epochs=1`，后被人工中断

`【代码/日志支持】` [plan-sft-subset-learning-run-v0.0.3-2026-07-08.md](plan-sft-subset-learning-run-v0.0.3-2026-07-08.md) 和 [fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md](fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md) 都写明：

- 更早一版脚本默认 `epochs=1`。
- 当时未启用 swanlab。
- 该次任务后续被用户人工中断。

`【待验证】` 当前仓库内没有找到那次旧版 `e1` run 的独立训练日志文件，因此这一节点应保守地写成“已有文档记录支持”，而不是“本轮重新从原始日志完整重放确认”。

### 5.5 当前脚本改为 `epochs=2 + swanlab`，并重新启动

`【本机已验证】` 当前 [start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh) 默认已经是：

- `subset_epochs="${SFT_SUBSET_EPOCHS:-2}"`
- `subset_use_swanlab="${SFT_SUBSET_USE_SWANLAB:-1}"`

`【本机已验证】` [full-sft-subset-current-run.env](../experiments/logs/full-sft-subset-current-run.env)、[full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log) 与 [swanlab-metadata.json](../trainer/swanlog/run-20260708_191956-8en8elmmsjtkpoazyfxxy/files/swanlab-metadata.json) 一致指向当前完成 run：

- `run_id=20260708-191946`
- `epochs=2`
- `wandb_project=MiniMind-Full-SFT-Subset`
- `command=train_full_sft.py --save_weight full_sft_subset ... --epochs 2 ... --use_wandb`

### 5.6 最终产物落地

`【本机已验证】` 2026-07-08 21:16 左右，这次 subset run 最终产出了：

- [out/full_sft_subset_768.pth](../out/full_sft_subset_768.pth)
- [checkpoints/full_sft_subset_768.pth](../checkpoints/full_sft_subset_768.pth)
- [checkpoints/full_sft_subset_768_resume.pth](../checkpoints/full_sft_subset_768_resume.pth)

对应 `stat` 时间分别为：

- `out/full_sft_subset_768.pth`：`2026-07-08 21:16:41 +0800`
- `checkpoints/full_sft_subset_768.pth`：`2026-07-08 21:16:41 +0800`
- `checkpoints/full_sft_subset_768_resume.pth`：`2026-07-08 21:16:42 +0800`

## 6. loss 曲线复盘：现象、代码语义、可能原因、不能下的结论

### 6.1 现象

`【本机已验证】` 当前 log interval 为 `20`，整个 2 epoch run 一共留下 `10000` 个 loss 日志点。直接看 [full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log) 会看到：

- 单点 loss 波动很大，范围大致从 `0.0109` 到 `4.3067`。
- 第一轮开头就出现 `4.0345 -> 2.8217 -> 3.5607 -> 4.0677 -> 2.6001` 这种跳动。
- 第二轮尾部仍然能看到 `0.0227`、`0.8125`、`2.8090`、`2.5706`、`2.6039`、`2.3578` 这样的混合波动。

`【本机已验证】` 但如果按本轮只读脚本做最粗的统计，仍能看到保守下行迹象：

- 第 1 个 epoch：`mean_first_100=2.8313`，`mean_last_100=2.0357`，全 epoch 均值 `2.1651`
- 第 2 个 epoch：`mean_first_100=2.1816`，`mean_last_100=1.9810`，全 epoch 均值 `1.9967`
- 全部日志点的 `rolling100` 从 `2.8313` 下降到中段 `2.0713`，最后约 `1.9810`

因此更准确的描述不是“完全没有下降”，而是“原始单点曲线非常抖，但粗粒度平均值仍有一定下降”。

### 6.2 代码语义：当前打印的不是 optimizer-step 平滑均值

`【本机已验证】` [trainer/train_full_sft.py](../trainer/train_full_sft.py) 当前训练循环的关键顺序是：

1. `res = model(input_ids, labels=labels)`
2. `logits_loss = res.loss`
3. `aux_loss = ...`
4. `total_loss = logits_loss + aux_loss`
5. 先对 `total_loss` 做 finite 检查
6. `loss = total_loss / args.accumulation_steps`
7. `scaler.scale(loss).backward()`
8. 日志打印使用的是 `current_loss = total_loss.item()`

也就是说：

- `【本机已验证】` 当前日志里的 `loss` 是每个 micro-step 的 `total_loss`。
- `【本机已验证】` 这个数发生在 `loss / accumulation_steps` 之前。
- `【本机已验证】` 它不是 6 个 micro-step 聚合后的 optimizer-step 均值。
- `【本机已验证】` 它也不是 rolling mean、EMA 或任何平滑统计。

因此，直接把这条原始曲线当成“每次参数更新后的平滑收敛曲线”，语义上就是错的。

### 6.3 为什么它会天然很抖

`【工程判断】` 这次曲线肉眼很抖，更可能是下面几类因素叠加，而不是单一异常：

- `batch_size=1`，每个日志点只看 1 条样本，样本难度与长度差异会直接暴露到单点 loss。
- `accumulation_steps=6` 只影响反向累积和参数更新频率，不会把日志点自动变成 6 条样本均值。
- 当前 SFT 样本的 assistant 监督 token 数天然不等长，不同样本可监督位置不同，单点损失方差会更大。
- log interval 为 `20`，但每个打印点依然只是“第 20 个 micro-step 的原始 total_loss”，不是过去 20 步平均。
- `【本机已验证】` 当前这次 subset 训练默认不是“直接取前 `100000` 条”。真实逻辑在 [trainer/train_full_sft.py](../trainer/train_full_sft.py) 里是：

  先看真实执行入口 [start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)。这个脚本当前默认写死的是：

  - `subset_max_samples="${SFT_SUBSET_MAX_SAMPLES:-100000}"`
  - `subset_seed="${SFT_SUBSET_SEED:-42}"`
  - `subset_mode="${SFT_SUBSET_MODE:-random}"`

  然后它会把这几个值原样传给训练入口：

  - `--max_train_samples ${subset_max_samples}`
  - `--train_subset_seed ${subset_seed}`
  - `--train_subset_mode ${subset_mode}`

  所以从“真实启动命令的来源”这层看，这次 run 先天就已经不是 `head` 模式，而是脚本默认的 `random` 模式。

  $$
  \text{subset\_indices} = \text{randperm}(N, \text{seed}=42)[:100000]
  $$

  也就是训练入口收到 `train_subset_mode=random` 后，会先对原始 `905718` 条样本做一次固定随机种子下的确定性随机打乱，再截取前 `100000` 个索引。因此这次 run 实际拿到的是“固定 seed=42 的随机子集”，不是“原始数据文件里的前 `100000` 条”。只有在 `train_subset_mode=head` 时，训练入口才会退化成真正的“取前 N 条”；但本次 manifest、启动脚本默认值和日志都明确写的是 `train_subset_mode=random`。

  另外还要再分清一层：先选出这 `100000` 条之后，每个 epoch 内部还会再次对这个 subset 的位置顺序做 `torch.randperm(len(train_ds))`。所以更准确地说，这次训练是“先固定随机抽出 10 万条样本，再在每个 epoch 内对这 10 万条的访问顺序重新打乱”。这也是为什么这里说“内容异质性仍然很高”，因为它不是按文件头部连续切出来的一段相对局部样本，而是从全量 `905718` 条里分散抽出来的一批随机子集。

### 6.4 这条曲线当前不能支持哪些结论

`【待验证】` 仅凭这条原始曲线，当前不能下以下结论：

- 不能下“训练已经发散”。
- 不能下“模型效果一般”。
- 不能下“第二轮没有意义”。
- 不能下“batch_size=1 一定不收敛”。
- 不能下“这次 subset 已达到正式 full SFT 的效果”。

如果要更稳地判断训练趋势，至少需要补：

- optimizer-step 级平均 loss
- rolling mean / EMA
- 固定 prompt 推理对照
- 独立验证集或固定样本评测

## 7. `epoch=2` 复盘：为什么会这么设、是否必要、这次项目里更合理的解释是什么

如果把这一段改成更适合口述和复盘的说法，可以直接讲成下面这样：

`【可直接口述】` 这次把 subset run 从 `epoch=1` 改成 `epoch=2`，更合理的理解不是“因为 SFT 天生就必须跑两轮”，而是“我想让这条学习链路多走一遍完整闭环”。因为在这个项目里，第一优先级不是追求 benchmark，而是确认 `pretrain -> SFTDataset -> forward/loss -> backward -> optimizer -> checkpoint -> out 权重` 这条链路在修完 NaN 之后，能不能稳定地连续运行、保存、结束。多跑一轮，最直接的价值是：同一批 `100000` 条 subset 数据会再被完整遍历一次，如果第二轮还不出 NaN、不出保存错位、不出中断后的奇怪状态，那就说明这次修复不是“侥幸撑过第一轮”，而是对训练闭环更有把握。

`【本机已验证】` 当前脚本默认 `epochs=2`，而 [plan-sft-subset-learning-run-v0.0.3-2026-07-08.md](plan-sft-subset-learning-run-v0.0.3-2026-07-08.md) 与 [fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md](fix-report-v0.0.3-dense-768-sft-subset-learning-run-2026-07-08.md) 也都明确写到：更早一版 subset 脚本默认是 `epochs=1`，后来才改成 `2` 并启用 swanlab。这说明 `2` 不是一个“历史默认就这样”的天然事实，而是一次有意识的工程调整。

如果从训练循环本身解释，当前学习率调度是：

$$
\text{lr}(t)=\text{base\_lr}\times \left(0.1 + 0.45 \times \left(1+\cos\left(\pi \frac{t}{T}\right)\right)\right)
$$

这里：

- $t$ 是当前的 micro-step。
- $T$ 是总 micro-step 数，也就是 `args.epochs * iters`。
- `base_lr` 在这次 run 里是 `1e-5`。

`【本机已验证】` 这次 subset run 的 `iters=100000`，`epochs=2`，所以：

$$
T = 2 \times 100000 = 200000
$$

这意味着学习率不是只为第一轮 `100000` 步设计的，而是为整整 `200000` 个 micro-step 设计的。也正因为如此，日志第二轮尾部能看到 `lr: 0.00000100`，说明这条调度曲线确实从开头一路走到了尾部，没有“第一轮刚开始还没看清趋势就提前结束”的问题。

如果再换一个更直观的比喻，可以把这次 `epoch=2` 想成“不是为了让学生多刷一遍题就一定更高分，而是为了确认这套教学流程第二遍不会突然散架”。第一轮更像“系统能不能从头跑到尾”，第二轮更像“同样的流程再来一遍，是否还稳定”。对当前这个 MiniMind 子任务来说，这种工程价值比“外部经验说 SFT 常跑几轮”更重要。

所以更稳妥的结论应该分三层说：

- `【本机已验证】` 这次项目里，`epoch=2` 确实是实际采用的配置，不是想象中的配置。
- `【工程判断】` 它的主要价值更像是二次遍历 subset、观察第二轮稳定性、验证 checkpoint 与最终权重完整落盘、让学习率调度完整走完。
- `【待验证】` 但当前证据还不足以证明“如果不是 2，就一定不行”。也就是说，它可以被解释为“工程上合理”，但不能被写成“已经证明必须”。

如果面试官或评审追问“那为什么不是 3 轮、4 轮”，更保守的答法是：当前项目目标不是做最优训练轮次搜索，而是先把修复后的 subset 学习链路稳定跑通。因此 `2` 更像一个“再验证一轮闭环”的工程选择，而不是一个从理论上推出来的最优值。

## 8. `batch_size=1` 影响分析：对显存、梯度方差、日志可读性、收敛观察和吞吐的影响

如果把这一段改成更适合口述的版本，也可以直接这样讲：

`【可直接口述】` 这次 `batch_size=1` 最容易让人误会的地方，是大家一看到 loss 很抖，就立刻把原因归结成“模型坏了”或者“训练快崩了”。其实更基础的原因是：当前每个 micro-step 只喂 1 条样本，日志里打印的又正好是这个单样本 micro-step 的原始 `total_loss`。那你看到的曲线，本质上更像“每次抽到一道题，这道题对模型有多难”，而不是“模型整体平均水平有多稳定”。

先把数学关系说清楚。当前单个 micro-step 的原始损失可以记成：

$$
L_t = \text{CE}(x_t, y_t)
$$

这里：

- $t$ 表示第 $t$ 个 micro-step。
- $x_t$ 表示这一步 forward 得到的 logits。
- $y_t$ 表示这一步对应的 labels。
- `CE` 就是 cross entropy。

在当前 [trainer/train_full_sft.py](../trainer/train_full_sft.py) 里，真正参与 `backward()` 的不是直接的 $L_t$，而是：

$$
\tilde{L}_t = \frac{L_t}{K}
$$

其中 $K = \text{accumulation\_steps} = 6$。

这一步的含义不是“日志里从此变成了 6 条样本的平均损失”，而只是“为了做梯度累积，先把每一步反向贡献缩小到六分之一”。换句话说：

- 打印出来给你看的，还是单个 micro-step 的原始 $L_t$。
- 真正累到参数更新里的，是连续 6 个 micro-step 的梯度贡献。

这也是为什么“每个优化步的等效样本数”和“每个日志点的统计稳定性”不是一回事。当前这一轮里，每次真实参数更新近似可以写成：

$$
\text{effective batch per update} \approx \text{batch\_size} \times \text{accumulation\_steps} = 1 \times 6 = 6
$$

但你不能因此直接说“这就等同于真正 batch size 6”。原因很简单：真正 batch size 6 的前向，是 6 条样本一起进模型、一起求一个 batch 平均；而现在是 6 条样本分 6 次进模型、6 次分别算 loss、6 次分别反向，再把梯度累起来。参数更新规模可能接近，但你看到的日志形状、每次前向的长度分布、单点噪声大小，并不一样。

可以用一个很直观的例子理解。假设第 1 条样本特别简单，第 2 条样本特别难，第 3 条样本回答区很长，第 4 条样本回答区很短，第 5 条样本几乎是模板化回复，第 6 条样本语义跳跃很大。若你是真正的 batch size 6，这 6 条样本会在同一次前向里被平均掉一部分波动；但现在是 `batch_size=1`，所以日志会先给你看“简单样本的低 loss”，再给你看“困难样本的高 loss”，于是曲线天然像心电图，而不是平滑坡线。

因此，`batch_size=1` 在这次项目里更合理的工程解释是：

- `【工程判断】` 它首先是在当前设备条件下压低激活显存压力，让 Dense 768 + `max_seq_len=384` 的 subset run 更容易先跑通。
- `【工程判断】` 它会显著放大单步梯度方差，因为每一步只看 1 条样本，样本难度变化会直接映射成 loss 抖动。
- `【工程判断】` 它会降低吞吐，所以需要 `accumulation_steps=6` 去换取更可用的有效更新频率。
- `【工程判断】` 它会让“肉眼看曲线判断收敛”变得更危险，因为你看到的主要是样本波动，而不是平滑后的整体趋势。

如果再用一个生活化比喻，真正 batch size 6 更像“老师看完 6 份作业后给一个平均评分”；而现在的 `batch_size=1 + accumulation=6` 更像“老师连续看 6 份作业，每看一份都先记一笔，最后再合起来决定这轮总评”。最后的总评可能接近，但你旁边围观时听到的每一声反馈，会比真正平均后的分数波动大得多。

所以这一节最重要的记忆点有三个：

1. `【本机已验证】` 这次日志抖，不等于训练一定坏；先分清它记录的是单样本 micro-step loss。
2. `【本机已验证】` `accumulation_steps=6` 影响的是更新边界，不会自动把日志点变成 6 样本均值。
3. `【工程判断】` 如果后续要更稳定地看趋势，应该补 rolling mean、optimizer-step 均值或固定 prompt 推理，而不是继续盯着原始单点曲线下结论。

`【待验证】` 另外，虽然 `batch_size=1` 在工程上通常意味着更省显存，但这次 [full-sft-subset-dense-768-e2-20260708-191946-memory.csv](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946-memory.csv) 没有提供可复核 GPU 峰值显存数字，所以本文仍然不把“省显存”写成某个精确显存结论。

## 9. 推理验证与结果保存位置核查

### 9.1 本地代码支持什么

`【本机已验证】` 当前 [eval_llm.py](../eval_llm.py) 已支持以下推理相关参数：

- `--weight`
- `--prompt`
- `--seed`
- `--use_cache`
- `--do_sample`
- `--stream`
- `--output_file`
- `--inference_rope_scaling`

`【本机已验证】` 运行 `direnv exec . python eval_llm.py --help` 也能确认这些参数确实在当前 argparse 中暴露。

### 9.2 2026-07-09 新增推理落盘证据

`【本机已验证】` 当前仓库已经存在真实的 `eval_llm.py --output_file` 落盘结果，统一放在：

- `experiments/inference/2026-07-09/full_sft_subset_768/`
- `experiments/inference/2026-07-09/pretrain_768/`

其中与 `full_sft_subset` 相关的真实结果文件至少包括：

- [fixed-prompt-greedy-cache-on.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-cache-on.jsonl)
- [builtin-prompts-greedy-cache-on-stream-20260709-101600.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-greedy-cache-on-stream-20260709-101600.jsonl)
- [builtin-prompts-greedy-cache-on-stream-max256-20260709-102104.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-greedy-cache-on-stream-max256-20260709-102104.jsonl)
- [builtin-prompts-greedy-cache-on-max128-20260709-102301.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-greedy-cache-on-max128-20260709-102301.jsonl)
- [builtin-prompts-sample-cache-on-t07-p09-max128-20260709-102330.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-sample-cache-on-t07-p09-max128-20260709-102330.jsonl)
- [builtin-prompts-greedy-cache-on-yarn-max128-20260709-102544.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-greedy-cache-on-yarn-max128-20260709-102544.jsonl)

与 `pretrain` 相关的真实结果文件至少包括：

- [fixed-continuation-greedy-cache-on-max128-20260709-102934.jsonl](../experiments/inference/2026-07-09/pretrain_768/fixed-continuation-greedy-cache-on-max128-20260709-102934.jsonl)
- [builtin-prompts-greedy-cache-on-max128-20260709-103003.jsonl](../experiments/inference/2026-07-09/pretrain_768/builtin-prompts-greedy-cache-on-max128-20260709-103003.jsonl)
- [continuation-prompts-greedy-cache-on-max128-20260709-103029.jsonl](../experiments/inference/2026-07-09/pretrain_768/continuation-prompts-greedy-cache-on-max128-20260709-103029.jsonl)
- [continuation-prompts-greedy-cache-on-max128-20260709-103045.jsonl](../experiments/inference/2026-07-09/pretrain_768/continuation-prompts-greedy-cache-on-max128-20260709-103045.jsonl)

### 9.3 这些推理文件目前能支持什么事实

`【本机已验证】` 现在已经可以明确写出：

- 这次 subset 训练完成后，确实实际执行过 `eval_llm.py` 推理验证。
- 至少验证过两个权重分支：
  - `--weight full_sft_subset`
  - `--weight pretrain`
- 至少验证过这些推理口径：
  - `full_sft_subset` 的固定 prompt 贪心推理
  - `full_sft_subset` 的内置 8 个 prompt 贪心推理
  - `full_sft_subset` 的内置 8 个 prompt 采样推理
  - `full_sft_subset` 的 `max_new_tokens=256` 贪心推理
  - `pretrain` 的内置问答 prompt 贪心推理
  - `pretrain` 的 continuation-style prompt 贪心推理

`【本机已验证】` 从 JSONL 结构化字段可以直接确认的共同参数包括：

- `device="cuda"`
- `use_cache=1`
- `historys=0`
- `open_thinking=0`

另外还可以从文件内容确认一些更具体的事实：

- [fixed-prompt-greedy-cache-on.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-cache-on.jsonl) 只有 1 条记录，对应 prompt 是“请用三句话介绍你自己。”，回答实际是一条简短自我介绍，并未真正展开成三句话。
- `full_sft_subset` 的内置 8 个 prompt 贪心结果里，多个回答出现明显的模板化自我介绍、重复段落或 `</think>` 残留。
- `full_sft_subset` 在“为什么天空是蓝色的”“请用Python写一个计算斐波那契数列的函数”这类问题上，能输出与题意相关的文本，但存在重复、循环展开或代码逻辑退化。
- `pretrain` 在 continuation-style prompt 上更像普通续写模型；在聊天问答 prompt 上虽然也能产出相关文本，但重复和套话问题也较明显。
- `【本机已验证】` [fixed-continuation-greedy-cache-on-max128-20260709-102934.jsonl](../experiments/inference/2026-07-09/pretrain_768/fixed-continuation-greedy-cache-on-max128-20260709-102934.jsonl) 与两份 continuation prompts 结果都表明：`pretrain_768.pth` 的基础语言建模能力是通的，它能围绕续写 prompt 生成相对连贯的段落，并且多条 continuation 记录是自然命中 `EOS` 结束，而不是一律撞到长度上限。
- `【本机已验证】` 但 `pretrain_768.pth` 仍然不是稳定的指令助手，而且在较长的贪心续写里，后半段已经能看到明显重复。例如 continuation-style 结果中会反复枚举相似条目、重复“智能交通”或重复句式；这说明它的基础续写能力已通，但长一点的贪心续写仍有退化苗头。

因此这里可以把两类权重的推理观察明确区分成两句话：

- `【本机已验证】` 推理链路已经跑通，含义是：`eval_llm.py -> 加载 tokenizer / 权重 -> generate -> decode -> JSONL 落盘` 这一整条链路已经在本机真实执行成功。
- `【本机已验证】` 但 [full_sft_subset_768.pth](../out/full_sft_subset_768.pth) 当前表现出更明显的重复、截断和模板残留问题；而 [pretrain_768.pth](../out/pretrain_768.pth) 的基础续写能力是通的，能自然生成 `EOS`，只是长一点的贪心续写后半段也已经出现退化和重复。

### 9.4 因此当前能写到什么程度

更保守、也更准确的表述应更新为：

- `【本机已验证】` 当前不仅确认训练产物存在，也确认已经做过真实推理，并且结果文件已经保存到 `experiments/inference/2026-07-09/`。
- `【本机已验证】` `full_sft_subset` 和 `pretrain` 都已经有 JSONL 推理记录，不再是“只有计划命令，没有执行证据”的状态。
- `【本机已验证】` 当前推理结果可以支持“推理链路已跑通、模型已可被权重加载并产出文本”这一级结论。
- `【本机已验证】` 同时也可以明确写出：`full_sft_subset_768.pth` 当前已经暴露出明显的重复、截断和模板残留问题。
- `【本机已验证】` 另外也可以明确写出：`pretrain_768.pth` 的基础语言建模能力是通的，continuation-style prompt 下能生成连贯段落并自然结束，但它仍不是稳定的指令助手，长一点的贪心续写后半段已有退化苗头。
- `【待验证】` 但当前仍不能把这些结果扩写成“模型效果已达标”或“subset SFT 已经优于 pretrain 的所有场景”，因为本轮还没有统一评分口径、固定评测标准或系统化对照表。

## 10. YaRN / RoPE 长文本外推核查

### 10.1 这条说法来自哪里

`【上游事实】` “推理时启用 YaRN 外推，可免训练地把上下文长度扩展到 2048 及以上”这条说法来自上游 [README.md](../../../references/minimind/README.md)，不是当前本地 [README.md](../README.md) 已经对本仓库运行事实做出的声明。

上游 README 中可以直接找到：

- 项目特性里写到支持通过 YaRN 实现 RoPE 长文本外推。
- `RoPE长度外推` 章节给出 `python eval_llm.py --weight full_sft --inference_rope_scaling` 的推理命令。
- `rope_scaling` 示例里写明 `original_max_position_embeddings = 2048`。

### 10.2 当前本地代码是否支持

`【本机已验证】` 当前本地代码确实支持这条能力：

- [eval_llm.py](../eval_llm.py) 暴露了 `--inference_rope_scaling`。
- [model/model_minimind.py](../model/model_minimind.py) 的 `MiniMindConfig` 有 `inference_rope_scaling` 开关。
- 当该开关开启时，`rope_scaling` 会被设置为：
  - `type="yarn"`
  - `factor=16`
  - `original_max_position_embeddings=2048`
  - `beta_fast=32`
  - `beta_slow=1`

`【上游事实】` 上游 [eval_llm.py](../../../references/minimind/eval_llm.py) 同样暴露了 `--inference_rope_scaling`，说明当前本地这部分不是凭空新增的新概念，而是沿着上游推理入口的同名能力在工作。

### 10.3 这次 subset 流程里有没有真的启用

`【本机已验证】` 现在已经找到一份与 YaRN 相关的真实推理落盘文件：

- [builtin-prompts-greedy-cache-on-yarn-max128-20260709-102544.jsonl](../experiments/inference/2026-07-09/full_sft_subset_768/builtin-prompts-greedy-cache-on-yarn-max128-20260709-102544.jsonl)

因此结论需要比之前更细一点：

- `【本机已验证】` 训练入口 [trainer/train_full_sft.py](../trainer/train_full_sft.py) 本身仍然没有 `--inference_rope_scaling`，因为它是训练脚本，不是推理脚本。
- `【本机已验证】` 但推理阶段已经至少落盘过一份带 `yarn` 命名的 `full_sft_subset` JSONL 结果文件，这说明“subset 完成后完全没做 YaRN 相关推理”的旧结论已经不成立。
- `【工程判断】` 由于当前 [eval_llm.py](../eval_llm.py) 的 JSONL 记录字段里还没有单独保存 `inference_rope_scaling` 布尔值，这份证据目前主要依赖文件命名与当次执行上下文，而不是依赖 JSONL 内部字段自描述。

因此这里最新、也最保守的结论应写成：

- `【本机已验证】` 本地代码支持 YaRN / RoPE 外推开关。
- `【上游事实】` 上游 README 明确宣称可在推理时通过 `--inference_rope_scaling` 启用。
- `【本机已验证】` 当前仓库已经出现了至少一份带 `yarn` 命名的 `full_sft_subset` 推理结果文件，可视为“已经做过一次 YaRN 相关推理尝试”的执行证据。
- `【待验证】` 但因为 JSONL 内部还没有 `inference_rope_scaling` 字段，所以后续若要让证据更硬，最好把该字段也写进输出记录。

## 11. 当前已经拿到什么、还没有拿到什么

### 11.1 已经拿到的

- `【本机已验证】` 一次完整结束的 `Dense 768 SFT subset` 训练 run。
- `【本机已验证】` 独立的 subset 普通权重、普通 checkpoint、resume checkpoint。
- `【本机已验证】` 一份能重建真实启动参数的训练日志、manifest、memory csv 和 swanlab metadata。
- `【本机已验证】` 一条没有再次出现 NaN 的 subset 学习链路。
- `【本机已验证】` 当前本地 `eval_llm.py` 的推理参数能力说明，以及 YaRN 开关的代码支持事实。
- `【本机已验证】` `full_sft_subset` 与 `pretrain` 的真实 JSONL 推理结果，且结果目录已经明确落到 `experiments/inference/2026-07-09/`。
- `【本机已验证】` 一次 `full_sft_subset` 的固定 prompt 推理、一组内置 8 prompt 推理、一组采样推理，以及一份带 `yarn` 命名的推理落盘文件。
- `【本机已验证】` `pretrain_768.pth` 在 continuation-style prompt 上已经表现出可用的基础语言建模能力，能生成连贯续写并自然命中 `EOS`。

### 11.2 还没有拿到的

- `【本机已验证】` 没有拿到正式 `full_sft_768` 完成结论。
- `【本机已验证】` 还没有拿到 `use_cache=0` 的真实对照输出文件。
- `【本机已验证】` 还没有拿到同一 prompt、同一口径下严格成对的 `YaRN off/on` 结构化对照结果说明。
- `【本机已验证】` 没有拿到 `--use_cache 0/1` 或 `--inference_rope_scaling on/off` 的最小对照结果。
- `【待验证】` 没有拿到可支撑“模型效果一般”或“模型效果达标”的本机推理证据。
- `【待验证】` 还没有把这些推理结果整理成统一的人工评分表或固定评测摘要。

## 12. 风险、未验证边界与下一步最小建议

### 12.1 当前仍需明确保留的边界

- `【待验证】` 当前不能把 subset run 写成正式 full SFT 完成。
- `【待验证】` 当前不能把原始 loss 曲线抖动写成“训练失败”。
- `【待验证】` 当前不能把训练产物存在写成“推理效果已验证”。
- `【待验证】` 当前不能把上游 README 的 YaRN 说明写成“本机本轮已启用成功”。

### 12.2 本轮验证中的两个特殊说明

`【本机已验证】` 本轮执行的日志摘要脚本里，`has_completed` 返回 `False`，不是因为训练没完成，而是因为日志尾部把 `Experiment ... has completed` 断成了两行。改为直接读取日志尾部后，可以看到完整的完成语义。

`【本机已验证】` 本轮执行的上游 `rg` 命令如果沿用仓库根目录下的 `../../../references/...` 会因路径不存在而失败；改用真实存在的 `../../references/...` 后，YaRN / RoPE 相关核验已完成。

### 12.3 下一步最小建议

1. `【工程判断】` 在不改训练语义的前提下，补一个只读 loss 平滑统计视图，例如基于现有日志额外计算 rolling mean / optimizer-step 均值，避免后续继续把原始 micro-step total_loss 当趋势图。
2. `【工程判断】` 当前已经有固定 prompt 和 builtin prompts 的 JSONL 落盘，下一步不再是“先跑起来”，而是要把这些文件整理成一个最小结论表，例如按“自我介绍 / 常识问答 / 代码题 / continuation”分组，明确哪些回答是模板化、哪些回答出现重复、哪些回答基本相关。
3. `【工程判断】` `eval_llm.py --output_file` 的归档规则现在已经有真实落地样例，不再只是建议。当前目录结构可继续沿用：

   `experiments/inference/<日期>/<权重名>_<hidden_size>/`

   例如：

   `experiments/inference/2026-07-09/full_sft_subset_768/`

   该目录下的文件名再显式编码测试口径，例如：

   - `fixed-prompt-greedy-cache-on.jsonl`
   - `fixed-prompt-greedy-cache-off.jsonl`
   - `fixed-prompt-greedy-yarn-on.jsonl`
   - `fixed-prompt-greedy-yarn-off.jsonl`

   这样回头看到文件名就能直接知道：它是哪一天跑的、对应哪个权重、有没有开 cache、有没有开 YaRN、是不是固定 prompt 的贪心验证。
4. `【工程判断】` 下一步最有价值的是补一组真正成对的对照文件：

   - `cache on` vs `cache off`
   - `YaRN off` vs `YaRN on`

   并保证 prompt、seed、`do_sample`、`max_new_tokens` 全部一致。

### 12.4 建议的 `eval_llm.py` 调用范式

下面这些命令是“建议如何继续做”，不是“本轮已经全部执行过”。其中 `experiments/inference/2026-07-09/` 目录结构本身已经被真实使用过。

#### 12.4.1 先建结果目录

```bash
mkdir -p experiments/inference/2026-07-09/full_sft_subset_768
```

#### 12.4.2 最小固定 prompt 验证

用途：先确认 `full_sft_subset_768.pth` 能否被正常加载，并把一条固定 prompt 的结果落成 JSONL。

```bash
direnv exec . python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_subset \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --prompt '请用三句话介绍你自己。' \
  --seed 42 \
  --do_sample 0 \
  --use_cache 1 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.85 \
  --max_new_tokens 128 \
  --historys 0 \
  --open_thinking 0 \
  --stream 0 \
  --output_file experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-cache-on.jsonl
```

#### 12.4.3 cache on/off 最小对照

用途：固定同一个 prompt、同一个 seed、同一个贪心口径，只切 `use_cache`，看结果文本和 `generated_token_ids` 是否一致。

```bash
direnv exec . python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_subset \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --prompt '请用三句话介绍你自己。' \
  --seed 42 \
  --do_sample 0 \
  --use_cache 1 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.85 \
  --max_new_tokens 128 \
  --historys 0 \
  --open_thinking 0 \
  --stream 0 \
  --output_file experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-cache-on.jsonl

direnv exec . python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_subset \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --prompt '请用三句话介绍你自己。' \
  --seed 42 \
  --do_sample 0 \
  --use_cache 0 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.85 \
  --max_new_tokens 128 \
  --historys 0 \
  --open_thinking 0 \
  --stream 0 \
  --output_file experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-cache-off.jsonl
```

#### 12.4.4 YaRN off/on 最小对照

用途：先在本机真正留下 “没开 YaRN” 和 “开了 YaRN” 的两份结构化输出证据，避免后面再把计划命令误写成已执行事实。

```bash
direnv exec . python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_subset \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --prompt '请概括下面这段长文本的主要观点：……' \
  --seed 42 \
  --do_sample 0 \
  --use_cache 1 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.85 \
  --max_new_tokens 128 \
  --historys 0 \
  --open_thinking 0 \
  --stream 0 \
  --output_file experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-yarn-off.jsonl

direnv exec . python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_subset \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --prompt '请概括下面这段长文本的主要观点：……' \
  --seed 42 \
  --do_sample 0 \
  --use_cache 1 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.85 \
  --max_new_tokens 128 \
  --historys 0 \
  --open_thinking 0 \
  --stream 0 \
  --inference_rope_scaling \
  --output_file experiments/inference/2026-07-09/full_sft_subset_768/fixed-prompt-greedy-yarn-on.jsonl
```

#### 12.4.5 调用时最容易犯的错

- `【本机已验证】` `--weight` 要和 `out/` 里的真实文件前缀一致。这次 subset 工件是 `full_sft_subset_768.pth`，所以这里应写 `--weight full_sft_subset`，不是 `full_sft`。
- `【本机已验证】` `--output_file` 如果目标文件已经存在，当前 [eval_llm.py](../eval_llm.py) 会拒绝覆盖，所以重复跑之前要么删旧文件，要么换新文件名。
- `【本机已验证】` `--historys` 必须是非负偶数；当前脚本会直接校验这一点。
- `【工程判断】` 做最小对照时，优先固定 `--seed 42`、`--do_sample 0`、`--stream 0`，否则输出随机性会干扰比较。

## 13. 本轮只读验证命令记录

`【本机已验证】` 本轮实际执行并用来支撑本文结论的命令包括：

- `pwd`
- `git status --short --branch`
- `direnv exec . python -V`
- `direnv exec . python - <<'PY' ... torch.load('checkpoints/full_sft_subset_768_resume.pth') ... PY`
- `direnv exec . python - <<'PY' ... 'Epoch:[2/2](100000/100000)' ... PY`
- `rg -n "eval_llm.py|full_sft_subset|full_sft_768|output_file|inference_rope_scaling|MiniMind-Full-SFT" docs experiments out checkpoints scripts`
- `rg -n "YaRN|2048|inference_rope_scaling|RoPE" ../../references/minimind/README.md ../../references/minimind/eval_llm.py model/model_minimind.py eval_llm.py`
- `direnv exec . python eval_llm.py --help`
- 只读日志统计脚本：解析 [full-sft-subset-dense-768-e2-20260708-191946.log](../experiments/logs/full-sft-subset-dense-768-e2-20260708-191946.log) 的 epoch 均值、首尾 `100` 点均值和 `rolling100`

## 14. 结论

`【本机已验证】` `v0.0.3` 这一轮真正闭环的是一次 `Dense 768 SFT subset` 的 2 epoch 完成 run，而不是正式 `full SFT` 完成。它以 [out/pretrain_768.pth](../out/pretrain_768.pth) 对应的 pretrain 底座为起点，经历了 full SFT NaN 诊断、subset 学习链路设计、旧版 `epochs=1` 尝试后人工中断、脚本改成 `epochs=2 + swanlab`，最终在 `run_id=20260708-191946` 产出了完整的 subset 权重和 checkpoint。

`【本机已验证】` 当前 loss 曲线之所以看起来非常震荡，主要原因不是“已经证明训练失败”，而是当前日志记录的是每个 micro-step 的原始 `total_loss`，又恰好叠加了 `batch_size=1`、样本难度差异和未做平滑统计的观察口径。粗粒度统计仍然显示一定下降。与此同时，当前已经补上了真实推理落盘证据：`full_sft_subset` 和 `pretrain` 都有 JSONL 结果文件，说明推理链路已经跑通、模型已经能被权重加载并产出文本。其中 [full_sft_subset_768.pth](../out/full_sft_subset_768.pth) 当前已经明显暴露出重复、截断和模板残留问题；而 [pretrain_768.pth](../out/pretrain_768.pth) 的基础语言建模能力是通的，能围绕续写 prompt 生成连贯段落，并且自然生成 `EOS`，但它仍然不是稳定的指令助手，后半段出现重复，说明长一点的贪心续写仍有退化苗头。因此现阶段更适合写成“推理链路已通，但两类权重都还有清晰的输出边界”，而不是“效果达标”。

`【本机已验证】` 当前本地代码支持 `eval_llm.py --inference_rope_scaling` 与模型侧 YaRN / RoPE 外推配置；`【上游事实】` 上游 README 也明确宣称可通过该参数把上下文长度扩展到 `2048` 及以上；并且 `【本机已验证】` 当前仓库已经出现了一份带 `yarn` 命名的 `full_sft_subset` 推理结果文件，说明本机至少做过一次 YaRN 相关推理尝试。`【待验证】` 只是由于 JSONL 内部还没单独记录 `inference_rope_scaling` 字段，后续还需要把这层证据补强。
