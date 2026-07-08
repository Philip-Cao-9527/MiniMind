# Fix Report v0.0.3：Dense 768 SFT Subset 学习链路改造（2026-07-08）

## 1. 范围

本报告记录的是一次 `Dense 768 full SFT` 学习链路改造，不是正式 full SFT 完成报告。

本轮目标是在不修改原始 `dataset/sft_t2t_mini.jsonl`、不覆盖正式 `full_sft_768` 输出路径、且不破坏当前 SFT 修复语义的前提下，为本机增加一条可控的 subset short run 入口。

## 2. 改动文件

- [trainer/train_full_sft.py](../trainer/train_full_sft.py)
- [scripts/start_full_sft_dense768_e2.sh](../scripts/start_full_sft_dense768_e2.sh)
- [scripts/monitor_full_sft_memory.sh](../scripts/monitor_full_sft_memory.sh)
- [scripts/start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)
- [scripts/monitor_full_sft_subset_memory.sh](../scripts/monitor_full_sft_subset_memory.sh)
- [docs/plan-sft-subset-learning-run-v0.0.3-2026-07-08.md](plan-sft-subset-learning-run-v0.0.3-2026-07-08.md)

## 3. 关键修复与新增能力

### 3.1 训练入口新增 subset 参数

[trainer/train_full_sft.py](../trainer/train_full_sft.py) 新增：

- `--max_train_samples`
- `--train_sample_ratio`
- `--train_subset_seed`
- `--train_subset_mode`

实现边界：

- 默认值保持全量行为不变。
- subset 逻辑放在 `SFTDataset` 构造之后、`DistributedSampler / DataLoader` 之前。
- 使用 `torch.utils.data.Subset` 包装原始 `SFTDataset`，不生成新的 JSONL。
- `random` 模式使用 `torch.Generator().manual_seed(train_subset_seed)` 与 `torch.randperm` 做确定性随机子集。
- `head` 模式只取前 `N` 条，便于快速定位问题。
- 当用户配置 subset 且同时开启 `--from_resume 1` 时，训练入口直接报错，避免把本次短跑试验数据顺序与历史 resume checkpoint 混用。
- 日志会明确打印原始样本数、subset 后样本数、ratio、seed、mode、`from_weight`、`from_resume` 以及是否仍然是 `pretrain / 0`。

### 3.2 保留当前已修复的 SFT 训练语义

本轮没有改坏以下防线：

- `SFTDataset` 仍然是“先完整生成 labels，再裁到尾部窗口”。
- `valid_label_tokens == 0` 仍然直接跳过。
- 非有限 `loss / grad norm` 仍然 fail-fast。
- checkpoint 仍然延后到有效 optimizer update 边界落盘。

### 3.3 新增 subset 启动脚本

新增 [scripts/start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)。

特点：

- 独立 `screen` session。
- 独立日志、监控 CSV、manifest 和 writer lock。
- 只检查 `full_sft_subset_*` 自己的输出路径，不碰正式 `full_sft_768` 文件。
- 强制使用：
  - `save_weight=full_sft_subset`
  - `from_weight=pretrain`
  - `from_resume=0`
- 支持通过环境变量覆盖：
  - `SFT_SUBSET_MAX_SAMPLES`
  - `SFT_SUBSET_EPOCHS`
  - 以及 `SFT_SUBSET_RATIO / SFT_SUBSET_SEED / SFT_SUBSET_MODE`

### 3.4 监控脚本改造成 manifest 驱动

[scripts/monitor_full_sft_memory.sh](../scripts/monitor_full_sft_memory.sh) 保留正式 full SFT 的默认 manifest 行为，但现在可以通过 `MANIFEST_PATH` 切换到其他 run。

当前监控输出包括：

- `screen` 状态
- Python 训练进程
- GPU 显存摘要
- WSL 内存
- swap
- `VmRSS / VmSwap / RssAnon / RssFile`
- checkpoint `mtime`
- 最近日志尾部
- 同时把关键指标持续写入 CSV

新增 [scripts/monitor_full_sft_subset_memory.sh](../scripts/monitor_full_sft_subset_memory.sh) 作为 subset 专用入口，默认指向：

- `experiments/logs/full-sft-subset-current-run.env`

### 3.5 推理入口检查结论

[eval_llm.py](../eval_llm.py) 当前本身按：

`out/{weight}_{hidden_size}.pth`

拼接普通权重路径，因此已经天然支持：

`--weight full_sft_subset --hidden_size 768`

本轮没有为此做额外代码修改。

## 4. 验收与验证

本轮只执行了静态检查与轻量只读验证，没有启动真实训练。

本轮已执行：

- `git status --short --branch`
- `./.venv/bin/python -m py_compile trainer/train_full_sft.py dataset/lm_dataset.py eval_llm.py tests/diagnose_sft_supervision.py`
- `bash -n scripts/start_full_sft_dense768_e2.sh`
- `bash -n scripts/start_full_sft_dense768_subset_e1.sh`
- `bash -n scripts/monitor_full_sft_memory.sh`
- `bash -n scripts/monitor_full_sft_subset_memory.sh`
- `./.venv/bin/python trainer/train_full_sft.py --help | grep -E "max_train_samples|train_sample_ratio|train_subset_seed|train_subset_mode"`
- 轻量 subset 数据加载验证：只加载 tokenizer 与 `SFTDataset`，构造 100 条 `Subset`，检查长度、前几个样本取值、`input_ids.shape == labels.shape`、至少存在一个 `valid_label_tokens > 0` 的样本
- `git diff --check`

## 5. 使用边界

- subset run 只能写成学习链路 / 短跑验证 / smoke-style run。
- subset 输出不能冒充正式 `full_sft_768.pth`。
- subset 推理只能用于学习性验证，不能把单次回答写成模型能力结论。
- 如果后续要做正式 `v0.0.3` full SFT 验收，仍需要回到正式脚本或明确全量配置。

## 6. 结论

本轮交付的是：

- subset 可控抽样能力
- subset 独立启动脚本
- subset 独立监控入口
- 学习链路计划文档
- 单独 fix-report v0.0.3 记录

本轮没有交付的是：

- 真实训练完成
- subset 权重产出
- subset 推理效果结论
- 正式 full SFT 完成结论
