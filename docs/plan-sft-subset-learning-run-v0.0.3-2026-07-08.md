# Dense 768 SFT Subset 学习链路计划（v0.0.3，2026-07-08）

## 1. 文档边界

这不是正式 `Dense 768 full SFT` 完成报告，也不是模型能力结论。

本计划只服务本机 `RTX 5060 Laptop 8GB + WSL` 环境下的学习链路 / 短跑验证 / smoke-style run，目标是用较短时间跑通：

`pretrain_768.pth -> SFT subset 训练 -> checkpoint / 日志 / 监控 -> 后续推理验证`

## 2. 为什么要做 subset run

- 当前目标不是尽快做完正式全量 SFT，而是先在本机上稳定复现训练链路。
- subset run 可以降低训练时间、显存占用、WSL 内存与 swap 压力，也更适合学习日志、checkpoint、监控和推理入口之间的关系。
- subset run 仍然保持 `Dense 768` 结构，不通过缩小 `hidden_size`、`num_hidden_layers`、`use_moe` 或 `dtype` 来规避真实训练链路。

## 3. 这次 subset run 的硬边界

- 使用 `save_weight=full_sft_subset`。
- 使用 `from_weight=pretrain`。
- 使用 `from_resume=0`。
- subset 输出不能冒充正式 `out/full_sft_768.pth`。
- subset checkpoint 只能写到 `checkpoints/full_sft_subset_768*.pth`。
- 训练完成后可以用于学习性推理验证，但不能把单次回答写成模型能力结论。
- 如果后续要做正式 `v0.0.3` full SFT 验收，仍需要恢复正式脚本或明确全量配置。

## 4. 当前默认配置

subset 启动脚本是 [start_full_sft_dense768_subset_e1.sh](../scripts/start_full_sft_dense768_subset_e1.sh)。

默认参数：

- `epochs=1`
- `batch_size=1`
- `max_seq_len=384`
- `accumulation_steps=6`
- `num_workers=0`
- `learning_rate=1e-5`
- `grad_clip=1.0`
- `log_interval=20`
- `save_interval=10000`
- `dtype=bfloat16`
- `hidden_size=768`
- `num_hidden_layers=8`
- `use_moe=0`
- `from_weight=pretrain`
- `from_resume=0`
- `save_weight=full_sft_subset`
- `save_dir=../out`
- `max_train_samples=50000`
- `train_sample_ratio=1.0`
- `train_subset_seed=42`
- `train_subset_mode=random`

## 5. Windows PowerShell 启动方式

启动 subset run：

```powershell
wsl -d Ubuntu-24.04 -- bash /home/harry/projects/MiniMind/scripts/start_full_sft_dense768_subset_e1.sh
```

在第二个 PowerShell 窗口启动监控：

```powershell
wsl -d Ubuntu-24.04 -- bash /home/harry/projects/MiniMind/scripts/monitor_full_sft_subset_memory.sh
```

如果想把训练样本进一步缩小到 `20000`：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && SFT_SUBSET_MAX_SAMPLES=20000 bash scripts/start_full_sft_dense768_subset_e1.sh"
```

如果还想改 epoch，可以同样覆盖：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && SFT_SUBSET_EPOCHS=1 SFT_SUBSET_MAX_SAMPLES=20000 bash scripts/start_full_sft_dense768_subset_e1.sh"
```

## 6. 训练完成后的使用边界

- 训练完成后可通过 [eval_llm.py](../eval_llm.py) 使用 `--weight full_sft_subset --hidden_size 768` 做学习性推理验证。
- 这只能说明“subset 学习链路工件可被加载并参与推理测试”，不能说明正式 full SFT 已完成，更不能外推成稳定模型能力结论。

## 7. 语义保护点

subset run 仍然沿用当前已修复的 SFT 语义，不单独开旁路：

- `SFTDataset` 先生成完整 `labels`，再裁到尾部窗口。
- `valid_label_tokens == 0` 直接跳过。
- 非有限 `loss / grad norm` 立即 fail-fast。
- checkpoint 只在有效 optimizer update 边界落盘。

## 8. 输出位置

- 推理权重：`out/full_sft_subset_768.pth`
- checkpoint：`checkpoints/full_sft_subset_768.pth`
- resume checkpoint：`checkpoints/full_sft_subset_768_resume.pth`
- 当前 run manifest：`experiments/logs/full-sft-subset-current-run.env`
- 日志：`experiments/logs/full-sft-subset-dense-768-e1-<run_id>.log`
- 监控 CSV：`experiments/logs/full-sft-subset-dense-768-e1-<run_id>-memory.csv`

## 9. 当前未完成事项

- 本文档不代表真实训练已经启动或完成。
- 本轮只覆盖代码改造、脚本准备和静态验证。
- 真正训练、checkpoint 产出、推理文本效果与长时稳定性，仍需用户后续手动启动后再验证。
