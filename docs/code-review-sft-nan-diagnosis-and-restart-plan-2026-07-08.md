# Code Review：SFT NaN 诊断与干净重启准备（2026-07-08）

## 审查范围

本次审查聚焦第一次 `Dense 768` full SFT 中断事件及其对应修复，范围包括：

- 数据语义：[lm_dataset.py](../dataset/lm_dataset.py#L58)
- 训练循环：[train_full_sft.py](../trainer/train_full_sft.py#L24)
- 只读诊断脚本：[diagnose_sft_supervision.py](../scripts/diagnose_sft_supervision.py#L19)
- 中断日志：[full-sft-dense-768-e2-20260708-070010.log](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L67)
- partial 工件归档目录：`../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/`

本报告保持 `minimind-code-reviewer` 风格：`findings first`，先写问题、证据、影响、建议修复和最小验证，再写已验证事实与剩余风险。

## 审查结论

本次 NaN 事件的主要机制已经得到支持性证据：旧版 `SFTDataset` 采用“先前缀截断、再找 assistant supervision”的语义，会在长对话样本上产生全 `-100 labels`，而 `CrossEntropy(ignore_index=-100, reduction=mean)` 在这种条件下可直接得到 `nan`。与此同时，旧版训练循环没有在 forward 之前拦截 zero-supervision batch，也没有保证 checkpoint 与有效梯度累积边界对齐，因此 `resume` 语义不完整。

## Findings

### 严重 1：旧版前缀截断语义会把末尾 assistant 回复整体裁掉，直接制造 zero-supervision batch

- 位置：
  - [lm_dataset.py](../dataset/lm_dataset.py#L88)
  - [diagnose_sft_supervision.py](../scripts/diagnose_sft_supervision.py#L79)
  - [full-sft-dense-768-e2-20260708-070010.log](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L67)
- 证据：
  - 本轮只读诊断按 `setup_seed(42)`、`torch.randperm(len(train_ds))`、`max_seq_len=384`、前 `10000` 个 micro-step 复现样本顺序。
  - 旧行为下前 `10000` step 中共有 `136` 个样本 `labels` 全为 `-100`；你给出的 NaN step `980 / 1060 / 1880 / 2960 / 4800 / 7580 / 8220 / 8840` 全部命中这些 zero-supervision step。
  - 重点样本的原始 token 长度分别为 `636 / 764 / 662 / 644 / 701 / 603 / 569 / 608`，前缀窗口内监督 token 数都为 `0`，而修复后的尾部窗口可恢复 `122 / 166 / 137 / 119 / 190 / 139 / 65 / 187` 个监督 token。
- 影响：
  - 旧逻辑优先保留长 prompt 的开头，而不是实际参与监督的末尾 assistant 回复。
  - 在 full SFT 中，这会稳定制造 zero-supervision batch，导致 loss 语义失真，进一步触发 NaN。
- 建议修复：
  - 保持当前修复后的语义：先在完整 token 序列上生成完整 `labels`，再对 `input_ids` 与 `labels` 同步裁剪，并在超长时优先保留最后 `max_seq_len` 的尾部窗口。
  - 不改 chat template，不改 assistant label 定义，不把 user/system/pad token 误计入 loss。
- 最小验证：
  - 运行 `./.venv/bin/python scripts/diagnose_sft_supervision.py --max_seq_len 384 --steps 10000 --seed 42 --focus_steps 980 1060 1880 2960 4800 7580 8220 8840`
  - 期望结果：`zero_supervision_steps_original=136`，`zero_supervision_steps_repaired=0`。

### 严重 2：旧版训练循环允许 zero-supervision batch 进入 forward/loss，NaN 会直接污染训练过程

- 位置：
  - [train_full_sft.py](../trainer/train_full_sft.py#L94)
  - [model_minimind.py](../model/model_minimind.py#L249)
  - [diagnose_sft_supervision.py](../scripts/diagnose_sft_supervision.py#L182)
- 证据：
  - 当前模型 loss 仍使用 `F.cross_entropy(..., ignore_index=-100)`，[model_minimind.py](../model/model_minimind.py#L266) 没有额外保护。
  - CPU toy 验证中，全部 `labels=-100` 时 `loss=nan` 且 `isfinite=False`；混合有效标签时 `loss` 为有限值。
  - 中断日志已多次记录 `loss: nan, logits_loss: nan`，例如 [log#L67](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L67)、[log#L112](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L112)、[log#L397](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L397)、[log#L460](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L460)。
- 影响：
  - 只要 zero-supervision batch 进入 forward/loss，NaN 就可能沿着 backward、梯度累积和 checkpoint 保存继续污染后续训练。
  - 即使单个 batch 的 NaN 是数据诱发的，训练侧若不显式 fail-fast，就无法准确保留现场。
- 建议修复：
  - 保持当前训练侧防线：
    - `valid_label_tokens == 0` 时直接 `skipped_no_supervision`，不进入 `model(...)`、`backward`、`optimizer.step`
    - `logits_loss`、`aux_loss`、总 `loss` 任一非有限时立即 `zero_grad` 并抛错
    - 真正 `optimizer.step` 前检查 `grad_norm` 是否 finite
- 最小验证：
  - 运行 `./.venv/bin/python -m py_compile trainer/train_full_sft.py dataset/lm_dataset.py`
  - 审核 [train_full_sft.py](../trainer/train_full_sft.py#L110) 到 [train_full_sft.py](../trainer/train_full_sft.py#L152) 的 skip / fail-fast 分支，确认 zero-supervision batch 不再进入 forward。

### 严重 3：旧版 checkpoint 保存点与有效梯度累积边界不一致，导致 resume 语义不完整

- 位置：
  - [train_full_sft.py](../trainer/train_full_sft.py#L173)
  - [full-sft-dense-768-e2-20260708-070010.log](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L518)
  - [README.md](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md)
- 证据：
  - 首次 run 的配置是 `accumulation_steps=6`、`save_interval=5000`，两者不对齐。
  - 日志显示训练在 `step=10000` 左右被用户主动 `SIGINT`，且堆栈停在 `torch.save(resume_data, resume_tmp)`，[log#L519](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L519) 到 [log#L529](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log#L529) 可见。
  - 归档前记录到：
    - 普通权重 `out/full_sft_768.pth`：`mtime=2026-07-08 07:06:42`
    - 普通 checkpoint `checkpoints/full_sft_768.pth`：`mtime=2026-07-08 07:06:43`
    - `resume checkpoint`：`mtime=2026-07-08 07:03:37`
    - `resume checkpoint .tmp`：`mtime=2026-07-08 07:06:47`
- 影响：
  - 旧版训练代码可能在“还有未落地累积梯度”的 micro-step 上保存 checkpoint，但 `resume checkpoint` 又不保存 `.grad`。
  - 恢复时若直接跳过这些 micro-step，就会造成“步数看似推进，梯度其实没完整落地”的语义断层。
  - 这就是为什么当前 partial full SFT 不能作为可信 `resume` 起点。
- 建议修复：
  - 保持当前延后保存语义：`save_interval` 只代表保存请求阈值，真正写 checkpoint 必须等到下一次有效 optimizer update 完成。
  - 后续新的 full SFT 必须从 `--from_weight pretrain --from_resume 0` 启动，不能使用当前 partial 工件恢复。
- 最小验证：
  - 审核 [train_full_sft.py](../trainer/train_full_sft.py#L150) 到 [train_full_sft.py](../trainer/train_full_sft.py#L208) 的 `pending_save_step`、`effective_backward_count` 和 `end_of_epoch_deferred_periodic` 逻辑。
  - 运行 `bash -n scripts/start_full_sft_dense768_e2.sh`，并确认脚本说明中已明确“实际 checkpoint step 可能晚于 `save_interval` 阈值”。

## 已验证事实

- 第一次 Dense 768 full SFT 于 `2026-07-08 07:00` 左右启动，并在 `step=10000` 附近被用户主动 `SIGINT` 中断。
- 本轮没有启动新的训练、没有推理、没有加载模型权重；数据诊断只加载了本地 tokenizer 和真实 JSONL。
- partial 工件已移出官方启动路径，归档目录为 `../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/`。
- 当前原始 full SFT 路径已清空：
  - `out/full_sft_768.pth`
  - `checkpoints/full_sft_768.pth`
  - `checkpoints/full_sft_768_resume.pth`
  - `checkpoints/full_sft_768.pth.tmp`
  - `checkpoints/full_sft_768_resume.pth.tmp`

## 剩余风险

- 本报告支持“全 `-100 labels` 是本次 NaN 的主要机制”，但不能单独证明所有 NaN 都只由这一机制导致。
- 本轮只做了静态验证和无模型数据审计，没有做新的 full SFT，因此 GPU 路径、长时训练稳定性和真实恢复闭环仍未重新验证。
- 归档目录内保留了关键 partial 工件，但 `out/full_sft_768.pth` 当前只保留了归档前 `size / mtime / sha256` 证据，没有保留原始字节副本；这一点已经在归档 README 中如实记录。

## 最小验证记录

- `./.venv/bin/python scripts/diagnose_sft_supervision.py --max_seq_len 384 --steps 10000 --seed 42 --focus_steps 980 1060 1880 2960 4800 7580 8220 8840`
- `./.venv/bin/python - <<'PY' ... SFTDataset ... zero_supervision_steps_after_fix ... PY`
- `./.venv/bin/python -m py_compile trainer/train_full_sft.py dataset/lm_dataset.py eval_llm.py scripts/diagnose_sft_supervision.py`
- `./.venv/bin/python trainer/train_full_sft.py --help`
- `./.venv/bin/python eval_llm.py --help`
- `bash -n scripts/start_full_sft_dense768_e2.sh`
- `bash -n scripts/monitor_full_sft_memory.sh`
- `git diff --check`

## 结论

这份报告对应的是一次 `code-review` 结论，而不是“实验完成报告”。

- 可以确认：本次 NaN 主要机制已有数据证据支持，数据集与训练循环的关键语义修复已落地，partial 工件也已隔离归档。
- 不能确认：新的 full SFT 已完成、模型已可推理、resume 已恢复可信、`v0.0.3` 已正式闭环。
- 后续若继续更新本事件记录，应优先续写 [fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md](fix-report-v0.0.3-dense-768-full-sft-interruption-and-restart-2026-07-08.md)。
