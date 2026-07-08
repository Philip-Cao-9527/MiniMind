# Fix Report v0.0.3：Dense 768 Full SFT 中断归档与重启准备（2026-07-08）

## 1. 范围

本报告不是“Dense 768 full SFT 已完成”的成功报告，而是一次中断 run 的诊断、修复、归档与干净重启准备记录。

## 2. 事件摘要

- 第一次 Dense 768 full SFT 于 `2026-07-08 07:00` 左右启动。
- 训练过程中在 `980 / 1060 / 1880 / 2960 / 4800 / 7580 / 8220 / 8840` 出现 `loss: nan / logits_loss: nan`。
- 用户随后主动发送 `SIGINT`。
- 中断发生在 `torch.save(resume_data, resume_tmp)`。
- 因此本次 partial full SFT 不允许用于 resume、推理或验收。

## 3. 数据诊断结论

见 [code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md](code-review-sft-nan-diagnosis-and-restart-plan-2026-07-08.md)。

本轮只读诊断确认：

- 旧的前缀截断行为在前 `10000` micro-step 中产生了 `136` 个全 `-100 labels` 样本；
- 提供的全部 NaN step 都命中这些全 `-100` step；
- CPU toy `CrossEntropy` 验证表明全部 `labels=-100` 时 loss 为 `nan`；
- 修复后的尾部窗口行为在同一前 `10000` step 审计中不再产出全 `-100 labels`。

## 4. 代码修复

- [dataset/lm_dataset.py](../dataset/lm_dataset.py)
  - 改为先在完整 token 序列上生成 labels，再同步裁剪到尾部窗口。
- [trainer/train_full_sft.py](../trainer/train_full_sft.py)
  - 无监督 batch 直接跳过；
  - 非有限 loss / grad norm 立即失败；
  - checkpoint 仅在无未落地累积梯度的边界写入。
- [scripts/diagnose_sft_supervision.py](../scripts/diagnose_sft_supervision.py)
  - 新增只读诊断脚本，复现真实样本顺序并比较旧行为 / 修复后行为。
- [scripts/start_full_sft_dense768_e2.sh](../scripts/start_full_sft_dense768_e2.sh)
  - 明确 `save_interval=5000` 只是保存请求阈值，不是严格 checkpoint micro-step。

## 5. partial 工件归档

第一次中断 run 的 partial 工件已移出官方启动路径，归档目录为：

`../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/`

当前归档目录中保留：

- [README.md](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/README.md)
- [checkpoints/full_sft_768.pth](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/checkpoints/full_sft_768.pth)
- [checkpoints/full_sft_768_resume.pth](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/checkpoints/full_sft_768_resume.pth)
- [checkpoints/full_sft_768_resume.pth.tmp](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/checkpoints/full_sft_768_resume.pth.tmp)
- [full-sft-dense-768-e2-20260708-070010.log](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-dense-768-e2-20260708-070010.log)
- [full-sft-current-run.env](../../../backups/MiniMind/local-artifacts/interrupted-20260708-151025/full-sft-dense768-e2-20260708-070010-nan-and-sigint/experiments/logs/full-sft-current-run.env)

归档前已记录到 `out/full_sft_768.pth` 的存在证据：

- `size=137684407 bytes`
- `mtime=2026-07-08 07:06:42.290270478 +0800`
- `sha256=42c38ad97a90ad352bd07c607a6acecdaa36228b08030c8b92f8e2d36e9a553e`

## 6. 后续启动约束

后续新的 full SFT 必须使用：

- `--from_weight pretrain`
- `--from_resume 0`

且不得把本次 partial full SFT 写成：

- 已完成 full SFT
- 可用于推理的权重
- 模型能力达标证据
- 正式 `v0.0.3` 完成结论

## 7. 结论

当前阶段完成的是：

- NaN 主要机制诊断
- 数据与训练语义修复
- partial 工件归档
- 干净重启准备

当前阶段没有完成的是：

- 新的 full SFT
- 训练后推理
- cache / EOS / history 验证
- “Dense 768 Full SFT 已成功完成”的正式结论
