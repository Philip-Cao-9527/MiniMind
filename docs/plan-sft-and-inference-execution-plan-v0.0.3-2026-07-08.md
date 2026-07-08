# Dense 768 Full SFT 与训练后推理验证执行计划（v0.0.3，2026-07-08）

> 2026-07-08 更新：原计划对应的第一次 Dense 768 full SFT 已于 2026-07-08 07:00 左右启动，并在 `step=10000` 附近出现多次 `loss: nan` 后由用户主动 `SIGINT` 中断。该次 partial full SFT 不可用于验收、不可用于推理、不可用于 resume。后续执行对象改为一次新的、从 `pretrain` 权重开始的干净 full SFT。正式 `v0.0.3` fix-report 仍只能在新的 full SFT 完成且推理 / cache / EOS / history 验证完成后生成。

## 1. 目标、范围与明确不做的事情

本计划现阶段只服务一次新的 `Dense 768` full SFT 干净重启与训练后推理验证的人工执行阶段。当前阶段只完成：

- 锁定 full SFT 参数方案
- 完成 NaN 数据根因诊断
- 修复 `SFTDataset` 截断语义
- 修复 full SFT 训练侧的无监督 batch、非有限数值与 accumulation / checkpoint 对齐语义
- 归档第一次中断的 partial full SFT 工件
- 修订执行前操作手册
- 增补 `eval_llm.py` 的最小本地验证能力
- 新增 WSL 启动与监控脚本，供未来从 Windows PowerShell 手动调用

本计划明确不做以下事情：

- 不启动 `trainer/train_full_sft.py`
- 不启动 `eval_llm.py` 的真实推理
- 不加载模型权重，不创建新的 checkpoint、训练日志、SwanLab run 或 screen 会话
- 不生成 `docs/fix-report-v0.0.3-dense-768-full-sft-and-inference-validation-2026-07-08.md`
- 不提前修改 [README.md](../README.md) 的版本号

## 2. 证据来源、冲突优先级与当前代码状态

证据优先级按以下顺序执行：

1. 本轮只读核验到的本机代码、Git 状态、权重、checkpoint、日志、数据集和系统状态
2. [fix-report-v0.0.2-dense-768-pretrain-completion-2026-07-08.md](fix-report-v0.0.2-dense-768-pretrain-completion-2026-07-08.md)
3. [minimind-initial-pretrain-full-retrospective-updated.md](minimind-initial-pretrain-full-retrospective-updated.md)
4. [minimind-pretrain-resume-incident-20260708.md](minimind-pretrain-resume-incident-20260708.md)
5. [pretrain-sft-manual-runbook-2026-07-07.md](pretrain-sft-manual-runbook-2026-07-07.md)
6. `/home/harry/references/minimind`
7. 上游公开 README 或其他公开材料
8. `/home/harry/references/learn-minimind`

本计划同时引用以下项目规则或总览：

- [AGENTS.md](../AGENTS.md)
- [README.md](../README.md)

### 2.1 当前代码状态与上游差异边界

当前必须准确区分以下事实：

- 在本轮推理验证准备前，关键训练、数据集与推理文件曾与上游参考同名文件一致。
- 当前 `dataset/lm_dataset.py` 已修复为“先在完整 token 序列上生成 labels，再对 `input_ids` 与 `labels` 同步裁剪到优先保留尾部 assistant 回复的窗口”。
- 当前 `trainer/train_full_sft.py` 已修复：
  - `valid_label_tokens == 0` 时不再进入 forward / backward / optimizer.step
  - 非有限 `logits_loss`、`aux_loss`、总 `loss` 与 `grad_norm` 会立即失败并保留现场
  - checkpoint 只在无未落地累积梯度的边界写入，`save_interval` 只代表保存请求阈值
- 当前 `eval_llm.py` 已在上游实现基础上做了最小本地增强。

当前 `eval_llm.py` 的本地差异只服务后续人工验证，目的如下：

1. `--prompt`
   - 支持固定 prompt 的非交互验证
2. `--seed`
   - 固定随机种子，方便保留可复现条件
3. `--top_k`
   - 暴露已有的 `generate(..., top_k=...)`
4. `--use_cache`
   - 暴露已有的 `generate(..., use_cache=...)`
5. `--do_sample`
   - 暴露已有的 `generate(..., do_sample=...)`
6. `--stream`
   - 允许关闭控制台流式输出，便于对比 cache on/off 结果
7. `--output_file`
   - 把每个 prompt 的生成结果写成 JSONL，供后续比对
8. history 参数校验
   - 显式拒绝负数和奇数 `historys`
9. 推理调用外层显式 `torch.inference_mode()`
   - 不改变生成语义，只避免无谓 autograd 状态
10. Meta 输出
   - 输出 `model_path`、`seed`、`use_cache`、`do_sample`、`ended_with_eos`、`elapsed_seconds`、`tokens_per_second` 等证据字段
11. 项目路径锚定
   - `eval_llm.py` 现在用脚本自身目录作为 `PROJECT_ROOT`
   - 相对 `save_dir` 会相对项目根目录解析
   - `--load_from model` 会稳定解析到项目内 `model/`
   - 其他本地相对目录仅在项目根下真实存在时才改写为本地绝对路径，不会误改 Hugging Face 模型 ID

上游归属说明：

- 本地 `eval_llm.py` 仍然沿用上游 `MiniMindForCausalLM.generate()` 主语义
- 没有重写推理框架
- 没有改 `model/model_minimind.py` 中的生成算法

## 3. Dense 768 预训练前置条件

本计划建立在以下本机已验证前提上：

- 最终预训练权重：`/home/harry/projects/MiniMind/out/pretrain_768.pth`
- 最终 resume checkpoint：`/home/harry/projects/MiniMind/checkpoints/pretrain_768_resume.pth`
- 预训练最终到达：`Epoch:[1/1](635119/635119)`
- 当前没有运行中的 `train_pretrain.py` 或 `train_full_sft.py` writer
- 第一次中断的 partial full SFT 工件已归档到：
  - `../experiments/interrupted/full-sft-dense768-e2-20260708-070010-nan-and-sigint/`
- 新一轮 full SFT 的原始启动路径已清空：
  - `out/full_sft_768.pth`
  - `checkpoints/full_sft_768.pth`
  - `checkpoints/full_sft_768_resume.pth`
  - `checkpoints/full_sft_768.pth.tmp`
  - `checkpoints/full_sft_768_resume.pth.tmp`
- 当前 `screen -ls` 返回 `No Sockets found`

full SFT 首次启动必须使用：

- `--from_weight pretrain`
- `--from_resume 0`

原因：

- `--from_weight pretrain` 会按当前 `trainer_utils.init_model()` 的真实实现加载 `../out/pretrain_768.pth`
- `--from_resume 0` 表示本轮 full SFT 首次启动不从已有 full SFT resume 状态恢复

## 4. Full SFT 真实调用链与固定路径

当前本地代码的真实主链路如下：

1. `trainer/train_full_sft.py` 解析 CLI 参数
2. `trainer_utils.init_model()` 构建 `MiniMindConfig`、初始化 tokenizer 和 `MiniMindForCausalLM`
3. `init_model(..., from_weight='pretrain')` 加载 `../out/pretrain_768.pth`
4. `dataset/lm_dataset.py` 中的 `SFTDataset` 读取 `../dataset/sft_t2t_mini.jsonl`
5. `SFTDataset.create_chat_prompt()` 调用 tokenizer 的 `apply_chat_template(...)`
6. `SFTDataset.generate_labels()` 只让 assistant 片段及其结尾 `eos` 参与监督，其他位置写 `-100`
7. `train_full_sft.py` 用 `DataLoader(..., pin_memory=True)` 组织 batch
8. 训练循环按 micro-step 计算学习率、前向、loss、反向、梯度累积、裁剪和保存
9. 普通权重保存到 `../out/full_sft_768.pth`
10. 普通 checkpoint 保存到 `../checkpoints/full_sft_768.pth`
11. resume checkpoint 保存到 `../checkpoints/full_sft_768_resume.pth`

必须明确：

- `checkpoint_dir=../checkpoints` 不是 CLI 参数
- 它是当前代码内部固定路径
- 对应 `.tmp` 文件是：
  - `../checkpoints/full_sft_768.pth.tmp`
  - `../checkpoints/full_sft_768_resume.pth.tmp`

## 5. 已锁定的 full SFT 参数方案

本轮已锁定的人工执行参数：

- `epochs=2`
- `batch_size=1`
- `max_seq_len=384`
- `accumulation_steps=6`
- `num_workers=0`
- `dtype=bfloat16`
- `learning_rate=1e-5`
- `grad_clip=1.0`
- `log_interval=20`
- `save_interval=5000`
- `from_weight=pretrain`
- `from_resume=0`
- `save_weight=full_sft`
- `save_dir=../out`

参数依据必须分开理解：

- 固定前向长度：`SFTDataset` 会把 `input_ids` pad 到 `max_seq_len`，因此当前配置的实际前向长度按固定 `384` 计算。
- 有限样本平均有效内容长度：`356.68` 只描述固定随机种子、4096 条样本统计下的平均有效内容长度，不能当作真实显存长度、真实前向长度或真实计算 token。
- 有限样本平均监督 token：`281.89` 只描述平均参与 loss 的 assistant token 量级。

当前有限样本统计结果：

- 样本总数：`905718`
- 有限样本数：`4096`
- 固定随机种子：`42`
- 平均原始长度：`494.24`
- P50：`493`
- P90：`702`
- P95：`771`
- P99：`909`
- `max_seq_len=384` 时的截断样本比例：`77.12%`
- `max_seq_len=384` 时的平均监督 token：`281.89`

与资源边界相关的工程解释：

- `num_workers=0` 只是降低 worker 常驻内存压力的工程选择，不能写成“已解决历史 host RSS / swap 问题”。
- `save_interval=5000` 只可写成“保存请求阈值”。在 `accumulation_steps=6` 下，checkpoint 的实际保存 step 可能略晚于 `5000` 的整数倍，因为必须等待到下一次有效 optimizer update 边界。
- 真实耗时不得写成固定值，尤其不能引用或暗示“3 小时”。两轮 `384` 的静态计算规模明显大于刚完成的 `128` 长度预训练，真实耗时只能在运行后记录。

## 6. 学习率调度、micro-step 与 update 的边界

当前 `trainer/train_full_sft.py` 中的学习率调度按 micro-step 推进，而不是按 optimizer update 推进。

准确含义：

- `get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)` 中的 `step` 是 dataloader micro-step
- `accumulation_steps=6` 会改变 optimizer update 数
- `accumulation_steps=6` 不会把学习率曲线重标到 update 维度

当前静态估算：

- `batch_size=1` 时每个 epoch 的 micro-step 数：`905718`
- `accumulation_steps=6` 时每个 epoch 的预计 optimizer update 数：`150953`
- `save_interval=5000` 时每个 epoch 的预计保存次数：`182`
- 2 epoch 的预计保存次数：`364`
- 最大恢复损失窗口：`4999` micro-step

当前必须补充理解：

- `save_interval` 是保存请求阈值，不是严格 checkpoint micro-step。
- 如果阈值命中时仍存在未落地的有效累积梯度，训练代码会把保存请求延后到下一次真正完成的 optimizer update 后再写 checkpoint。
- 无监督 batch 被跳过时，不参与有效梯度累积计数，也不会造成“已跳过 micro-step 但未落地梯度却被 resume 视为已完成”的状态错位。

## 7. WSL 官方脚本与 writer lock 覆盖范围

本轮新增两个脚本：

- `scripts/start_full_sft_dense768_e2.sh`
- `scripts/monitor_full_sft_memory.sh`

### 7.1 `start_full_sft_dense768_e2.sh`

用途：

- 作为后续首次 full SFT 的唯一官方启动入口

脚本会做的事情：

- 自动定位项目根目录，不依赖调用目录
- 检查：
  - `out/pretrain_768.pth` 存在
  - `dataset/sft_t2t_mini.jsonl` 存在
  - `/.venv/bin/python` 存在
- 拒绝在以下任一路径已存在时启动：
  - `out/full_sft_768.pth`
  - `checkpoints/full_sft_768.pth`
  - `checkpoints/full_sft_768_resume.pth`
  - `checkpoints/full_sft_768.pth.tmp`
  - `checkpoints/full_sft_768_resume.pth.tmp`
- 拒绝在已有 `train_full_sft.py` Python writer 时启动
- 使用时间戳 run id 生成独立：
  - screen session
  - 训练日志路径
  - 推荐监控 CSV 路径
- 生成固定 manifest：
  - `/home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env`

writer lock 真实覆盖范围：

- 该官方启动脚本会在 detached runner shell 中通过 advisory `flock -n` 持有：
  - `checkpoints/full_sft_768.writer.lock`
- 该锁会覆盖整个 `train_full_sft.py` + `tee` 生命周期
- 如果 lock 已被占用，脚本会快速失败，不等待
- 直接绕开该官方脚本、手工执行 `python train_full_sft.py ...` 的命令仍可绕过 lock

### 7.2 `monitor_full_sft_memory.sh`

用途：

- 只读监控一个已经启动的 `train_full_sft.py` Python 进程

脚本边界：

- 不启动训练
- 不停止训练
- 不修改 checkpoint、日志、模型或数据
- 仅当精确找到一个 `train_full_sft.py` Python 进程时才开始
- 0 个或多于 1 个候选时明确报错并退出
- 默认采样间隔 `30` 秒
- CSV 字段与 pretrain 监控保持可比：
  - `timestamp`
  - `pid`
  - `mem_available_kb`
  - `swap_used_kb`
  - `vmrss_kb`
  - `vmswap_kb`
  - `rssanon_kb`
  - `rssfile_kb`
  - `checkpoint_mtime_epoch`

## 8. 在 Windows PowerShell 执行：启动前只读核验

后续真实执行前，先在 Windows PowerShell 运行：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && git status --short --branch"
wsl -d Ubuntu-24.04 -- screen -ls
wsl -d Ubuntu-24.04 -- bash -lc "ps -eo pid,ppid,tty,etime,cmd | grep -E '[t]rain_(pretrain|full_sft)[.]py' || true"
wsl -d Ubuntu-24.04 -- stat -c "%n | size=%s bytes | mtime=%y" /home/harry/projects/MiniMind/out/pretrain_768.pth /home/harry/projects/MiniMind/checkpoints/pretrain_768_resume.pth
wsl -d Ubuntu-24.04 -- du -h /home/harry/projects/MiniMind/dataset/sft_t2t_mini.jsonl
wsl -d Ubuntu-24.04 -- wc -l /home/harry/projects/MiniMind/dataset/sft_t2t_mini.jsonl
wsl -d Ubuntu-24.04 -- free -h
wsl -d Ubuntu-24.04 -- nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
wsl -d Ubuntu-24.04 -- bash -lc "find /home/harry/projects/MiniMind/out /home/harry/projects/MiniMind/checkpoints /home/harry/projects/MiniMind/experiments/logs -maxdepth 2 -type f \( -name '*full_sft*' -o -name '*sft*' \) -printf '%p | %s bytes | %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort || true"
```

## 9. 在 Windows PowerShell 执行：启动 full SFT

后续首次 full SFT 只通过官方启动脚本启动：

```powershell
wsl -d Ubuntu-24.04 -- bash /home/harry/projects/MiniMind/scripts/start_full_sft_dense768_e2.sh
```

该脚本内部实际调用的训练参数固定为：

- `--epochs 2`
- `--batch_size 1`
- `--max_seq_len 384`
- `--accumulation_steps 6`
- `--num_workers 0`
- `--learning_rate 1e-5`
- `--grad_clip 1.0`
- `--log_interval 20`
- `--save_interval 5000`
- `--use_wandb`
- `--dtype bfloat16`
- `--hidden_size 768`
- `--num_hidden_layers 8`
- `--use_moe 0`
- `--from_weight pretrain`
- `--from_resume 0`

必须明确：

- 后续启动是一次新的干净 full SFT，不是继续第一次已中断 run。
- 绝不能从第一次 partial full SFT 的普通权重或 resume checkpoint 恢复。

## 10. 在 Windows PowerShell 执行：核验 screen、PID、日志与 manifest

启动脚本成功后，先读取 manifest：

```powershell
wsl -d Ubuntu-24.04 -- cat /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env
```

核验当前训练 screen 与唯一 Python PID：

```powershell
wsl -d Ubuntu-24.04 -- screen -ls
wsl -d Ubuntu-24.04 -- bash -lc "ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_full_sft.py'"
```

核验 manifest 中记录的当前 screen、日志和推荐监控 CSV 路径：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'grep -E "^(SCREEN_SESSION|LOG_PATH|MONITOR_CSV_PATH)=" /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env'
```

## 11. 在 Windows PowerShell 执行：启动内存监控

在确认训练 screen 与唯一 Python PID 后，再手动启动监控：

```powershell
wsl -d Ubuntu-24.04 -- bash /home/harry/projects/MiniMind/scripts/monitor_full_sft_memory.sh
```

说明：

- 监控脚本不会自动随训练启动
- 是否启动监控仍由用户显式控制
- 该命令会持续占用当前 PowerShell 窗口
- 请在第二个 Windows PowerShell 窗口中启动
- 监控窗口必须保持打开
- 关闭该 PowerShell 只会停止监控，不会停止训练 screen

## 12. 在 Windows PowerShell 执行：查看日志、CSV 与 artifact mtime

查看最新日志尾部：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'source /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env && tail -n 80 "$LOG_PATH"'
```

查看最新监控 CSV 一行：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'source /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env && tail -n 1 "$MONITOR_CSV_PATH"'
```

格式化读取最新内存样本：

```powershell
$line = wsl -d Ubuntu-24.04 -- bash -lc 'source /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env && tail -n 1 "$MONITOR_CSV_PATH"'
$c = $line -split ','

"latest={0} | VmRSS={1:N1} MiB | VmSwap={2:N1} MiB | system_swap={3:N1} MiB | MemAvailable={4:N2} GiB | RssAnon={5:N1} MiB | RssFile={6:N1} MiB" -f `
  $c[0], ([double]$c[4] / 1024), ([double]$c[5] / 1024), ([double]$c[3] / 1024), ([double]$c[2] / 1048576), ([double]$c[6] / 1024), ([double]$c[7] / 1024)
```

查看权重与 resume checkpoint 的 `mtime`：

```powershell
wsl -d Ubuntu-24.04 -- stat -c "%n | size=%s bytes | mtime=%y" /home/harry/projects/MiniMind/out/full_sft_768.pth /home/harry/projects/MiniMind/checkpoints/full_sft_768_resume.pth
```

监控时重点看：

- 当前训练 screen 是否仍在
- `train_full_sft.py` 是否仍是唯一 Python writer
- CSV 中的 `VmRSS`、`VmSwap`、系统 `swap_used_kb`、`MemAvailable`
- 权重与 resume checkpoint 的 `mtime` 是否推进
- 当前 full SFT 监控 CSV 字段顺序与 pretrain 监控相同：
  - 第 1 列：`timestamp`
  - 第 2 列：`pid`
  - 第 3 列：`mem_available_kb`
  - 第 4 列：`swap_used_kb`
  - 第 5 列：`vmrss_kb`
  - 第 6 列：`vmswap_kb`
  - 第 7 列：`rssanon_kb`
  - 第 8 列：`rssfile_kb`
  - 第 9 列：`checkpoint_mtime_epoch`
- PowerShell 数组从 `0` 开始，因此：
  - 第 3 列 `MemAvailable` 对应 `$c[2]`
  - 第 4 列 `swap_used_kb` 对应 `$c[3]`
  - 第 5 列 `VmRSS` 对应 `$c[4]`
  - 第 6 列 `VmSwap` 对应 `$c[5]`

## 13. 在 Windows PowerShell 执行：SFT 结束后的验收

训练结束后，至少执行：

```powershell
wsl -d Ubuntu-24.04 -- screen -ls
wsl -d Ubuntu-24.04 -- bash -lc "ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_full_sft.py' || true"
wsl -d Ubuntu-24.04 -- bash -lc 'source /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env && tail -n 120 "$LOG_PATH"'
wsl -d Ubuntu-24.04 -- bash -lc 'source /home/harry/projects/MiniMind/experiments/logs/full-sft-current-run.env && rg -n "Traceback|RuntimeError|CUDA out of memory|OutOfMemory|KeyboardInterrupt|FileNotFoundError|Error:" "$LOG_PATH" || true'
wsl -d Ubuntu-24.04 -- stat -c "%n | size=%s bytes | mtime=%y" /home/harry/projects/MiniMind/out/full_sft_768.pth /home/harry/projects/MiniMind/checkpoints/full_sft_768_resume.pth
```

验收目标：

- 原计划中的第一次 full SFT 已中断，不可用于验收
- 后续新的 full SFT 需要自然结束或按计划受控停止
- 最终权重存在
- 最终 resume checkpoint 存在
- 日志中能确认 2 epoch 的最终进度
- 不把未验证内容扩写成“能力已证明”

## 14. `eval_llm.py` 的未来推理验证能力边界

当前 `eval_llm.py --help` 与 argparse 已确认支持：

- `--load_from model`
- `--prompt`
- `--seed`
- `--top_k`
- `--use_cache`
- `--do_sample`
- `--stream`
- `--output_file`

当前 `eval_llm.py` 的 JSONL 证据字段至少包含：

- `timestamp`
- `prompt_index`
- `prompt`
- `response`
- `model_path`
- `weight`
- `device`
- `seed`
- `use_cache`
- `do_sample`
- `temperature`
- `top_p`
- `top_k`
- `max_new_tokens`
- `historys`
- `open_thinking`
- `input_tokens`
- `generated_tokens`
- `generated_token_ids`
- `eos_token_id`
- `ended_with_eos`
- `elapsed_seconds`
- `tokens_per_second`

参数语义边界：

- `eval_llm.py` 不再依赖外部当前工作目录来解析项目内 `model/` 与 `out/`；文档命令仍显式 `cd /home/harry/projects/MiniMind`，便于人工复核
- `do_sample=0` 时，`temperature`、`top_p`、`top_k` 仍会被记录，但不应写成影响贪心输出的随机因素
- 固定 seed 对 `do_sample=0` 的输出通常不是核心决定因素，但仍应记录
- cache off 的慢速路径是预期比较对象，不应直接认定为异常
- 后续 cache on/off 默认使用同一 prompt、同一 `max_new_tokens=128`、同一模型、同一 device、`do_sample=0`、`stream=0`
- 不得使用默认 `max_new_tokens=8192` 做 cache off 验证

## 15. 在 Windows PowerShell 执行：固定 prompt、cache on/off、EOS、采样、history 验证

未来推理输出文件必须分开保存，例如：

- `/home/harry/projects/MiniMind/experiments/inference/full-sft-fixed-cache-on-$RunId.jsonl`
- `/home/harry/projects/MiniMind/experiments/inference/full-sft-fixed-cache-off-$RunId.jsonl`

### 15.1 固定 prompt 基线

```powershell
$RunId = Get-Date -Format 'yyyyMMdd-HHmmss'
$OutputPath = "/home/harry/projects/MiniMind/experiments/inference/full-sft-fixed-baseline-$RunId.jsonl"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 0 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --historys 0 --open_thinking 0 --stream 0 --output_file '$OutputPath'"
```

### 15.2 cache on/off 对照

cache on / cache off 必须使用同一组 `$RunId`，再分别写入两个不同 JSONL：

```powershell
$RunId = Get-Date -Format 'yyyyMMdd-HHmmss'
$OutputPathOn = "/home/harry/projects/MiniMind/experiments/inference/full-sft-fixed-cache-on-$RunId.jsonl"
$OutputPathOff = "/home/harry/projects/MiniMind/experiments/inference/full-sft-fixed-cache-off-$RunId.jsonl"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 0 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --historys 0 --open_thinking 0 --stream 0 --output_file '$OutputPathOn'"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 0 --use_cache 0 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --historys 0 --open_thinking 0 --stream 0 --output_file '$OutputPathOff'"
```

对比目标：

- `response`
- `generated_token_ids`
- `generated_tokens`
- `ended_with_eos`
- `elapsed_seconds`
- `tokens_per_second`

对于 `do_sample=0` 的 cache on/off，对比重点是 token id 序列与最终文本；速度不同不是错误。

结论边界：

- 在 `do_sample=0`、固定 prompt、固定模型、固定设备、固定 `max_new_tokens=128` 下，应优先比较 `generated_token_ids`、`response`、`generated_tokens` 与 `ended_with_eos`
- 若 token id 完全一致，只能支持“该样例上 cache on/off 两条路径行为一致”
- 若不一致，不能立刻写成模型错误或 cache bug；应先保留 JSONL、运行参数、GPU / dtype 条件，再判断是否属于 BF16 / CUDA 数值路径差异、实现问题或状态处理问题
- cache off 更慢是预期现象，不作为异常

### 15.3 EOS 验证

观察：

- `ended_with_eos`
- `generated_tokens`
- 输出文本是否自然结束

### 15.4 采样验证

```powershell
$RunId = Get-Date -Format 'yyyyMMdd-HHmmss'
$OutputPathLow = "/home/harry/projects/MiniMind/experiments/inference/full-sft-sampling-low-$RunId.jsonl"
$OutputPathDefault = "/home/harry/projects/MiniMind/experiments/inference/full-sft-sampling-default-$RunId.jsonl"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 1 --use_cache 1 --top_k 20 --top_p 0.90 --temperature 0.70 --max_new_tokens 128 --stream 0 --output_file '$OutputPathLow'"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 1 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --stream 0 --output_file '$OutputPathDefault'"
```

### 15.5 history 验证

```powershell
$RunId = Get-Date -Format 'yyyyMMdd-HHmmss'
$OutputPath = "/home/harry/projects/MiniMind/experiments/inference/full-sft-history-2-$RunId.jsonl"
wsl -d Ubuntu-24.04 -- bash -lc "cd /home/harry/projects/MiniMind && ./.venv/bin/python ./eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '你好。' --prompt '请重复上一句问候并继续自我介绍。' --seed 42 --do_sample 0 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --historys 2 --stream 0 --output_file '$OutputPath'"
```

## 16. 生成 `v0.0.3` fix-report 前必须收集的真实证据

未来允许生成 `docs/fix-report-v0.0.3-dense-768-full-sft-and-inference-validation-2026-07-08.md` 的前提是以下证据齐全：

- 新的一次干净 full SFT 启动命令（不是第一次已中断 run）
- 真实 full SFT 启动命令
- 真实 full SFT 日志路径与尾部证据
- `out/full_sft_768.pth` 的 `stat`
- `checkpoints/full_sft_768_resume.pth` 的 `stat`
- 若需要恢复训练，恢复命令与恢复后日志证据
- 固定 prompt 推理命令
- cache on/off 对照结果
- EOS 是否触发的证据
- 采样参数对照结果
- history 验证结果
- 仍未验证边界

## 17. 未验证边界

- 当前没有真实 full SFT 显存峰值证据
- 当前没有真实 full SFT host RSS / swap 曲线证据
- 当前没有真实 2 epoch 耗时证据
- 当前没有真实 full SFT 训练质量与推理质量证据
- 当前没有真实 cache on/off、一致性、EOS、采样、history 验证结果
- 当前只有通过官方启动脚本的 advisory writer lock；绕开脚本仍可绕过
