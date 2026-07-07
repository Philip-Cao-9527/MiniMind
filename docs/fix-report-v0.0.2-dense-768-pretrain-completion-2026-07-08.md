# MiniMind Dense 768 预训练完成复盘与验收报告（2026-07-08）

## 1. 问题背景、目标与文档范围

本文只复盘并验收 MiniMind `Dense 768` 这条真实预训练主线，目标是把“初始 checkpoint 并发 writer 事故”“后续 resume 长时内存 / swap 压力”“受控恢复后最终完成 1 个 epoch”三段事实合并成一份可复核报告。

本文不做以下事情：

- 不修改训练代码、不启动训练、不改 checkpoint、日志、数据或运行脚本。
- 不把 loss 数值、训练完成或权重可读取，扩展表述成“模型能力已证明优秀”或“loss 已完全收敛”。
- 不把尚未定位的内存增长来源写成“已确认内存泄漏并已修复”。

为避免历史表述相互污染，本文统一使用四类证据标签：

- `【本机已验证】`：本轮在当前仓库与本机文件系统中重新做过只读核验。
- `【代码/日志支持】`：当前代码、训练日志、监控 CSV 或高优先级复盘文档可以支持，但本轮无法从单一现场重新完整重放。
- `【工程判断】`：基于现有证据作出的保守工程判断。
- `【待验证】`：当前仍未闭环或不能下硬结论的部分。

## 2. 本次使用的证据来源与优先级

### 2.1 证据来源

1. 项目规则与总览：[AGENTS.md](../AGENTS.md)、[README.md](../README.md)
2. 高优先级事实来源：[minimind-initial-pretrain-full-retrospective-updated.md](minimind-initial-pretrain-full-retrospective-updated.md)、[minimind-pretrain-resume-incident-20260708.md](minimind-pretrain-resume-incident-20260708.md)
3. 较早阶段 runbook，仅作历史背景：[pretrain-sft-manual-runbook-2026-07-07.md](pretrain-sft-manual-runbook-2026-07-07.md)
4. 当前训练入口与保存实现：[train_pretrain.py](../trainer/train_pretrain.py)、[trainer_utils.py](../trainer/trainer_utils.py)
5. 本轮只读核验对象：
   - `/home/harry/projects/MiniMind/experiments/logs/pretrain-resume-20260708-010508.log`
   - `/home/harry/projects/MiniMind/out/pretrain_768.pth`
   - `/home/harry/projects/MiniMind/checkpoints/pretrain_768_resume.pth`
   - `/home/harry/projects/MiniMind/experiments/logs/pretrain-memory-resume-20260708.csv`

### 2.2 优先级与冲突处理原则

- `【本机已验证】` 高于历史文档表述。
- 两份 2026-07-08 新文档高于 2026-07-07 的 runbook。
- runbook 与两份新文档冲突时，以两份新文档为准；新文档与当前文件系统、日志或 checkpoint 元数据冲突时，以本轮只读核验为准。

本文据此采用的冲突处理结果如下：

- `【代码/日志支持】` 初始事故与后续 resume 高 swap 压力是两类独立问题，不能混写成同一根因。
- `【代码/日志支持】` 当前受控恢复阶段“只有一个 writer”是通过 `pgrep`、`ps`、`screen -ls` 的人工核验得到的运行状态，不是 `flock` 或其他机制级互斥锁。
- `【代码/日志支持】` runbook 中的 `--num_workers 2 --save_interval 200 --from_resume 0` 只对应更早的启动方案，不能覆盖最终完成这次 resume 的真实边界。

## 3. Dense 768 预训练最终验收结论

### 3.1 最终结论

`【本机已验证】` Dense 768 预训练在真实预训练数据上完成了 1 个 epoch。当前最终训练日志到达 `Epoch:[1/1](635119/635119)`，训练进程已自然退出；`screen -ls` 返回 `No Sockets found in /run/screen/S-harry.`，`pgrep -af train_pretrain.py` 为空；最终权重与完整 resume checkpoint 已落盘并可被 `torch.load` 正常读取。

`【本机已验证】` 训练日志末尾同时出现 SwanLab 完成提示：`Experiment MiniMind-Pretrain-Epoch-1-BatchSize-2-LearningRate-0.0005 has completed`。

### 3.2 本次验收不支持的扩展结论

- `【待验证】` 不支持“模型能力已证明优秀”。
- `【待验证】` 不支持“loss 已完全收敛”。
- `【待验证】` 不支持“MiniMind 内存泄漏已定位并修复”。
- `【待验证】` 不支持把 `save_interval=5000` 表述为“内存问题已经被彻底解决”。
- `【待验证】` 不支持“当前训练通过 flock 或原子锁机制强制保证唯一 writer”。

## 4. 最终训练配置与真实运行边界

### 4.1 训练类型与模型配置

- `【本机已验证】` 训练类型：Dense 768 真实预训练，`1 epoch`。
- `【本机已验证】` 模型结构：`hidden_size=768`、`num_hidden_layers=8`、`use_moe=0`。
- `【代码/日志支持】` 当前 [train_pretrain.py](../trainer/train_pretrain.py) 中，`MiniMindConfig` 由 `hidden_size`、`num_hidden_layers`、`use_moe` 直接构造；最终产物命名 `pretrain_768*.pth` 与 `Dense 768` 配置一致。

### 4.2 最终完成这次 resume 已确认显式传入的参数

下表是本文认可的“已确认显式传入参数”。它不是仅从最终日志单独重建，而是以高优先级文档中的 `PID 441` 现场 `ps` 记录为主，再与当前日志、SwanLab run 名、产物命名和训练代码实现交叉核对。

| 参数 | 结论 | 证据边界 |
| --- | --- | --- |
| `--epochs` | `1` | `【代码/日志支持】` 受控恢复 `ps` 现场；最终日志是 `Epoch:[1/1]` |
| `--batch_size` | `2` | `【代码/日志支持】` 受控恢复 `ps` 现场；SwanLab run 名含 `BatchSize-2` |
| `--max_seq_len` | `128` | `【代码/日志支持】` 受控恢复 `ps` 现场 |
| `--accumulation_steps` | `8` | `【代码/日志支持】` 受控恢复 `ps` 现场 |
| `--num_workers` | `0` | `【代码/日志支持】` 高优先级 resume 复盘文档中的高压现场与受控恢复 `ps` 现场 |
| `--log_interval` | `20` | `【代码/日志支持】` 受控恢复 `ps` 现场；最终日志按 20 step 打印 |
| `--save_interval` | `5000` | `【代码/日志支持】` 受控恢复 `ps` 现场；本文明确只把它写成工程调参尝试 |
| `--dtype` | `bfloat16` | `【代码/日志支持】` 受控恢复 `ps` 现场 |
| `--hidden_size` | `768` | `【本机已验证】` 产物命名、权重键数量、训练代码与文档一致 |
| `--num_hidden_layers` | `8` | `【代码/日志支持】` 受控恢复 `ps` 现场 |
| `--use_moe` | `0` | `【本机已验证】` 最终产物命名无 `_moe`；训练代码与文档一致 |
| `--save_weight` | `pretrain` | `【代码/日志支持】` 受控恢复 `ps` 现场；产物命名一致 |
| `--save_dir` | `../out` | `【代码/日志支持】` 受控恢复 `ps` 现场；权重实际落盘路径一致 |
| `--from_weight` | `none` | `【代码/日志支持】` 受控恢复 `ps` 现场 |
| `--from_resume` | `1` | `【本机已验证】` 最终日志从已有 step 跳过恢复；checkpoint 元数据与高优先级文档一致 |
| `--use_wandb` | 已显式传入 | `【本机已验证】` 最终日志出现 SwanLab run 初始化与完成信息 |

### 4.3 真实运行边界

- `【本机已验证】` 最终完成这次日志开头显示：`Epoch [1/1]: 跳过前226200个step，从step 226201开始`。
- `【代码/日志支持】` 这说明最终完成 run 不是从头训练，而是基于已有 resume checkpoint 继续推进。
- `【工程判断】` 高压现场曾确认 `step=219400`，而最终完成 run 从 `226200` 继续，说明在最终完成前恢复链路又向前推进过一次可恢复 checkpoint；现有材料足以支撑最终验收，但不足以逐分钟重建 `219400 -> 226200` 之间所有人工操作细节。

## 5. 最终权重、resume checkpoint、日志与 SwanLab 证据

### 5.1 最终训练日志

- `【本机已验证】` 日志路径：`/home/harry/projects/MiniMind/experiments/logs/pretrain-resume-20260708-010508.log`
- `【本机已验证】` `stat` 结果：`size=2404090 bytes`，`mtime=2026-07-08 04:56:44.464920488 +0800`
- `【本机已验证】` 日志末尾到达：`Epoch:[1/1](635119/635119)`
- `【本机已验证】` 日志末尾出现 SwanLab 完成提示，且未见异常结束语义

### 5.2 最终权重

- `【本机已验证】` 路径：`/home/harry/projects/MiniMind/out/pretrain_768.pth`
- `【本机已验证】` `stat` 结果：`size=137684407 bytes`，`mtime=2026-07-08 04:56:42.812167800 +0800`
- `【本机已验证】` `torch.load(..., map_location='cpu')` 可正常读取，返回 `dict`
- `【本机已验证】` `tensor_keys=91`
- `【本机已验证】` 前几个参数键为：
  - `model.embed_tokens.weight`
  - `model.layers.0.self_attn.q_proj.weight`
  - `model.layers.0.self_attn.k_proj.weight`
  - `model.layers.0.self_attn.v_proj.weight`
  - `model.layers.0.self_attn.o_proj.weight`
  - `model.layers.0.self_attn.q_norm.weight`
  - `model.layers.0.self_attn.k_norm.weight`
  - `model.layers.0.input_layernorm.weight`

### 5.3 最终 resume checkpoint

- `【本机已验证】` 路径：`/home/harry/projects/MiniMind/checkpoints/pretrain_768_resume.pth`
- `【本机已验证】` `stat` 结果：`size=649064413 bytes`，`mtime=2026-07-08 04:56:44.356264874 +0800`
- `【本机已验证】` `torch.load(..., map_location='cpu')` 读取出的关键元数据为：
  - `epoch=0`
  - `step=635119`
  - `wandb_id=gd3zf7856ek4ad8divdij`
  - `has_model=True`
  - `has_optimizer=True`
  - `has_scaler=True`
- `【代码/日志支持】` `epoch=0` 与 [train_pretrain.py](../trainer/train_pretrain.py) 的 0 基 epoch 保存方式一致；对“1 个 epoch 训练完成”的理解应看最终日志进度 `1/1` 与 step 到达终点，而不是把 `epoch=0` 误读成“没训练”。

### 5.4 训练结束状态与异常扫描

- `【本机已验证】` `screen -ls`：`No Sockets found in /run/screen/S-harry.`
- `【本机已验证】` `pgrep -af train_pretrain.py`：无输出
- `【本机已验证】` 对训练日志末尾 350 行执行关键词扫描，未发现：
  - `Traceback`
  - `RuntimeError`
  - `CUDA out of memory`
  - `OutOfMemory`
  - `KeyboardInterrupt`
  - `FileNotFoundError`
  - `Error:`

## 6. 初始 checkpoint 并发 writer 事故复盘

### 6.1 事故定义

`【代码/日志支持】` 初始事故发生在更早阶段：交互式 `screen` 与长训练命令混合粘贴，导致终端输入归属不清，随后出现两个独立训练主进程竞争同一组 checkpoint 与固定 `.tmp` 临时文件，并在 `os.replace(...)` 附近出现 `FileNotFoundError`。

### 6.2 为什么会撞到固定 `.tmp`

`【本机已验证】` 当前 [trainer_utils.py](../trainer/trainer_utils.py) 中 `lm_checkpoint(...)` 的保存路径是固定命名：

- 普通权重：`{save_dir}/{weight}_{hidden_size}.pth`
- resume checkpoint：`{save_dir}/{weight}_{hidden_size}_resume.pth`
- 临时文件：在正式路径后直接追加 `.tmp`

`【本机已验证】` 代码会先 `torch.save(..., ckp_tmp)`，再 `os.replace(ckp_tmp, ckp_path)`；resume checkpoint 也采用相同模式。

### 6.3 本次事故的准确结论

- `【代码/日志支持】` 根因不是 DataLoader worker、SwanLab、CUDA 或模型结构本身。
- `【工程判断】` 更合理的解释是：人为操作导致两个独立训练主进程同时写同一路径，固定 `.tmp` 文件名缺少跨进程互斥，最终触发替换阶段竞争。
- `【待验证】` 当前训练入口并未部署 `flock` 或其他机制级互斥锁，因此不能把后续“没有再撞车”写成“代码层面已彻底防住并发 writer”。

## 7. resume 长时内存 / swap 压力复盘

### 7.1 与初始 writer 事故的边界

`【代码/日志支持】` 后续 resume 高 swap 压力是另一类独立问题。高压现场真实训练 `PID 409` 的参数边界是 `--num_workers 0 --save_interval 200`；长期运行后出现高 RSS、高匿名内存、高进程 swap、系统 swap 接近上限，随后由用户主动发送 `SIGINT`，并出现 `KeyboardInterrupt`。

因此不能写成：

- `num_workers=2` 是本次高 swap 压力根因。
- checkpoint `.tmp` 竞争是本次高 swap 压力根因。
- 初始并发 writer 事故与后续高 swap 压力是同一个故障。

### 7.2 已确认的停止原因与未确认的底层原因

- `【代码/日志支持】` 已确认的直接停止原因：高压现场 swap 接近上限，用户为避免 OOM 主动中断。
- `【待验证】` 未确认的底层对象级原因：当前不能把问题唯一归因于 MiniMind 某一行代码、PyTorch allocator、DataLoader、checkpoint 序列化或 WSL 回收策略中的任何一个。

## 8. 受控恢复与最终完成过程

### 8.1 受控恢复阶段实际做了什么

- `【代码/日志支持】` 保持 Dense 768、真实预训练数据、`batch_size=2`、`max_seq_len=128`、`accumulation_steps=8`、`dtype=bfloat16`、`from_resume=1` 不变。
- `【代码/日志支持】` 将 `save_interval` 从 `200` 调整为 `5000`，目的是降低 checkpoint 保存频率，观察其与内存 / swap 压力的关系。
- `【代码/日志支持】` 通过 `pgrep`、`ps`、`screen -ls` 在特定时点人工核验只有一个 writer。

### 8.2 为什么可以认定这次受控恢复最终完成

- `【本机已验证】` 最终日志从 `step 226201` 持续推进到 `635119/635119`。
- `【本机已验证】` 最终权重与 resume checkpoint 的 `mtime` 都落在 `2026-07-08 04:56:42` 到 `04:56:44` 之间，与日志结束时间一致。
- `【本机已验证】` SwanLab 完成提示与最终日志尾部一致。
- `【本机已验证】` 当前没有遗留 screen 会话，也没有运行中的 `train_pretrain.py` 进程。

## 9. 已采取措施、有效证据与仍未验证边界

### 9.1 已采取措施

- `【代码/日志支持】` 降低 checkpoint 保存频率：`save_interval 200 -> 5000`
- `【代码/日志支持】` 建立内存 / swap / checkpoint mtime 连续监控
- `【代码/日志支持】` 在受控恢复阶段人工核验当前只有一个训练 writer

### 9.2 已观察到的有效证据

- `【本机已验证】` 当前可读监控文件是 `/home/harry/projects/MiniMind/experiments/logs/pretrain-memory-resume-20260708.csv`
- `【本机已验证】` 该 CSV 中，训练进程 `PID 441` 的 `VmSwap` 全程为 `0`，系统 `swap_used_kb` 仅低位波动，终盘附近约为 `356 kB`
- `【本机已验证】` 同一监控窗口中，`MemAvailable` 约在 `5053112 kB` 到 `6714804 kB` 之间波动，末段仍保留较大余量
- `【本机已验证】` 同一监控窗口中，`VmRSS` 约在 `4153456 kB` 到 `5916632 kB` 之间波动，后期可回落到约 `4.29 GiB` 到 `4.31 GiB`
- `【工程判断】` 这削弱了“RSS 严格线性上涨、必然爆掉”的判断，但不等于已经定位或根治对象级内存增长来源

### 9.3 与既有文档口径的差异说明

`【代码/日志支持】` 既有 resume 复盘文档的保守口径强调“约 4.05–5.38 GiB，后期回落到约 4.1 GiB，VmSwap 与系统 swap 接近 0”。本轮重新读取完整监控 CSV 时，看到的可直接复核范围略高，峰值约为 `5.64 GiB`，末段回落约 `4.3 GiB`。

`【工程判断】` 这不改变核心结论：

- 受控恢复窗口内没有重现此前高 swap 逼近上限的局面；
- RSS 呈波动并出现回落，而不是单调直线上升；
- 但这仍不足以把“底层内存增长来源已定位 / 已修复”写成结论。

### 9.4 仍未验证边界

- `【待验证】` `save_interval=5000` 与内存 / swap 关系只证明“相关性值得继续观察”，没有证明因果闭环。
- `【待验证】` 当前没有机制级单 writer 锁，后续仍可能被人工误启动绕过。
- `【待验证】` 训练完成不等于推理质量、能力边界、loss 收敛质量已经完成专项验收。

## 10. 后续长时训练的操作约定与风险控制建议

1. `【工程判断】` 避免再把交互式 `screen` 与长训练命令混合粘贴；应使用单条后台启动命令或统一启动脚本，减少终端输入归属不清。
2. `【工程判断】` 每次长训前先做只读检查：`pgrep -af train_pretrain.py`、`ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'`、`screen -ls`。
3. `【工程判断】` 若未来要把“唯一 writer”上升为机制级约束，应在唯一官方启动入口中固化同一个 `flock` 锁文件；当前还没有做到这一步。
4. `【工程判断】` 长时恢复阶段继续保留内存 / swap / checkpoint mtime 监控；若再次出现 `VmSwap` 连续增长、系统 swap 持续逼近上限、`MemAvailable` 持续下滑，再回到训练循环与保存逻辑做更细粒度定位。
5. `【工程判断】` 发生高压风险时，优先保住 checkpoint 和日志证据，再做受控中断；不要在高风险现场重复启动第二个训练进程。

## 11. 本轮文件改动、验证方式、版本同步与报告结论

### 11.1 本轮文件改动

- `【本机已验证】` 本轮只新增本报告：`docs/fix-report-v0.0.2-dense-768-pretrain-completion-2026-07-08.md`

### 11.2 本轮实际执行的只读验证

`【本机已验证】` 已执行并复核以下只读命令或等价检查：

- `git status --short --branch`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,220p' README.md`
- `sed -n '1,260p' docs/minimind-initial-pretrain-full-retrospective-updated.md`
- `sed -n '1,320p' docs/minimind-pretrain-resume-incident-20260708.md`
- `sed -n '1,260p' docs/pretrain-sft-manual-runbook-2026-07-07.md`
- `sed -n '1,260p' trainer/train_pretrain.py`
- `sed -n '1,260p' trainer/trainer_utils.py`
- `screen -ls`
- `pgrep -af train_pretrain.py`
- `stat` 检查最终日志、权重、resume checkpoint 的大小与时间戳
- `tail -n 40` 与 `tail -n 350 | rg ...` 检查最终日志结尾与异常关键词
- 使用项目虚拟环境 `torch.load` 读取最终权重与 resume checkpoint
- 读取 `pretrain-memory-resume-20260708.csv` 校验受控恢复窗口的 `VmRSS`、`VmSwap`、`MemAvailable`

### 11.3 版本同步

- `【本机已验证】` 本轮未修改 [README.md](../README.md) 中的 `v0.0.1` 徽章
- `【本机已验证】` 本轮没有改版本号，也没有改训练代码、数据、checkpoint、日志或运行脚本

### 11.4 报告结论

`【本机已验证】` 本报告认可的最终验收结论是：

> Dense 768 预训练在真实预训练数据上完成 1 个 epoch，训练日志到达 `635119/635119`，训练进程自然退出，SwanLab 标记实验完成，最终权重与完整 resume checkpoint 已落盘并可读取。

`【待验证】` 但本报告同时保留以下边界：初始并发 writer 事故与后续 resume 高 swap 压力是两类独立问题；受控恢复阶段只有一个 writer 只是人工核验状态，不是机制级锁；内存 / swap 压力的底层对象级增长来源仍需后续专项定位。
