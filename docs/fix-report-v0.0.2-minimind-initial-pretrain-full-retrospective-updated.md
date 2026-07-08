# MiniMind：初始预训练的完整故障复盘  
## screen 启动归属混乱、checkpoint 并发竞争，以及后续 WSL / Remote-WSL 失去响应

> **文档用途**
>
> 本文完整复盘最初 Dense 768 预训练阶段已经确认或有证据支撑的问题，并沉淀后续启动长时训练的操作约定。
>
> 本文覆盖两个独立阶段：
>
> 1. `screen` 使用方式与长命令混合粘贴引发终端归属混乱，进而重复启动训练并竞争同名 checkpoint 临时文件；
> 2. 在后续训练尝试中，WSL / Remote-WSL 失去响应，且当时 WSL 内存与 swap 余量偏小。
>
> **本文不分析最近一轮 resume 训练的内存、swap、save interval、监控脚本问题。** 那些问题发生在之后，应单独复盘，不能反向混入本次初始故障的根因判断。

---

## 1. 结论先行

最初预训练不是一个单点问题，而是先后暴露了两个独立阶段的风险。

### 阶段一：重复训练实例竞争 checkpoint 路径

最初将交互式 `screen -S minimind-pretrain` 与长训练命令放进同一大段命令粘贴执行，导致后续训练命令实际落在 screen 内部 shell 还是外层 VS Code WSL 终端的归属不清。

结合两个 SwanLab run 的创建时间、两份训练日志、进程关系，以及第一条训练在 checkpoint 保存阶段 `os.replace(...)` 周围出现的 `FileNotFoundError`，最合理的复盘是：

> screen 命令与训练命令混合粘贴造成终端归属混乱，随后重复启动了同名预训练。两条独立训练同时写同一组 checkpoint 和固定 `.tmp` 临时文件，最终第一条训练在 checkpoint 替换阶段与第二条训练竞争临时路径并中断。

这不是模型、CUDA、SwanLab 或 PyTorch DataLoader 自身造成的重复训练。

### 阶段二：WSL / Remote-WSL 失去响应

并发 checkpoint 竞争之后的后续训练尝试中，VS Code Remote-WSL 在训练仍有输出时已开始失去响应，表现为本地回环 WebSocket 超时、重连失败和 WebSocket `1006`。随后原始 WSL 实例结束，训练 Python、screen 和可能的 DataLoader worker 一并消失。

这一阶段不能直接简化为“VPN 断了”或“某一次 Linux OOM 一定杀掉了那条训练”。

更准确的判断是：

> 当时 WSL 内的 VS Code Remote 服务、扩展宿主，或承载它们的 WSL VM 已无法稳定响应；与此同时，WSL 当时只有约 7.5GiB RAM 和 2GiB swap，训练与 Remote-WSL / 扩展宿主共处的资源余量偏小，是明确的稳定性风险。

---

## 2. 初始训练的阶段一：screen 与长命令混合粘贴导致重复训练

### 2.1 当时的启动方式

当时曾将下列命令放在同一大段中执行：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs
screen -S minimind-pretrain
../.venv/bin/python train_pretrain.py \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 128 \
  --accumulation_steps 8 \
  --num_workers 2 \
  --log_interval 20 \
  --save_interval 200 \
  --use_wandb \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --from_resume 0 \
  2>&1 | tee ../experiments/logs/pretrain-$(date +%F-%H%M).log
```

风险不在于这些训练参数，而在于：

```bash
screen -S minimind-pretrain
```

这条命令会进入新的虚拟终端环境，而不是像普通短命令一样执行结束后立即把控制权交还给同一个外层 shell。

当它与后续几十行训练命令被一次性粘贴时，后续命令到底被哪一个 shell 接收容易不清晰：

```text
可能性 A：训练命令进入 screen 内部 shell
可能性 B：训练命令进入外层 VS Code WSL 终端
可能性 C：此前训练已启动，但因为不确定归属又再次执行了同一训练命令
```

本次证据不能百分之百还原当时每一次按键与每一次粘贴的精确落点；但从两个 SwanLab run、两份日志和 checkpoint 报错的组合看，重复训练已经实际发生。

### 2.2 为什么重复训练会让 checkpoint 报错

两个独立训练主进程使用同样的输出与 checkpoint 约定，例如：

```text
普通权重：
out/pretrain_768.pth

普通 checkpoint：
checkpoints/pretrain_768.pth

普通 checkpoint 临时文件：
checkpoints/pretrain_768.pth.tmp

resume checkpoint：
checkpoints/pretrain_768_resume.pth

resume checkpoint 临时文件：
checkpoints/pretrain_768_resume.pth.tmp
```

保存流程通常是：

```text
写入固定 .tmp 临时文件
        ↓
通过 os.replace(...) 替换正式 checkpoint
```

这在只有一个写入者时是合理的。

但两条训练主进程共享同一固定 `.tmp` 路径时，可能发生：

1. 训练 A 创建并写入 `.tmp`；
2. 训练 B 也创建、覆盖、替换或移动同名 `.tmp`；
3. 训练 A 再执行 `os.replace(...)`；
4. 它预期的 `.tmp` 已被另一条训练处理；
5. 因此出现 `FileNotFoundError`。

本质是：

> 固定 checkpoint 临时文件不是跨进程锁。训练代码默认同一个 checkpoint 路径只有一个写入者，而重复训练破坏了这个前提。

### 2.3 不应错误归因

以下说法都不准确：

```text
screen 有 bug 导致训练重复。
DataLoader worker 导致重复训练。
SwanLab 导致 checkpoint 损坏。
CUDA 或模型结构导致 os.replace 报错。
```

更准确的表述是：

> screen 的交互式使用方式本身没有被证明存在 bug；问题是将会改变输入归属的 `screen -S ...` 与长训练命令混合粘贴，增加了人工操作不确定性，随后重复启动训练并竞争同名 checkpoint 临时文件。

---

## 3. 必须区分：DataLoader worker 与重复训练主进程

当训练参数为：

```bash
--num_workers 2
```

时，训练主 Python 可能启动两个 DataLoader worker 子进程。这是正常的数据加载并行行为，不是重复训练。

典型进程关系类似：

```text
训练主 Python
├── DataLoader worker 1
└── DataLoader worker 2
```

真正危险的是两条互不属于父子关系的训练主进程：

```text
独立训练主进程 A：train_pretrain.py
独立训练主进程 B：train_pretrain.py
```

两者同时写相同的：

```text
save_weight = pretrain
save_dir    = ../out
checkpoint  = ../checkpoints/pretrain_768*.pth
```

以后判断是否误启动重复训练时，不能只看：

```bash
pgrep -af train_pretrain.py
```

有几行。

应使用：

```bash
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'
```

重点检查：

| 字段 | 用途 |
|---|---|
| `PID` | 进程本身的编号 |
| `PPID` | 父进程编号，用于判断是否是父子关系 |
| `TTY` | 该进程属于哪个终端或 pseudo terminal |
| `ETIME` | 已运行时间，辅助判断启动顺序 |
| `CMD` | 是否真正是 `train_pretrain.py` 主训练命令 |

判断原则：

- 一条 `train_pretrain.py` 主训练进程，加若干 DataLoader worker：通常正常；
- 两条彼此独立、PPID / TTY 不同的 `train_pretrain.py` 主训练进程：高风险；
- 一旦看到疑似重复训练：立即停止新增启动动作，先只读检查进程树，不要再执行第三条训练命令。

---

## 4. 后续推荐的默认启动方式：由 screen 直接后台承接训练命令

后续不再推荐“先交互式进入 screen，再人工粘贴长训练命令”的方式。

默认推荐：

```bash
screen -dmS <session_name> bash -lc '<完整训练命令>'
```

原因是它让 screen 在创建 detached session 的同时，直接执行明确的训练命令，避免人工交互式 screen 中的“后续长命令到底输入到了哪个 shell”问题。

### 4.1 本轮后续 resume 实际采用的启动方式

之后恢复训练时，使用过以下模式：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs ../out ../checkpoints

screen -dmS minimind-pretrain-resume bash -lc '
cd /home/harry/projects/MiniMind/trainer
../.venv/bin/python train_pretrain.py \
  --save_dir ../out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 128 \
  --accumulation_steps 8 \
  --num_workers 2 \
  --log_interval 20 \
  --save_interval 200 \
  --use_wandb \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --from_weight none \
  --from_resume 1 \
  2>&1 | tee ../experiments/logs/pretrain-resume-$(date +%F-%H%M%S).log
'

screen -r minimind-pretrain-resume
```

这次方式相较最初交互式 screen 启动的关键改进是：

| 改进 | 作用 |
|---|---|
| `screen -dmS minimind-pretrain-resume bash -lc '...'` | 由 screen 直接在后台承接完整训练命令，避免进入交互式 screen 后再粘贴长命令 |
| 使用不同 session 名 | 与最初训练会话区分，减少误操作旧任务的可能 |
| 使用 `--from_resume 1` | 从完整 resume checkpoint 继续，包括模型、优化器和 scaler 状态 |
| 显式传入输出与模型参数 | 保证恢复使用的配置可复核 |
| 日志文件带秒级时间戳 | 区分不同恢复尝试，减少日志文件名冲突 |
| 使用 `tee` 保存标准输出与错误输出 | 保留训练过程证据，便于事后定位 |

### 4.2 后续更稳妥的模板

以下适用于未来**确认没有其他训练主进程**、且当前并没有运行同名训练时。

在 **WSL Bash** 中执行：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs ../out ../checkpoints

session_name="minimind-pretrain-resume"
log="../experiments/logs/pretrain-resume-$(date +%F-%H%M%S).log"

screen -dmS "$session_name" bash -lc "
set -o pipefail
cd /home/harry/projects/MiniMind/trainer || exit 1

../.venv/bin/python -u train_pretrain.py \
  --save_dir ../out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 128 \
  --accumulation_steps 8 \
  --num_workers 2 \
  --log_interval 20 \
  --save_interval 200 \
  --use_wandb \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --from_weight none \
  --from_resume 1 \
  2>&1 | tee -a '$log'
"
```

> 注意：这只是后续恢复训练或下一轮实验的推荐写法。  
> **当前已经有训练在运行时，绝对不要为了“换成更稳的命令”而再执行一次该后台启动命令。** 否则可能再次创建第二条训练主进程并重演 checkpoint 竞争。

### 4.3 启动后的只读确认

新的训练启动后，再单独做只读确认：

```bash
screen -ls

ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'

tail -n 80 ../experiments/logs/pretrain-resume-*.log
```

应确认：

- 只有预期的一个训练主进程；
- 日志开始输出；
- SwanLab run 初始化正常；
- 没有 argparse 参数错误；
- checkpoint 位置与当前训练任务匹配。

---

## 5. 初始训练的阶段二：WSL / Remote-WSL 失去响应

这一阶段发生在 checkpoint 并发竞争之后的后续训练尝试中，是另一件独立问题。

### 5.1 关键时间线

现有日志对应的大致顺序如下：

| 时间 | 观察到的现象 | 可以得出的结论 |
|---|---|---|
| 约 21:39:51 | 最后一条预训练日志仍在推进 | 训练此前仍有正常计算输出 |
| 约 21:41:01 | VS Code Remote Extension Host 被记录为 unresponsive | Remote-WSL 侧已出现失去响应迹象 |
| 随后 | `WebSocket(127.0.0.1:3334)` socket timeout、重连失败、WebSocket `1006` | Windows 与 WSL 中 VS Code Server 的本地回环通道失去正常通信 |
| 约 21:41:23 | 某个 checkpoint `.tmp` 文件表现为 0 字节 | 当时保存流程没有正常完成，但该现象本身不能单独证明原因 |
| 约 21:44 | 原始 WSL 实例结束 | 该实例内的 screen、Python、DataLoader worker 都会消失 |
| 约 21:56 | 另一次后续 WSL 启动中记录到 Linux OOM killer 杀 Python | 证明当晚确实发生过 WSL 内存耗尽，但不能反推它一定是 21:41 那次原始训练的直接死因 |

### 5.2 为什么不能直接说“只是 VPN 问题”

关键故障目标是：

```text
WebSocket(127.0.0.1:3334)
```

这是 Windows 本机与 WSL 内 VS Code Server 之间的本地回环通信，不是 SwanLab、GitHub 或其他外网服务。

因此，更严谨的说法是：

> 当时 WSL 内的 VS Code Remote 服务、扩展宿主，或承载它们的 WSL VM 已经无法稳定响应；VS Code 的反复重连是故障表现，不是已经被证明的唯一根因。

VPN / Clash 仍不能完全排除。当前 WSL 使用：

```ini
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

Mirrored networking 会把 Windows 网络接口映射到 Linux；DNS tunneling 旨在提高 VPN 和复杂网络环境下的兼容性。它们说明网络链路可能是排查变量之一，但不构成“Clash 就是根因”的证据。

因此，以下两种说法都不成立：

```text
关闭 VPN 就一定能解决训练中断。
Clash 就是本次 WSL 故障的根因。
```

正确策略不是在训练中途随意关 VPN、切节点或重启网络，而是：

1. 在启动训练前固定网络状态；
2. 一轮训练中不再引入网络变量；
3. 只有复现相似故障时，才在下一轮启动前做明确的单变量 A/B 对照；
4. 训练期间优先保留日志、进程树、WSL 内存和系统事件证据。

### 5.3 当时明确存在的资源风险

当时 WSL 实际可见资源约为：

```text
RAM:  7.5GiB
Swap: 2.0GiB
```

同时 WSL 内运行或常驻的内容包括：

- 训练 Python；
- VS Code Server；
- Remote Extension Host；
- Codex app-server；
- 其他系统服务；
- 文件缓存；
- 可能的 DataLoader worker。

这不能证明“21:41 时 Linux 内存直接杀掉了原训练”，但它足以说明：

> 在仅约 7.5GiB RAM、2GiB swap 的 WSL 环境中，让长时训练与 Remote-WSL、扩展宿主和其他开发服务共处，稳定性余量较小。

### 5.4 后续对承载环境做的改进

之后已将 `%UserProfile%\.wslconfig` 调整为：

```ini
[wsl2]
memory=10GB
swap=8GB
swapFile=D:\\WSL\\swap.vhdx
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

并在 WSL VM 重启后验证到：

```text
Mem:   9.7GiB
Swap:  8.0GiB
```

这项调整的含义是：

- 给训练更多物理内存上限；
- 增大 swap 缓冲；
- 不等于已经证明根因是内存；
- 不等于可以在训练期间随意开启额外 WSL 服务；
- 不等于可以忽略后续 `VmRSS`、`VmSwap`、`MemAvailable` 和系统 swap 使用趋势。

---

## 6. 对完整初始故障的准确归纳

### 已确认事实

- 最初出现过两条独立 `train_pretrain.py` 主训练进程。
- 它们竞争同一组 checkpoint / `.tmp` 路径。
- 第一条训练在 checkpoint 原子替换阶段报错。
- 后续训练尝试中，VS Code Remote-WSL 在训练停止前已经出现本地回环 WebSocket 超时。
- 原始 WSL 实例随后结束，导致其内部训练和 screen 会话消失。
- 当晚另一次后续 WSL 启动中发生过 Linux OOM killer 杀 Python。
- 初始 WSL 资源边界约为 7.5GiB RAM、2GiB swap，后续已扩展为约 9.7GiB RAM、8GiB swap。

### 合理但仍需保留边界的判断

- 交互式 screen 与训练长命令混合粘贴，是重复启动训练的最合理直接诱因。
- WSL / Remote-WSL 的失去响应与较小资源余量有关联风险。
- VPN / Clash / mirrored networking 可能是间接变量，但不是当前最有力的单一根因。
- 后来出现的 Linux OOM 证明资源问题真实存在，但不能倒推出它一定直接杀掉 21:41 那个原始训练实例。

### 不应写成的结论

```text
screen 有 bug。
DataLoader worker 导致重复训练。
VPN 是唯一根因。
21:56 的 OOM 一定杀掉了 21:41 的训练。
仅仅增加 swap 就从根本解决了所有稳定性问题。
预训练已经完成。
```

---

## 7. 后续运行的硬约定

### 7.1 启动前

在 **WSL Bash** 中先做只读检查：

```bash
echo "========== screen sessions =========="
screen -ls || true

echo
echo "========== train_pretrain.py =========="
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py' || true
```

只有确认：

- 没有已有训练主进程；
- 没有仍需检查或继续的同名 screen session；
- 没有其他训练正在写同一 checkpoint 路径；

才允许新启动训练。

### 7.2 启动时

- 默认使用：
  ```bash
  screen -dmS <session_name> bash -lc '<完整训练命令>'
  ```
- 训练会话、日志文件名、输出目录和 checkpoint 命名要明确对应。
- 每一轮启动保留独立、带时间戳的日志。
- 不把交互式 `screen -S ...` 与长训练命令混在同一大段粘贴。

### 7.3 训练期间

不要执行：

```text
wsl --shutdown
wsl --terminate
Windows 重启或关机
关闭或强杀训练所在 screen
在未确认现有进程前重复运行训练启动命令
在训练中途切 VPN、切 Clash 节点或改网络模式
```

可以进行只读检查：

```bash
screen -ls
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'
tail -n 80 <log_file>
free -h
nvidia-smi
```

### 7.4 出现异常时

若发现：

- 外层终端仍在刷 Epoch 日志；
- screen 状态与预期不一致；
- 出现多个 `train_pretrain.py`；
- checkpoint 写入异常；
- VS Code Remote-WSL 断连；

先停止新增启动动作，不要先“再开一个训练试试”。

正确顺序是：

1. 只读检查 screen；
2. 只读检查训练进程树；
3. 只读检查日志；
4. 记录 checkpoint mtime 和内部 step；
5. 再判断是重复训练、环境失联、资源压力，还是训练脚本本身报错。

---

## 8. 可复用的一句话复盘

> 最初 MiniMind 预训练先因交互式 screen 与长训练命令混合粘贴造成终端归属不清，进而重复启动训练，使两条同名主进程竞争固定 checkpoint 临时文件；随后又出现 WSL / Remote-WSL 本地回环通信失去响应以及资源余量偏小的问题。后续采用 `screen -dmS <session> bash -lc '...'` 由 screen 直接后台承接完整训练命令，并在启动前强制检查唯一训练主进程，以降低重复启动和 checkpoint 并发写入风险。

---

## 9. 文档边界

本文只记录：

- 初始预训练的启动与 checkpoint 竞争问题；
- 后续 WSL / Remote-WSL 失去响应的证据与边界；
- 后续启动方式的操作约定；
- 已执行的 WSL 资源边界调整。

本文没有：

- 修改训练脚本；
- 修改 checkpoint；
- 删除日志；
- 修改版本号；
- 生成修复报告；
- 声称预训练、SFT 或推理已经完成；
- 分析最近一轮 resume 中的内存 / swap / 保存间隔 / 监控脚本问题。

---

## 10. 参考依据

- GNU Screen 官方手册：detached session 与会话管理机制。
- Microsoft WSL 官方文档：`.wslconfig` 的 WSL2 VM 资源设置。
- Microsoft WSL 官方网络文档：mirrored networking、DNS tunneling 与 auto proxy 的行为说明。
