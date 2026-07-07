# MiniMind Dense 768 预训练 Resume：内存 / Swap 压力与单 Writer 边界复盘（第三次修订，2026-07-08）

> **本文范围**：复盘这次 Dense 768 `minimind-pretrain-resume` 断点续训在长时间运行后出现的主机内存 / swap 压力、用户主动中断、当前受控恢复、监控建立，以及“唯一训练 writer”究竟做到什么、没有做到什么。
>
> **最重要的更正**：当前这次 resume 运行期间，并没有给训练入口加 `flock` 或其他原子互斥锁。因此“只有一个 writer”是通过 `ps/pgrep` 与 `screen -ls` 在特定时点**人工确认**的运行状态，不是机制级保证。此前文档若写成“保证唯一 writer”或“已避免并发 writer”，均应以本版本为准。
>
> **不混入的历史事故**：曾有另一次独立事故：两个训练实例同时操作同一个 `checkpoints/pretrain_768_resume.pth.tmp`，并在 `os.replace(...)` 附近发生 `FileNotFoundError`。那次事故说明 checkpoint writer 必须单实例，但**不是本文这次内存 / swap 压力的根因**。
>
> **状态边界**：截至本文记录，Dense 768 预训练尚未完成；full SFT、SFT checkpoint 验证、固定样例推理、cache / 非 cache 验证均尚未开始。

---

## 1. 先给结论：这次任务为什么停、现在解决到哪一步

这次任务不能写成“resume 训练报错后被彻底修复”。准确结论应分四层：

1. **训练停止的直接原因（已验证）**：在长时间运行后，训练 Python `PID 409` 的常驻内存和进程 swap 占用很高，系统 swap 一度达到 `7.3 GiB / 8.0 GiB`。用户为了避免 WSL 或 Linux OOM killer 被动杀死训练，主动执行 `kill -INT 409`。之后出现的 `KeyboardInterrupt` 是人工 `SIGINT` 的预期结果，不是模型自行报出的 NaN、CUDA OOM 或 checkpoint 损坏。
2. **底层代码级根因（未证实）**：现有证据能证明“长期训练时主机内存与 swap 压力显著升高”，但还不能定位到 MiniMind 某一行代码、DataLoader、checkpoint 序列化、PyTorch allocator 或 WSL 回收策略中的唯一根因。
3. **当前已经完成的处置（已验证）**：保留了可恢复 checkpoint；启动当前受控恢复时，将 `save_interval` 从 `200` 改为 `5000`；建立了常驻 RSS / swap / checkpoint mtime 监控；并在启动、监控时检查到当前仅有一个 `train_pretrain.py` 进程。
4. **当前尚未完成的事情（未验证）**：尚未证明 `save_interval=5000` 已消除长期内存增长；也尚未在训练入口实现机制级单 writer 锁。因此不能写“内存问题已根治”或“唯一 writer 已被强制保证”。

一句话复盘：

> 这次 resume 的直接终止原因是严重的内存 / swap 压力下，用户主动发送 SIGINT；当前采取的是低频 checkpoint + 连续监控的受控恢复策略。单 writer 在运行时被人工核验，但尚未通过锁机制强制保证。

---

## 2. 运行脉络：不要把三段训练与两类问题混在一起

本轮对话涉及多次训练和多类问题，复盘必须分开。

| 阶段 / 对象 | 当前掌握的事实 | 与本文关系 |
|---|---|---|
| 早先的 `minimind-pretrain` screen | 用户曾通过 `screen -r minimind-pretrain` 查看早期训练；本文没有重新构造其完整命令、PID 与全部运行结果 | 只作为此前预训练过程背景，不拿它替代 resume 现场证据 |
| 历史 `.tmp` checkpoint 竞争事故 | 曾有两个训练实例同时写同一 `pretrain_768_resume.pth.tmp`，在 `os.replace(...)` 附近报 `FileNotFoundError` | 独立的并发 writer 事故；不是本次内存 / swap 压力根因 |
| 高压 resume 现场 | `PID 409` 运行 `train_pretrain.py`，`--num_workers 0 --save_interval 200`；RSS、匿名内存、进程 swap 与系统 swap 都很高 | 本文要复盘的主要故障现场 |
| 当前受控恢复 | `screen=439.minimind-pretrain-resume`，训练 PID `441`，`--num_workers 0 --save_interval 5000`，另有监控 screen `947.minimind-pretrain-monitor-v2` | 当前正在观察的恢复任务，不等于已经完成或根治 |

### 2.1 一个必须保留的参数证据冲突

你提供过一份外层启动命令，里面显式写了：

```text
--num_workers 2
--save_interval 200
```

但接近一小时、准备主动中断时，`ps` 显示真正高压的 `PID 409` 实际参数为：

```text
--num_workers 0
--save_interval 200
```

因此，后续任何复盘必须让**高压时刻的 `ps` 进程参数**优先于曾复制保存的启动模板。不能写：

- “本次内存问题是 `num_workers=2` 导致的”；
- “把 worker 从 2 改成 0 后解决了问题”；
- “外层启动命令必然等于高压时 PID 409 的真实命令行”。

---

## 3. Resume 与 checkpoint：哪些已经验证、哪些没有

### 3.1 断点续训本身可以工作（已验证）

最初可恢复 checkpoint 的元数据为：

```text
epoch: 0
step: 73000
wandb_id: gd3zf7856ek4ad8divdij
has_model: True
has_optimizer: True
has_scaler: True
```

这说明该 checkpoint 至少包含模型、优化器、AMP scaler、step 与实验标识。恢复后训练能继续推进到十多万 step 之后，因此本次问题不能归类为：

- checkpoint 无法加载；
- 模型结构不匹配；
- `--from_resume 1` 在启动阶段失败；
- BF16、数据路径或 GPU 在启动阶段立即异常。

上游 MiniMind 文档把 `--from_resume 1` 用于恢复包含模型、优化器与训练进度的 checkpoint；这只是机制背景，不等于本机训练已经完成。参考：[上游 MiniMind README](https://github.com/jingyaogong/minimind/blob/master/README_en.md)。

### 3.2 高压现场最后确认到的 checkpoint

接近一小时、发送 SIGINT 前，检查到：

```text
path=checkpoints/pretrain_768_resume.pth
modified=2026-07-08 00:52:42.146651825 +0800
size=649064413 bytes

epoch: 0
step: 219400
wandb_id: gd3zf7856ek4ad8divdij
```

正确解释：

- 当时有可用的已落盘 checkpoint，内部 `step=219400`；
- 这个 step 是最后保存成功的状态，并不必然等于 `SIGINT` 到达时 Python 正在执行的实时 step；
- 它支持“训练停止前保留了恢复点”，不支持“训练已经完成”。

---

## 4. 用户为什么主动中断：完整内存 / swap 现场证据

### 4.1 第一次检查：系统 swap 已达到 6.3 GiB

```text
Mem:   total 9.7GiB, used 8.1GiB, free 81MiB, buff/cache 1.8GiB, available 1.6GiB
Swap:  total 8.0GiB, used 6.3GiB, free 1.7GiB
```

`free=81MiB` 不应脱离 `available=1.6GiB` 单独判断；但在训练持续运行、swap 已被大量使用时，可用内存只有约 `1.6GiB` 已属于明显高风险状态。

### 4.2 `PID 409` 的内存状态：匿名内存与进程 swap 均很高

```text
VmHWM:   8554988 kB
VmRSS:   7991340 kB
RssAnon: 7787792 kB
RssFile:  126172 kB
RssShmem:  77376 kB
VmSwap:  6923824 kB
```

| 指标 | 现场值 | 可以确认的含义 | 不能据此断定的内容 |
|---|---:|---|---|
| `VmHWM` | 约 8.16 GiB | 历史 RSS 峰值曾达到很高水平 | 峰值的对象级来源 |
| `VmRSS` | 约 7.62 GiB | 当前驻留 RAM 的内存很高 | 一定存在 Python 代码泄漏 |
| `RssAnon` | 约 7.43 GiB | 压力主要不是文件页缓存 | 匿名内存一定由某个特定模块造成 |
| `RssFile` | 约 0.12 GiB | 文件映射驻留页相对较小 | 文件 I/O 与问题完全无关 |
| `VmSwap` | 约 6.60 GiB | 训练进程已有大量页被换出到 swap | `VmRSS + VmSwap` 可直接当成真实内存总占用 |

### 4.3 第二次检查：swap 已进一步升至 7.3 GiB

```text
Mem:   total 9.7GiB, used 7.7GiB, free 87MiB, buff/cache 2.2GiB, available 2.0GiB
Swap:  total 8.0GiB, used 7.3GiB, free 706MiB

PID   RSS      %MEM   CMD
409   7408900  72.7   ../.venv/bin/python train_pretrain.py \
      --epochs 1 --batch_size 2 --max_seq_len 128 --accumulation_steps 8 \
      --num_workers 0 --log_interval 20 --save_interval 200 \
      --dtype bfloat16 --hidden_size 768 --num_hidden_layers 8 \
      --use_moe 0 --save_weight pretrain --save_dir ../out \
      --from_weight none --from_resume 1 --use_wandb
```

此处能确认：

1. `PID 409` 是 WSL 中 RSS 最大的进程；
2. 系统 swap 已到 `7.3 / 8.0 GiB`，剩余约 706 MiB；
3. 高压现场真实参数是 `num_workers=0`、`save_interval=200`；
4. 主动中断是基于容量风险的保守处置，而不是由于 NaN、CUDA OOM traceback、SwanLab 报错或 checkpoint 损坏。

---

## 5. 用户实际执行的受控中断

用户在确认风险后执行：

```powershell
wsl -d Ubuntu-24.04 -- kill -INT 409

Start-Sleep -Seconds 15

wsl -d Ubuntu-24.04 -- bash $checkPathWsl
wsl -d Ubuntu-24.04 -- bash $checkpointCheckPathWsl
wsl -d Ubuntu-24.04 -- free -h
```

这组操作的含义：

1. `kill -INT 409` 向训练 Python 发送 `SIGINT`；
2. Python 在可响应的位置收到中断，表现为 `KeyboardInterrupt`；
3. 等待 15 秒后检查训练是否真正结束、checkpoint 保留到哪个 step、内存与 swap 是否回落；
4. 已记录的停止后状态为：

```text
Mem available: 8.9GiB
Swap used:     51MiB
```

这支持“高内存 / swap 压力与训练进程运行强相关”的判断。它仍然不能把压力唯一归因于训练代码的某一行。

---

## 6. 根因判断：直接终止原因与底层原因必须分开

### 6.1 已验证的直接终止原因

> 长时间 resume 训练的 `PID 409` 处于高 RSS、高匿名内存、高进程 swap 状态，且系统 swap 逼近 8 GiB 上限。用户为避免 OOM 主动发送 SIGINT，训练因此退出。

这是最准确的“为什么停了”。

### 6.2 尚未定位的底层技术原因

| 候选因素 | 现有证据 | 当前结论边界 |
|---|---|---|
| 训练循环长期保留 Python / tensor / batch / log 引用 | `RssAnon` 高，值得代码审计 | 未做对象级 profile，不能称为泄漏 |
| 每 200 step checkpoint 序列化 / I/O | 高压现场确为 `save_interval=200` | 需要保存前后连续曲线，不能称为已证实根因 |
| PyTorch / Python allocator 保留 | 可能造成长期匿名内存高位 | 尚未采集 allocator 或 profile 证据 |
| WSL 内存 / swap 回收策略 | WSL 的 10GB RAM / 8GB swap 是实际容量边界 | 不是单一根因的证据 |
| `num_workers=2` | 高压现场 PID 409 为 `num_workers=0` | 不能作为本次高压问题的必要条件或主因 |
| VPN / Clash / VS Code | 历史上有相关现象 | 没有唯一因果证据，不能写成根因 |

### 6.3 可以与不可以使用的项目表述

可以写：

> Dense 768 预训练断点续训中，训练 Python 的 RSS、匿名内存与进程 swap 在长时间运行后处于高位，系统 swap 一度达到 7.3 / 8.0 GiB。为避免 OOM，主动中断训练，并保留 checkpoint 继续做受控恢复与连续监控。底层对象级增长来源尚未定位。

不能写：

- “已定位并修复 MiniMind 内存泄漏”；
- “DataLoader worker 是根本原因”；
- “checkpoint 保存必然导致泄漏”；
- “关闭 VPN 就能解决”；
- “本次 resume 错误就是 `.tmp` checkpoint 竞争”。

---

## 7. 已采取的处置：什么是事实，什么只是实验假设

### 7.1 受控停止，而不是等待 OOM（已完成）

在 swap 接近耗尽前主动 `SIGINT`，避免由 WSL 或 Linux OOM killer 被动终止，保住已有 checkpoint。

### 7.2 当前受控恢复：`save_interval=200 -> 5000`（已完成，但效果待验证）

当前恢复保留：Dense 768、真实预训练数据、`batch_size=2`、`max_seq_len=128`、`accumulation_steps=8`、BF16、`--from_resume 1`。

当前确认改变的是：

```text
--save_interval 5000
```

其工程目的，是减少频繁 checkpoint 序列化、临时文件写入、文件替换与 I/O page-cache 扰动的次数，从而观察保存频率是否与长期内存 / swap 压力相关。

这不是“已修复”的证据。它的代价是异常中断时，未保存训练进度的最大损失窗口增大。

### 7.3 当前任务名称、screen、PID 与实际命令（已验证）

建议逻辑名称：

> **Dense 768 pretrain — controlled resume #2**

最后一次已验证的运行对象：

```text
training screen: 439.minimind-pretrain-resume
training PID: 441
monitor screen: 947.minimind-pretrain-monitor-v2
```

最后一次已验证的训练命令：

```bash
../.venv/bin/python train_pretrain.py \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 128 \
  --accumulation_steps 8 \
  --num_workers 0 \
  --log_interval 20 \
  --save_interval 5000 \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --save_weight pretrain \
  --save_dir ../out \
  --from_weight none \
  --from_resume 1 \
  --use_wandb
```

备注：这段命令来自当时 `PID 441` 的 `ps` 现场。未显示的参数可能由训练脚本默认值提供，但本文不把它们写成已确认显式参数。

### 7.4 常驻监控已启动（已验证）

监控 CSV 字段：

```text
timestamp,pid,mem_available_kb,swap_used_kb,vmrss_kb,vmswap_kb,rssanon_kb,rssfile_kb,checkpoint_mtime_epoch
```

前台验证样本：

```text
2026-07-08T01:26:44+08:00,441,6707872,0,4134508,0,2560740,1493892,1783445050
2026-07-08T01:27:14+08:00,441,6708972,0,4138124,0,2560900,1497348,1783445216
```

短时可确认：

- `VmSwap=0`；
- 系统 `swap_used_kb=0`；
- 30 秒内 `VmRSS` 仅小幅变化；
- 当前短窗口没有复现此前的 swap 接近上限状态。

短时不能确认：

- `save_interval=5000` 已根治问题；
- 长时间或跨多个 checkpoint 周期仍会保持稳定；
- 底层内存增长来源已被定位。

---

## 8. “唯一训练 writer”：准确表述、实际做法与未完成边界

这是本次文档必须写清楚的部分。

### 8.1 为什么需要单 writer

历史 `.tmp` checkpoint 竞争事故说明：两个 `train_pretrain.py` 同时写同一个 checkpoint 路径时，可能同时操作固定临时路径：

```text
checkpoints/pretrain_768_resume.pth.tmp
```

一个进程替换或删除临时文件后，另一个进程可能在 `os.replace(...)` 附近找不到预期文件。这个事故与本文高 swap 问题独立，但它要求以后每次启动训练都必须避免重复 writer。

### 8.2 当前实际做到的是“检查到一个 writer”，不是“强制保证一个 writer”

本轮用于确认当前训练数量的命令包括：

```powershell
wsl -d Ubuntu-24.04 -- pgrep -af "train_pretrain[.]py"
```

以及：

```powershell
wsl -d Ubuntu-24.04 -- screen -ls
```

在当前受控恢复时，已观察到：

```text
439.minimind-pretrain-resume
947.minimind-pretrain-monitor-v2
```

其中真实训练 Python 为 `PID 441`，监控为独立 screen。监控启动器还执行过一次“当下训练进程数必须恰好为 1”的检查，核心逻辑是：

```bash
mapfile -t train_pids < <(
  ps -eo pid=,comm=,args= |
  awk '
    $2 ~ /^python([0-9.]*)?$/ &&
    $0 ~ /train_pretrain[.]py/ {
      print $1
    }
  '
)

if [ "${#train_pids[@]}" -ne 1 ]; then
  echo "ABORT: expected exactly one train_pretrain.py" >&2
  exit 10
fi
```

这个检查可以做到：

- 在该检查执行的**那个时点**发现 0 个、2 个或更多训练进程；
- 避免在“已明显存在重复训练”的状态下继续启动监控；
- 记录真实训练 PID，防止监控误采集 screen PID。

它**不能**做到：

- 阻止用户随后在另一个 PowerShell / screen 中再启动第二个训练；
- 让两个几乎同时启动的训练命令原子地互斥；
- 防止绕开此检查、直接执行 `python train_pretrain.py ...` 的重复启动。

因此本轮准确写法必须是：

> 当前受控恢复期间，通过 `pgrep`、`ps` 与 `screen -ls` 人工确认当前只有一个 `train_pretrain.py` writer；并未部署原子互斥锁，因此这是一项运行时核验，不是机制级保证。

### 8.3 `screen` 不是单 writer 锁

`screen -dmS` 的作用是创建 detached 会话，让命令脱离当前终端继续运行；它不是排他锁。同一个机器上可以有多个不同名称的 screen，会话也可以各自运行第二个 Python。GNU Screen 的 detached 语义是“与终端断开并在后台保持会话”，不是“禁止同类命令再次启动”。参考：[GNU Screen Detach 文档](https://www.gnu.org/software/screen/manual/html_node/Detach.html)。

所以以下逻辑是不成立的：

```text
“训练在 screen 里运行” => “只能有一个训练 writer”
```

### 8.4 将来怎样做到机制级单 writer：`flock`（尚未部署）

未来若需要真正阻止误启动，可在**所有训练启动入口**统一采用同一个 lock 文件，例如：

```bash
flock -n \
  /home/harry/projects/MiniMind/checkpoints/pretrain_768.writer.lock \
  /home/harry/projects/MiniMind/.venv/bin/python -u train_pretrain.py \
  --epochs 1 \
  --batch_size 2 \
  ...
```

含义：

- `flock -n` 尝试获取指定 lock 文件的独占锁；
- 如果该锁已被一个训练启动器持有，第二次启动会立即失败，而不是等待；
- 锁持有期间，`flock` 启动并保持训练子进程；训练退出后锁自动释放；
- 这能防止**同样遵守同一 lock 路径的启动命令**并发运行。

仍需保留一个边界：`flock` 是 advisory lock。它只能约束使用同一个 lock 文件、并主动通过 `flock` 启动的命令；若有人故意绕过锁，直接执行 Python，内核不会自动阻止。因此应把 `flock` 固化到唯一官方启动脚本里，而不是只把它写在一条临时命令中。Linux `flock` 的独占锁语义与非阻塞选项说明见：[flock(1) 手册](https://man7.org/linux/man-pages/man1/flock.1.html) 与 [flock(2) 手册](https://man7.org/linux/man-pages/man2/flock.2.html)。

**重要：本次正在运行的 `PID 441` 没有确认通过 `flock` 启动。**因此不能追溯性地声称它受 `flock` 保护。

---

## 9. 后续如何判定当前恢复方案是否有效

至少观察更长时间，并跨越一个或多个 checkpoint 更新，再判断趋势：

| 观察项 | 支持“当前较稳定”的表现 | 需要警惕的表现 |
|---|---|---|
| `VmSwap` | 长期保持 0 或低位波动 | 连续上升且不回落 |
| 系统 `swap_used_kb` | 不持续逼近 8 GiB | 连续接近 8 GiB |
| `MemAvailable` | 保持合理余量 | 持续接近极低值 |
| `RssAnon` | 波动或阶段性稳定 | 跨多个采样窗口持续单调增长 |
| checkpoint mtime | 正常更新且内存无异常跃迁 | 每次保存后内存阶跃并持续堆积 |
| 训练 screen / PID | 存活、step 推进 | 退出、OOM killer 记录或 checkpoint 不更新 |

若再次出现：

```text
VmSwap 持续增长
+ 系统 swap 接近 8 GiB
+ MemAvailable 持续下降
+ RssAnon 跨多个采样窗口持续上涨
```

下一次恢复前再读取训练入口、数据集实现与 checkpoint 保存逻辑，重点检查：

- 训练循环是否将 loss、tensor、batch、log 或结果持续保留在容器中；
- checkpoint 保存前后 RSS / swap 是否出现稳定阶跃；
- 数据读取或 worker 行为是否额外保留对象；
- Python / PyTorch allocator 与文件页是否在长时间运行中持续扩张。

在此之前，不应为“跑通”而缩小模型、换玩具数据、删除 checkpoint、重装 CUDA / PyTorch / WSL，或把 VPN / Clash 写成唯一根因。

---

## 10. 可复用的项目表述

### 10.1 可以写进实验记录

> Dense 768 预训练断点续训可以从 checkpoint 恢复模型、优化器、scaler 与训练进度。一次长时间 resume 运行中，训练进程的 RSS、匿名内存和进程 swap 显著升高，系统 swap 一度达到 7.3 / 8.0 GiB，因此我在发生 OOM 前主动发送 SIGINT，并确认已有 checkpoint 可继续恢复。后续在不改变模型、真实数据和核心训练语义的前提下，将 checkpoint 保存间隔从 200 调整为 5000，并建立了 RSS、swap 与 checkpoint 时间戳的连续监控。当前短时间样本显示 swap 为 0，但底层内存增长来源仍在验证中。当前受控恢复通过进程与 screen 状态核验只有一个训练 writer；尚未部署原子互斥锁。

### 10.2 暂时不能写

- “我已定位并修复 MiniMind 内存泄漏。”
- “`num_workers=2` 是这次问题根因。”
- “`save_interval=5000` 已根治内存问题。”
- “当前训练通过机制级锁保证了唯一 writer。”
- “只要在 screen 中运行，就不会重复启动训练。”
- “Dense 768 预训练已经完成。”
- “SFT 与推理验证已经完成。”
- “本次 resume 的错误就是 checkpoint `.tmp` 并发竞争。”

---

## 11. 最终结论

本次 `minimind-pretrain-resume` 的准确复盘是：

- **停止的直接原因**：高压现场 `PID 409` 的 RSS、匿名内存与进程 swap 很高，系统 swap 接近 8 GiB 上限；用户主动发送 `SIGINT`，避免 OOM；
- **底层根因状态**：还没有定位到唯一源码行或唯一组件；
- **当前已完成处置**：保留恢复 checkpoint，将 `save_interval` 从 200 改为 5000，启动 `PID 441` 的受控恢复，建立连续内存 / swap 监控；
- **单 writer 的真实状态**：当前通过 `pgrep`、`ps` 与 `screen -ls` 核验当下只有一个训练进程；未部署 `flock` 等原子锁，因此不是机制级保证；
- **当前不能越界的结论**：短窗口 `VmSwap=0` 是积极信号，但不能代表长期稳定，更不能代表 Dense 768 预训练已完成。
