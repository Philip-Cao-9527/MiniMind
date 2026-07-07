# MiniMind 今晚手动长跑 Runbook（2026-07-07）

本文档只服务今晚这条 `768×8 Dense` 主线：先手动启动真实预训练，再视产物与状态决定是否进入 SFT，最后做最小推理验证。本文不表示训练已经完成，也不表示下面的运行参数已经在本机长时间验证通过。

## 0. 事实边界与来源

- 本地仓库事实：
  当前项目根目录是 [MiniMind](../README.md)，本地已存在 `dataset/lm_dataset.py`、`model/model_minimind.py`、`trainer/train_pretrain.py`、`trainer/trainer_utils.py`，其中预训练主链与上游对应文件逐字一致。本轮额外同步的入口文件是 [trainer/train_full_sft.py](../trainer/train_full_sft.py)、[eval_llm.py](../eval_llm.py) 与 [model/model_lora.py](../model/model_lora.py)。
- 上游引用事实：
  当前上游引用仓库是 `../../../references/minimind`，本轮关键依据来自 [上游 README.md](../../../references/minimind/README.md)、[上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py)、[上游 train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)、[上游 trainer_utils.py](../../../references/minimind/trainer/trainer_utils.py) 与 [上游 eval_llm.py](../../../references/minimind/eval_llm.py)。
- 学习材料：
  [源码导览](./minimind-source-guide.md)、[SFT 理解记录](./sft-from-zero-minimind.md)、[KV Cache 理解记录](./kv-cache-from-zero.md) 和 `../../../references/learn-minimind` 只用于帮助理解调用链，不作为上游默认行为来源。

## 1. 上游默认值 vs 本机本轮采用值

### 1.1 上游默认模型结构与训练脚本默认值

当前上游 `Dense` 主线默认结构来自 [上游 train_pretrain.py](../../../references/minimind/trainer/train_pretrain.py) 和 [上游 train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py)：

| 阶段 | hidden_size | num_hidden_layers | batch_size | max_seq_len | accumulation_steps | dtype | save_interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 预训练默认 | 768 | 8 | 32 | 340 | 8 | bfloat16 | 1000 |
| Full SFT 默认 | 768 | 8 | 16 | 768 | 1 | bfloat16 | 1000 |

补充默认值：

- 预训练默认数据路径：`../dataset/pretrain_t2t_mini.jsonl`
- SFT 默认数据路径：`../dataset/sft_t2t_mini.jsonl`
- 预训练默认 `from_weight='none'`
- Full SFT 默认 `from_weight='pretrain'`
- 两条训练脚本都支持 `--from_resume 1`

### 1.2 本机本轮最终采用的 768×8 Dense 与运行参数

本轮只锁定 `768×8 Dense` 主线，不改保存命名逻辑，不为未来架构对照预埋命名改造。

| 阶段 | 模型结构 | batch_size | max_seq_len | accumulation_steps | dtype | num_workers | save_interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 今晚预训练 | 768×8 Dense | 2 | 128 | 8 | bfloat16 | 2 | 200 |
| 后续 SFT 起始建议 | 768×8 Dense | 1 | 256 | 4 | bfloat16 | 2 | 100 |

补充说明：

- 今晚预训练继续保留上游默认 `learning_rate=5e-4`、`from_weight='none'`；首次正式启动使用 `from_resume=0`，只有当“本轮同一预训练阶段”已经产出对应 `pretrain_768_resume.pth`，且训练中断后需要继续时，才改为 `from_resume=1`
- 后续 SFT 继续保留上游默认 `learning_rate=1e-5`、`from_weight='pretrain'`；首次正式启动使用 `from_resume=0`，只有当“本轮同一 Full SFT 阶段”已经产出对应 `full_sft_768_resume.pth`，且训练中断后需要继续时，才改为 `from_resume=1`
- 不要让历史遗留 checkpoint 被误用为一次新实验的恢复来源
- 本轮没有修改 [train_pretrain.py](../trainer/train_pretrain.py) 或 [train_full_sft.py](../trainer/train_full_sft.py) 的保存命名逻辑

## 2. 权重命名约定与阶段衔接

### 2.1 当前上游命名约定

当前上游命名逻辑来自 [trainer/train_pretrain.py](../trainer/train_pretrain.py)、[trainer/train_full_sft.py](../trainer/train_full_sft.py) 和 [trainer_utils.py](../trainer/trainer_utils.py)：

- Dense 预训练默认权重：`out/pretrain_768.pth`
- Dense Full SFT 默认权重：`out/full_sft_768.pth`
- 文件名由 `save_weight`、`hidden_size` 和是否 `MoE` 决定
- 文件名不包含 `num_hidden_layers`
- `checkpoints/` 下会同时存在同维度命名的普通 checkpoint 和 `_resume` checkpoint
  - 例如：`checkpoints/pretrain_768.pth`
  - 例如：`checkpoints/pretrain_768_resume.pth`
  - 例如：`checkpoints/full_sft_768.pth`
  - 例如：`checkpoints/full_sft_768_resume.pth`

### 2.2 预训练 -> SFT -> 推理验证的权重衔接关系

本轮主链按下面的关系理解：

```text
pretrain
  -> out/pretrain_768.pth
  -> Full SFT 默认 from_weight='pretrain'
  -> out/full_sft_768.pth
  -> eval_llm.py --weight full_sft
```

对应到手动阶段：

- 预训练产物：
  `out/pretrain_768.pth`
- 预训练续训状态：
  `checkpoints/pretrain_768_resume.pth`
- Full SFT 起始权重：
  `--from_weight pretrain` 会经 [trainer_utils.py](../trainer/trainer_utils.py) 中的 `init_model(..., save_dir='../out')` 直接从 `out/pretrain_768.pth` 加载
- Full SFT 产物：
  `out/full_sft_768.pth`
- Full SFT 续训状态：
  `checkpoints/full_sft_768_resume.pth`
- 推理验证：
  - 预训练权重走 `python eval_llm.py --load_from model --weight pretrain`
  - SFT 权重走 `python eval_llm.py --load_from model --weight full_sft`

## 3. 数据下载与落盘位置

上游 README 推荐快速复现 Zero 模型的数据组合是 [上游 README.md](../../../references/minimind/README.md) 中给出的：

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

推荐下载位置：

```text
/home/harry/projects/MiniMind/dataset/pretrain_t2t_mini.jsonl
/home/harry/projects/MiniMind/dataset/sft_t2t_mini.jsonl
```

上游 README 中给出的参考体积：

- `pretrain_t2t_mini.jsonl`：约 `1.2GB`
- `sft_t2t_mini.jsonl`：约 `1.6GB`

手动下载建议：

1. 打开 [上游 README.md](../../../references/minimind/README.md) 中指向的数据集页面链接，优先只下载这两个文件。
2. 下载完成后，把文件放到 `dataset/` 目录下，不要改文件名。
3. 手动检查：

```bash
cd /home/harry/projects/MiniMind
ls -lh dataset/pretrain_t2t_mini.jsonl dataset/sft_t2t_mini.jsonl
```

如果你更想用命令行下载，可以自行选择对应平台的 CLI，但本文不把任何特定下载工具写成默认前提；今晚最重要的是文件名与落盘路径准确。

## 4. 手动运行前检查

先在项目根目录确认环境：

```bash
cd /home/harry/projects/MiniMind
git status --short --branch
./.venv/bin/python -V
./.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")
print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ls -lh dataset/pretrain_t2t_mini.jsonl dataset/sft_t2t_mini.jsonl
```

再确认会话工具：

```bash
command -v screen
```

## 5. 今晚预训练手动命令

为避免把 `screen` 命令和训练命令混在同一段里误贴，下面改成可整段复制的后台启动写法；仍然沿用上游默认相对路径：

启动前先只读检查是否已经有同名预训练主进程：

```bash
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'
```

再只读检查同名 `screen` 会话状态：

```bash
screen -ls
```

这里要明确：训练进程检查与 `screen` 会话检查缺一不可。

- 如果 `screen -ls` 已显示同名 `minimind-pretrain` 会话，不要直接重复执行下面的启动命令。
- 如果该会话确认仍对应本轮正在运行的训练，使用 `screen -r minimind-pretrain` 恢复，不得重新启动。
- 如果已确认没有关联训练进程，且它只是空壳或 `dead` session，再由你决定是否后续手动清理；本文不提供自动 `screen -wipe`、自动 kill session 或自动删除 session 的命令。

只有在确认没有重复主训练，且没有需要直接恢复的同名 `screen` 会话后，再执行下面这一整段：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs ../out ../checkpoints
screen -dmS minimind-pretrain bash -lc '
cd /home/harry/projects/MiniMind/trainer
echo "STY=${STY:-<empty>}"
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
  --from_resume 0 \
  2>&1 | tee ../experiments/logs/pretrain-$(date +%F-%H%M%S).log
'
screen -r minimind-pretrain
```

MiniMind 参数名保留为 `--use_wandb`，但当前实际使用 SwanLab；训练启动后应把 `loss`、`logits_loss`、`learning_rate` 等指标记录到你已登录的 SwanLab 实验。

首次正式启动使用 `--from_resume 0`。只有当本轮预训练阶段已经产出同名 `pretrain_768_resume.pth`，并且训练因中断需要继续时，才把该参数改为 `--from_resume 1`；不要让旧 checkpoint 被误用为本轮首次训练的恢复来源。

这里把日志文件名精确到秒，是为了避免同一分钟误操作重复启动时，新的 `tee` 覆盖或混淆同名日志证据。

上面这段命令里显式写出了 `--save_weight pretrain`、`--save_dir ../out` 和 `--from_weight none`，目的是把这条主线依赖的命名约定直接固定在可复制命令里，减少后续手改参数时把保存前缀或加载来源写错的风险。

这里不单独跑一个短训再重启。真实训练进程从一开始就直接是最终命令；前若干 step 只作为嵌入式门禁观察窗口。

### 嵌入式门禁判定

前若干 step 重点观察：

- 数据是否从 `dataset/pretrain_t2t_mini.jsonl` 正常读入
- `nvidia-smi` 中显存是否稳定，没有立刻 OOM
- 日志中的 loss 是否是有限值，不是 `nan` / `inf`
- 训练进程是否跨过 `accumulation_steps=8` 的首个参数更新边界，且没有异常退出
- 日志是否持续输出，而不是在前几步卡死或直接中断

当前 `save_interval=200`，因此 `out/pretrain_768.pth` 与 `checkpoints/pretrain_768_resume.pth` 正常情况下应在 `micro-step 200` 或 epoch 结束时才出现；它们不是前几步的即时门禁条件。等训练推进到对应保存时机后，再把这两个文件作为后续保存验证项检查。

建议观察命令：

```bash
nvidia-smi
ls -lh ../out ../checkpoints
tail -n 50 ../experiments/logs/pretrain-*.log
```

满足条件后，不中断、不重启，直接从 `screen` detach：

```text
Ctrl+A D
```

之后用下面的命令恢复会话：

```bash
screen -r minimind-pretrain
```

## 6. 后续 Full SFT 手动命令

只有在你已经人工确认预训练权重与状态正常后，才进入 SFT。这里同样改成可整段复制的后台启动写法：

启动前先只读检查是否已经有同名 Full SFT 主进程：

```bash
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_full_sft.py'
```

再只读检查同名 `screen` 会话状态：

```bash
screen -ls
```

这里也要明确：训练进程检查与 `screen` 会话检查缺一不可。

- 如果 `screen -ls` 已显示同名 `minimind-sft` 会话，不要直接重复执行下面的启动命令。
- 如果该会话确认仍对应本轮正在运行的训练，使用 `screen -r minimind-sft` 恢复，不得重新启动。
- 如果已确认没有关联训练进程，且它只是空壳或 `dead` session，再由你决定是否后续手动清理；本文不提供自动 `screen -wipe`、自动 kill session 或自动删除 session 的命令。

只有在确认没有重复主训练，且没有需要直接恢复的同名 `screen` 会话后，再执行下面这一整段：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs ../out ../checkpoints
screen -dmS minimind-sft bash -lc '
cd /home/harry/projects/MiniMind/trainer
echo "STY=${STY:-<empty>}"
../.venv/bin/python train_full_sft.py \
  --save_dir ../out \
  --save_weight full_sft \
  --epochs 1 \
  --batch_size 1 \
  --max_seq_len 256 \
  --accumulation_steps 4 \
  --num_workers 2 \
  --log_interval 10 \
  --save_interval 100 \
  --use_wandb \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --from_weight pretrain \
  --from_resume 0 \
  2>&1 | tee ../experiments/logs/full-sft-$(date +%F-%H%M%S).log
'
screen -r minimind-sft
```

MiniMind 参数名保留为 `--use_wandb`，但当前实际使用 SwanLab；训练启动后应把 `loss`、`logits_loss`、`learning_rate` 等指标记录到你已登录的 SwanLab 实验。

首次正式启动使用 `--from_resume 0`。只有当本轮 Full SFT 阶段已经产出同名 `full_sft_768_resume.pth`，并且训练因中断需要继续时，才把该参数改为 `--from_resume 1`；不要让旧 checkpoint 被误用为本轮首次训练的恢复来源。

这里把日志文件名精确到秒，是为了避免同一分钟误操作重复启动时，新的 `tee` 覆盖或混淆同名日志证据。

上面这段命令里显式写出了 `--save_weight full_sft`、`--save_dir ../out` 和 `--from_weight pretrain`，目的是把 SFT 与预训练的衔接前缀固定在可复制命令里，减少后续因为默认值或手动改动而把链路接错的风险。

进入 SFT 前至少确认：

- `out/pretrain_768.pth` 已存在；这是 Full SFT 的 `--from_weight pretrain` 通过 [trainer_utils.py](../trainer/trainer_utils.py) 中 `init_model(..., save_dir='../out')` 直接加载的起始权重
- `checkpoints/pretrain_768_resume.pth` 如已存在，可作为预训练断点恢复与保存闭环的验证证据；建议检查，但它不是 Full SFT 直接加载预训练权重的必需输入
- 预训练日志中 loss 有限

## 7. 最小推理验证命令

预训练权重最小验证：

```bash
cd /home/harry/projects/MiniMind
ls -lh out/pretrain_768.pth
./.venv/bin/python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --max_new_tokens 128
```

SFT 权重最小验证：

```bash
cd /home/harry/projects/MiniMind
ls -lh out/full_sft_768.pth
./.venv/bin/python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda \
  --max_new_tokens 256
```

## 8. 本机训练持续性边界

本轮训练承载固定为本机 `Windows 11 + WSL2 Ubuntu + 本机 GPU`。这里必须区分清楚：

- `screen` / `tmux` / `nohup` 只能解决终端断开、SSH 断开或你关闭当前 shell 的问题
- 它们不能解决 Windows 关机、休眠、断电、WSL 被退出或宿主机崩溃
- 如果训练真跑在本机 GPU 上，只要宿主机停了，训练就停
- 只有训练实际跑在远端主机上时，本地电脑关机后远端训练才可能继续

因此，今晚如果选择本机长跑，前提就是：

- Windows 不关机
- 不休眠
- WSL 不被手动关闭
- 电源稳定

## 9. 未来同 hidden_size、不同 num_hidden_layers 的防覆盖约定

当前保存命名逻辑不包含 `num_hidden_layers`，而 [trainer_utils.py](../trainer/trainer_utils.py) 中 `init_model` 默认从 `../out` 按 `from_weight + hidden_size (+ moe)` 加载前序权重。

因此，未来如果你要做“相同 `hidden_size`、不同 `num_hidden_layers`”的对照：

- 优先通过不同的 `save_weight` 前缀区分整条 `pretrain -> SFT` 链路
- 不要只改 `save_dir`

推荐思路：

```text
pretrain_l8 -> out/pretrain_l8_768.pth -> full_sft_l8
pretrain_l6 -> out/pretrain_l6_768.pth -> full_sft_l6
```

不推荐只改目录的原因：

- 当前 `init_model` 默认从 `../out` 加载前序权重
- 只改 `save_dir` 容易让后续 `from_weight`、推理验证和整条链路的语义变得不清晰
- 通过 `save_weight` 前缀区分，能把预训练、SFT、推理验证的命名保持成一条可追溯链路

本轮不会提前为这个未来对照去改 [train_pretrain.py](../trainer/train_pretrain.py)、[train_full_sft.py](../trainer/train_full_sft.py) 或 [trainer_utils.py](../trainer/trainer_utils.py)。

## 10. 当前未验证项

截至本文写入时，下面这些都还没有被本轮真实长跑验证：

- 真实显存峰值
- 吞吐
- 完整 epoch 时间
- loss 曲线
- 断点恢复
- 生成质量

更保守的说法是：当前只完成了上游默认行为核验、主链入口同步、运行参数定稿和手动命令准备；真实训练表现仍以后续你的手动运行结果为准。

## 11. 实战避坑：同名训练并发会竞争 checkpoint 临时文件

本轮曾误启动两条参数相同的预训练主程序。两条命令都使用了 `--save_weight pretrain`，并沿用默认 `../out` 与 `../checkpoints`，因此实际共享了下面这些保存目标：

- `out/pretrain_768.pth`
- `checkpoints/pretrain_768.pth`
- `checkpoints/pretrain_768_resume.pth`
- `checkpoints/pretrain_768_resume.pth.tmp`

这次并发训练的直接诱因，更合理的复盘不是模型、CUDA、SwanLab 或 PyTorch DataLoader 本身，而是 `screen` 的使用方式和命令粘贴方式导致了终端归属混乱。本轮曾把 `screen -S minimind-pretrain` 和后续较长的 `python train_pretrain.py ...` 命令放在同一大段里粘贴执行；但 `screen -S minimind-pretrain` 不是普通“执行完就返回”的命令，它会立即进入新的虚拟终端并接管当前输入。因此，后续那段训练命令到底落在 `screen` 内部 shell，还是落在外层 VS Code WSL 终端，容易变得不清晰。

结合 [pretrain-2026-07-07-2040.log](../experiments/logs/pretrain-2026-07-07-2040.log) 与 [pretrain-2026-07-07-2051.log](../experiments/logs/pretrain-2026-07-07-2051.log) 各自创建了不同的 SwanLab run，再结合当前只剩一条主训练进程加两个 DataLoader worker 的进程树，本轮最合理的复盘是：`screen` 命令与训练命令混合粘贴导致重复启动训练，随后两条同名训练竞争 checkpoint 临时文件。这里不能写成 `screen` 有 bug，也不能把它还原成百分百精确的按键历史；更稳妥的事实边界是，这次实际表现为出现了两个 SwanLab run，第一条训练后来在保存 checkpoint 时与第二条训练竞争同名 `.tmp` 文件并中断。

第一条训练的实际失败证据来自 [pretrain-2026-07-07-2040.log](../experiments/logs/pretrain-2026-07-07-2040.log)。日志显示它在 [trainer_utils.py](../trainer/trainer_utils.py) 的 `lm_checkpoint` 中执行 `os.replace(resume_tmp, resume_path)` 时退出，对应报错是：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'../checkpoints/pretrain_768_resume.pth.tmp'
-> '../checkpoints/pretrain_768_resume.pth'
```

[trainer_utils.py](../trainer/trainer_utils.py) 当前保存逻辑也是先写 `resume_path + '.tmp'`，再执行原子替换，因此这里的结论只能谨慎写成：结合两条训练共享固定 `.tmp` 路径这一事实，该错误与并发保存时的文件竞争一致。至少从这份日志看，这次退出不是 CUDA OOM、loss NaN、模型收敛失败，也不是 SwanLab 自身故障。

这里要明确区分两类“看起来像有多个 Python 进程”的情况：

- 两条独立训练主程序：危险。它们互不属于父子关系，会创建多个 SwanLab run，并竞争同名权重、checkpoint 和 `.tmp` 文件。
- 一个主训练进程 + `num_workers=2` 产生的两个 DataLoader worker：正常。这不是重复训练，也不是“三次训练”，只是一个主训练进程带两个数据加载子进程。

```text
主训练进程
├─ DataLoader worker 1
└─ DataLoader worker 2
```

启动前检查建议固定执行：

```bash
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_pretrain.py'
```

判断原则：

- 如果已经存在一个主训练进程及其 DataLoader 子进程，不要再次执行同一条训练命令。
- 不要只看 `pgrep -af train_pretrain.py` 有几行，也不要只用 `pgrep` 输出行数判断是否重复训练，因为 worker 也会带出同样的命令行。
- 关键是看父子关系：DataLoader worker 的 `PPID` 应指向同一个主训练进程，而不是再出现第二个独立主进程。

以后正确启动 `screen` 的操作约定：

1. 先进入项目训练目录。
2. 执行 `screen -S minimind-pretrain`。
3. 进入 `screen` 后先运行 `echo "STY=${STY:-<empty>}"`。
4. 只有看到 `STY` 不是 `<empty>`，才说明当前确实在 `screen` 内。
5. 然后再单独粘贴 `python train_pretrain.py ...` 训练命令。
6. 确认日志输出和 SwanLab run 创建后，再用 `Ctrl+A`，松开，再按 `D` detach。

补充判断：

- 不要把 `screen -S ...` 和训练命令写在同一段里一次性粘贴。
- 如果 `Ctrl+A D` 之后外层终端仍继续刷 `Epoch` 日志，说明训练很可能没有真正挂在 `screen` 里；这时要立即停止新增启动动作，先只读检查进程树，不能再重复执行训练命令。

为避免“先进入 `screen`，再粘贴长命令”的终端归属混乱，前面的预训练与 SFT 命令都已经改成了让 `screen` 直接在后台拉起训练命令的写法，例如：

```bash
screen -dmS minimind-pretrain bash -lc 'cd /home/harry/projects/MiniMind/trainer && ../.venv/bin/python train_pretrain.py ... 2>&1 | tee ../experiments/logs/pretrain-$(date +%F-%H%M%S).log'
```

这个写法能避免“先进入 `screen`，再粘贴长命令”的终端归属混乱。但要明确：本轮当前训练已经在运行，不要现在执行这条后台启动命令；它只作为后续恢复训练或下一轮实验的推荐写法，不是本轮已执行事实。

如果要并行跑不同实验或做结构对照，必须从命名上把整条链路隔开：

- 同时运行不同实验时，必须使用不同的 `--save_weight`，例如 `pretrain_l8`、`pretrain_l6`。
- 后续 SFT 的 `--from_weight` 也必须对应各自的预训练前缀。
- 不允许两条训练共用同一个 `save_weight + hidden_size` 组合。
- 本轮当前只保留 `768×8 Dense` 单线训练；不要为了未来对照去修改训练脚本的保存逻辑。

当前状态说明：

- 发生竞争后，第一条训练已退出。
- 当前只保留一条主训练进程及其两个 DataLoader worker。
- 当前正式权重、resume checkpoint 与 SwanLab 记录，应以后续仍在运行的这一条训练为准。
- 这里不能写成“训练已经完成”；最终效果仍以后续真实运行结果为准。
