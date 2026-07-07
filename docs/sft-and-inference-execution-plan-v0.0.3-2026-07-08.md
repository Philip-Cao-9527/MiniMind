# Dense 768 Full SFT 与训练后推理验证执行计划（v0.0.3，2026-07-08）

## 1. 目标、范围与明确不做的事情

本计划只服务后续 `Dense 768` full SFT 与训练后推理验证的人工执行阶段。当前阶段只完成代码准备、参数方案、手动命令、监控命令、验收矩阵和证据清单沉淀。

本计划明确不做以下事情：

- 不启动 `trainer/train_full_sft.py`
- 不启动 `eval_llm.py` 的真实推理
- 不加载模型权重，不创建新的 checkpoint、日志、SwanLab run 或 screen 会话
- 不生成 `docs/fix-report-v0.0.3-dense-768-full-sft-and-inference-validation-2026-07-08.md`
- 不提前修改 [README.md](../README.md) 的版本号

## 2. 证据来源与冲突优先级

证据优先级按以下顺序执行：

1. 本轮只读核验到的本机代码、Git 状态、权重、checkpoint、日志、数据集和系统状态
2. [fix-report-v0.0.2-dense-768-pretrain-completion-2026-07-08.md](fix-report-v0.0.2-dense-768-pretrain-completion-2026-07-08.md)
3. [minimind-initial-pretrain-full-retrospective-updated.md](minimind-initial-pretrain-full-retrospective-updated.md)
4. [minimind-pretrain-resume-incident-20260708.md](minimind-pretrain-resume-incident-20260708.md)
5. [pretrain-sft-manual-runbook-2026-07-07.md](pretrain-sft-manual-runbook-2026-07-07.md)
6. `/home/harry/references/minimind`
7. 上游公开 README 或其他公开材料
8. `/home/harry/references/learn-minimind`

本计划同时引用以下仓库内长期规则或总览：

- [AGENTS.md](../AGENTS.md)
- [README.md](../README.md)

说明：

- 当前关键代码文件 `trainer/train_full_sft.py`、`trainer/trainer_utils.py`、`dataset/lm_dataset.py`、`eval_llm.py` 与 `/home/harry/references/minimind` 同名文件 `diff` 为空，因此本轮不复制上游文件。
- 当前 worktree 中 [pretrain-sft-manual-runbook-2026-07-07.md](pretrain-sft-manual-runbook-2026-07-07.md) 处于删除态；本计划仅把它作为历史参考路径，不把它写成本轮已恢复文件。

## 3. Dense 768 预训练前置条件

本计划建立在以下本机已验证前提上：

- 最终预训练权重：`/home/harry/projects/MiniMind/out/pretrain_768.pth`
- 最终 resume checkpoint：`/home/harry/projects/MiniMind/checkpoints/pretrain_768_resume.pth`
- 预训练最终到达：`Epoch:[1/1](635119/635119)`
- 当前无 `train_pretrain.py` 或 `train_full_sft.py` 活跃进程
- 当前 `screen -ls` 返回 `No Sockets found`

full SFT 首次启动必须使用：

- `--from_weight pretrain`
- `--from_resume 0`

原因：

- `--from_weight pretrain` 会按当前 `trainer_utils.init_model()` 的真实实现加载 `../out/pretrain_768.pth`
- `--from_resume 0` 表示本轮 full SFT 首次启动不从已有 full SFT resume 状态恢复

## 4. Full SFT 真实调用链

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
10. resume checkpoint 保存到 `../checkpoints/full_sft_768_resume.pth`

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
- `save_interval=5000` 只可写成“参考 pretrain 受控恢复中的低频保存经验”，不能写成“已被 full SFT 验证安全”。
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
- 2 epoch 的预计 optimizer update 数：`301906`

## 7. 真实 SFT 数据、tokenizer、chat template、labels 与 `-100` 边界

当前 `SFTDataset` 的真实数据契约：

- 顶层字段：`conversations`
- 每条对话消息字段：`role`、`content`、`reasoning_content`、`tools`、`tool_calls`

当前标签语义：

- prompt 由 `apply_chat_template(...)` 生成
- labels 初始全为 `-100`
- 只把 assistant 片段与其结尾 `eos` 对应位置改写成真实 token id
- user、system、模板控制 token、padding 位置不参与 loss

因此必须区分三种量：

- 模型前向实际长度：固定 `384`
- 样本中真实有效内容长度：有限样本平均 `356.68`
- 实际参与监督 loss 的 assistant token：有限样本平均 `281.89`

## 8. 权重、checkpoint、`.tmp`、日志与 SwanLab 路径

当前代码和后续手动命令下的真实路径如下：

- full SFT 普通权重：`../out/full_sft_768.pth`
- full SFT 普通 checkpoint：`../checkpoints/full_sft_768.pth`
- full SFT resume checkpoint：`../checkpoints/full_sft_768_resume.pth`
- full SFT 普通 checkpoint 临时文件：`../checkpoints/full_sft_768.pth.tmp`
- full SFT resume 临时文件：`../checkpoints/full_sft_768_resume.pth.tmp`
- 手动训练日志：`../experiments/logs/full-sft-dense-768-e2-$(date +%F-%H%M%S).log`
- SwanLab project：`MiniMind-Full-SFT`
- SwanLab run 名默认格式：`MiniMind-Full-SFT-Epoch-2-BatchSize-1-LearningRate-1e-05`

注意：

- `../checkpoints/full_sft_768_resume.pth` 是代码内部固定路径，不是 CLI 参数。
- 当前没有 `flock` 或其他机制级互斥锁；单 writer 只能靠启动前和运行中的人工只读检查确认。

## 9. WSL 主机内存 / swap、GPU 显存与单 writer 风险

当前机器边界：

- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB`
- WSL 当前资源上限：RAM `10GB`，swap `8GB`

风险说明：

- `max_seq_len=384` 的实际前向长度固定为 `384`，因此显存与 host 内存压力应按固定长度理解，不能用 `356.68` 平均有效长度替代。
- `pin_memory=True` 当前写死在训练脚本里，本轮不改；它可能增加 host 侧 pinned memory 压力。
- 历史上 `.tmp` checkpoint 竞争事故与 host RSS / swap 高压是两类独立问题，不能混写。
- `save_interval=5000` 只能降低保存频率，不等于根治 host 内存问题。

## 10. 用户手动启动前的只读检查

后续真实启动前，先在项目根目录执行只读检查：

```bash
cd /home/harry/projects/MiniMind
git status --short --branch
screen -ls || true
ps -eo pid,ppid,tty,etime,cmd | grep -E '[t]rain_(pretrain|full_sft)[.]py' || true
stat -c '%n | size=%s bytes | mtime=%y' out/pretrain_768.pth checkpoints/pretrain_768_resume.pth
du -h dataset/sft_t2t_mini.jsonl
wc -l dataset/sft_t2t_mini.jsonl
free -h
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
find out checkpoints experiments/logs -maxdepth 2 -type f \( -name '*full_sft*' -o -name '*sft*' \) -printf '%p | %s bytes | %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort || true
```

只有确认没有活跃 writer、没有需要恢复的旧 full SFT 会话，且不会覆盖已有工件后，才允许手动启动。

## 11. 用户手动启动 full SFT 的完整命令

以下命令只写入计划，不在当前阶段执行：

```bash
cd /home/harry/projects/MiniMind/trainer
mkdir -p ../experiments/logs ../out ../checkpoints
screen -dmS minimind-full-sft-dense768-e2 bash -lc '
set -o pipefail
cd /home/harry/projects/MiniMind/trainer || exit 1
../.venv/bin/python -u train_full_sft.py \
  --save_dir ../out \
  --save_weight full_sft \
  --epochs 2 \
  --batch_size 1 \
  --max_seq_len 384 \
  --accumulation_steps 6 \
  --num_workers 0 \
  --log_interval 20 \
  --save_interval 5000 \
  --use_wandb \
  --wandb_project MiniMind-Full-SFT \
  --dtype bfloat16 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --from_weight pretrain \
  --from_resume 0 \
  2>&1 | tee ../experiments/logs/full-sft-dense-768-e2-$(date +%F-%H%M%S).log
'
```

## 12. 训练期间人工监控命令

训练期间建议只读监控以下信号：

```bash
screen -ls
ps -eo pid,ppid,tty,etime,%mem,rss,cmd | grep '[t]rain_full_sft.py'
tail -n 80 /home/harry/projects/MiniMind/experiments/logs/full-sft-dense-768-e2-*.log
free -h
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv
stat -c '%n | size=%s bytes | mtime=%y' /home/harry/projects/MiniMind/out/full_sft_768.pth /home/harry/projects/MiniMind/checkpoints/full_sft_768_resume.pth 2>/dev/null
grep -E 'VmRSS|VmSwap|RssAnon|RssFile' /proc/<训练PID>/status
```

重点观察：

- 日志是否持续推进
- 是否出现 `Traceback`、`RuntimeError`、`CUDA out of memory`
- `VmSwap`、系统 swap、`MemAvailable` 是否持续恶化
- checkpoint 的 `mtime` 是否按预期更新
- 是否始终只有一个 full SFT writer

## 13. 异常时应保留的证据与禁止动作

出现异常时优先保留：

- 训练日志尾部
- `screen -ls`
- 训练进程 `ps` 输出
- `free -h`
- `nvidia-smi`
- `/proc/<训练PID>/status` 中的 `VmRSS`、`VmSwap`、`RssAnon`、`RssFile`
- `out/full_sft_768.pth` 与 `checkpoints/full_sft_768_resume.pth` 的 `stat`

禁止动作：

- 不要删除、覆盖、移动已有预训练或 full SFT 权重
- 不要在异常现场重复启动第二个 full SFT 进程
- 不要把 `save_interval=5000` 写成内存问题已经被彻底解决
- 不要把人工确认到的“当前只有一个 writer”写成机制级保证

## 14. SFT 结束后的验收清单

真实 full SFT 完成后，至少执行以下验收：

```bash
screen -ls || true
ps -eo pid,ppid,tty,etime,cmd | grep '[t]rain_full_sft.py' || true
stat -c '%n | size=%s bytes | mtime=%y' out/full_sft_768.pth checkpoints/full_sft_768_resume.pth
tail -n 120 experiments/logs/full-sft-dense-768-e2-*.log
rg -n 'Traceback|RuntimeError|CUDA out of memory|OutOfMemory|KeyboardInterrupt|FileNotFoundError|Error:' experiments/logs/full-sft-dense-768-e2-*.log || true
```

验收目标：

- 训练自然结束或按计划停止
- 最终权重存在
- 最终 resume checkpoint 存在
- 日志中能确认 2 epoch 的最终进度
- 未把未验证内容扩写成“能力已证明”

## 15. Full SFT 权重存在后的固定 prompt 推理命令

以下命令只写入计划，不在当前阶段执行。默认使用当前已补足 CLI 的 `eval_llm.py`。

固定 prompt 基线命令：

```bash
cd /home/harry/projects/MiniMind
./.venv/bin/python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft \
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
  --open_thinking 0
```

预训练权重对照命令：

```bash
cd /home/harry/projects/MiniMind
./.venv/bin/python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
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
  --max_new_tokens 128
```

## 16. cache on/off、EOS、采样、history、权重来源验证矩阵

cache on/off 对照：

- 固定 prompt：`请用三句话介绍你自己。`
- 固定 `seed=42`
- 固定 `do_sample=0`
- 固定 `max_new_tokens=128`
- 比较最终文本是否一致
- 记录 `generated_tokens`、`ended_with_eos` 与耗时

cache on：

```bash
./.venv/bin/python eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 0 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128
```

cache off：

```bash
./.venv/bin/python eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 0 --use_cache 0 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128
```

EOS 验证：

- 观察 `ended_with_eos`
- 观察 `generated_tokens`
- 结合输出文本确认是否自然结束

采样验证：

```bash
./.venv/bin/python eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 1 --use_cache 1 --top_k 20 --top_p 0.90 --temperature 0.70 --max_new_tokens 128
./.venv/bin/python eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '请用三句话介绍你自己。' --seed 42 --do_sample 1 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128
```

history 验证：

```bash
./.venv/bin/python eval_llm.py --load_from model --save_dir out --weight full_sft --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --prompt '你好。' --prompt '请重复上一句问候并继续自我介绍。' --seed 42 --do_sample 0 --use_cache 1 --top_k 50 --top_p 0.95 --temperature 0.85 --max_new_tokens 128 --historys 2
```

权重来源验证：

- `weight=pretrain` 与 `weight=full_sft` 分别执行同一固定 prompt
- 记录 `eval_llm.py` 输出的 `model_path`
- 不把一次随机输出写成能力证明

## 17. 生成 `v0.0.3` fix-report 前必须收集的真实证据

未来允许生成 `docs/fix-report-v0.0.3-dense-768-full-sft-and-inference-validation-2026-07-08.md` 的前提是以下证据齐全：

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

## 18. 未验证边界

- 当前没有真实 full SFT 显存峰值证据
- 当前没有真实 full SFT host RSS / swap 曲线证据
- 当前没有真实 2 epoch 耗时证据
- 当前没有真实 full SFT 训练质量与推理质量证据
- 当前没有真实 cache on/off、一致性、EOS、采样、history 验证结果
- 当前没有机制级 single-writer 锁
