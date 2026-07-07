## 体检结论：可以进入唯一一次恢复训练

恢复前的硬条件都已满足：

+ 当前 WSL 发行版是 `Ubuntu-24.04`，项目路径与 `.venv` 正常。
+ CUDA、RTX 5060 Laptop GPU、BF16 均正常；`nvidia-smi` 进程表为空，当前没有 Python 训练进程。
+ 没有残留 `train_pretrain.py` 主进程，也没有残留 screen session。
+ `checkpoints/pretrain_768_resume.pth` 可正常读取，仍是 `epoch=0, step=73000`，并且含 model、optimizer、scaler 与 SwanLab run id。
+ 两个真实数据文件仍在。
+ 最新日志已经跑到 `73100`，但最后可恢复 checkpoint 是 `73000`，因此恢复后从约 `73001` 往后继续是正常的，不应期待精确从日志最后一行 `73100` 接上。

还有一个需要**保留但暂不处理**的历史痕迹：

```text
checkpoints/pretrain_768.pth.tmp
```

它目前是 0 字节，时间为 `21:41:23`。这说明此前曾出现一次未完整结束的普通权重临时写入；但恢复训练实际读取的是健康的：

```text
checkpoints/pretrain_768_resume.pth
```

当前本地源码也表明，`--from_resume 1` 读取的是 `_resume.pth`，而下一次 checkpoint 会重新写入同名 `.tmp` 后再替换正式文件。因此现在**不要删除它，也不要移动任何 checkpoint**。它不能单独证明当时为何停训。

上游 MiniMind 也明确将 `--from_resume 1` 作为恢复完整 checkpoint、训练步数和可视化 run 连续性的入口；但本次实际行为仍以你刚核验的本地代码与 checkpoint 为准。([GitHub](https://github.com/jingyaogong/minimind))

------

# 第二阶段：启动唯一一条预训练恢复任务

下面命令在 **WSL Ubuntu Bash** 执行。

它会：

+ 进入 `trainer/`；
+ 用唯一 session 名 `minimind-pretrain-resume`；
+ 使用绝对 `.venv` Python，避免 screen 非交互环境没有激活 direnv；
+ 写入一个精确到秒的新日志；
+ 从 `step=73000` 恢复；
+ 不创建第二条训练；
+ 不开始 SFT；
+ 保留并继续使用当前 SwanLab run id。

```bash
cd /home/harry/projects/MiniMind

screen -dmS minimind-pretrain-resume bash -lc '
set -uo pipefail

cd /home/harry/projects/MiniMind/trainer || exit 1
mkdir -p ../experiments/logs

log="../experiments/logs/pretrain-resume-$(date +%F-%H%M%S).log"

echo "[$(date -Is)] launch: pretrain resume from checkpoint" | tee -a "$log"

/home/harry/projects/MiniMind/.venv/bin/python -u train_pretrain.py \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 2 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 200 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 128 \
  --use_moe 0 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight none \
  --from_resume 1 \
  --save_weight pretrain \
  --save_dir ../out \
  --wandb_project MiniMind-Pretrain \
  --use_compile 0 \
  --use_wandb \
  2>&1 | tee -a "$log"

status="${PIPESTATUS[0]}"
echo "[$(date -Is)] train process exited with status=${status}" | tee -a "$log"
exit "$status"
'
```

这里显式写出 `learning_rate=5e-4`，是为了与本地脚本默认值及旧日志中约 `0.000485` 的调度位置保持一致；不是新增调参。

------

## 启动后立刻查看屏幕输出

仍在 **WSL Ubuntu Bash** 执行：

```bash
screen -ls
screen -r minimind-pretrain-resume
```

进入 screen 后，重点确认以下四点：

1. **恢复点不是从 1 开始。**
   首条周期日志不一定刚好打印 `73001`，因为 `log_interval=20`；正常情况下应很快看到类似：

   ```text
   Epoch:[1/1](73020/635119)
   ```

   或其他约 `73000` 附近的编号。

2. **checkpoint 已读取。**
   预期会出现加载 checkpoint / resume 相关信息；不能出现 `FileNotFoundError`、权重 shape 不匹配、optimizer load 失败等报错。

3. **SwanLab 应恢复旧 run。**
   checkpoint 内的 run id 是：

   ```text
   gd3zf7856ek4ad8divdij
   ```

   本地代码会在存在该 id 时以恢复模式初始化可视化 run。若日志明确显示新建了不同 id 的 run，而不是恢复旧 run，先停止继续操作并把输出发来。

4. **下一次保存应接近 step 73200。**
   当前 checkpoint 是 `73000`，`save_interval=200`；因此下一次成功 checkpoint 通常会在约 `73200` 附近发生。

确认输出正常后，正确 detach：

```text
Ctrl+A
松开
D
```

detach 后应回到外层 WSL shell；外层终端**不应该继续刷 Epoch 日志**。

------

## detach 后的只读监控命令

在 **WSL Ubuntu Bash** 执行：

```bash
cd /home/harry/projects/MiniMind

echo "========== screen =========="
screen -ls || true

echo
echo "========== train process tree =========="
ps -eo pid,ppid,tty,etime,stat,cmd | grep '[t]rain_pretrain.py' || true

echo
echo "========== GPU =========="
nvidia-smi

echo
echo "========== latest resume log =========="
latest_log="$(
  find experiments/logs -maxdepth 1 -type f -name 'pretrain-resume-*.log' \
    -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"
echo "latest_log: ${latest_log:-<not found>}"
[[ -n "${latest_log:-}" ]] && tail -n 50 "$latest_log"

echo
echo "========== checkpoint timestamps =========="
for f in \
  checkpoints/pretrain_768_resume.pth \
  checkpoints/pretrain_768.pth \
  out/pretrain_768.pth
do
  [[ -e "$f" ]] && stat -c '%y | %s bytes | %n' "$f" || echo "MISSING | $f"
done
```

等日志明确超过约 `73200` 后，再读取 checkpoint 内 step 是否更新：

```bash
cd /home/harry/projects/MiniMind

./.venv/bin/python - <<'PY'
import torch

path = "checkpoints/pretrain_768_resume.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

print("epoch:", checkpoint.get("epoch"))
print("step:", checkpoint.get("step"))
print("wandb_id:", checkpoint.get("wandb_id"))
print("has_model:", checkpoint.get("model") is not None)
print("has_optimizer:", checkpoint.get("optimizer") is not None)
print("has_scaler:", checkpoint.get("scaler") is not None)
PY
```

------

## 出现这些情况时立刻停在当前阶段

不要开启第二个 screen，不要重跑命令，不要开始 SFT：

+ 日志从 `1`、`20` 或远小于 `73000` 的位置开始；
+ 明确创建了新的 SwanLab run id；
+ resume checkpoint load 报错；
+ `screen -ls` 没有 session 且日志显示 Python 异常退出；
+ `nvidia-smi` 中没有 Python，但日志并未显示自然完成；
+ 下一次 checkpoint 仍无法把 `step` 推进到约 `73200`。

若恢复明显错误并且仍在 screen 内运行，按一次 `Ctrl+C` 停掉**这条刚启动的错误训练**，不要删除旧 checkpoint，然后把 screen 输出和最新 `pretrain-resume-*.log` 尾部贴过来。



