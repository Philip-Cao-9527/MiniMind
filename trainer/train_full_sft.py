import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def build_failure_context(epoch, step, valid_label_tokens, lr, extra=None):
    parts = [
        f"epoch={epoch + 1}/{args.epochs}",
        f"step={step}",
        f"valid_label_tokens={valid_label_tokens}",
        f"lr={lr:.8f}",
        f"dtype={args.dtype}",
        f"max_seq_len={args.max_seq_len}",
        f"accumulation_steps={args.accumulation_steps}",
    ]
    if extra:
        parts.append(extra)
    return ", ".join(parts)


def ensure_finite_tensor(name, tensor, epoch, step, valid_label_tokens, lr):
    if tensor is None:
        return
    if not torch.isfinite(tensor).all():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"non_finite_{name}: {build_failure_context(epoch, step, valid_label_tokens, lr)}"
        )


def save_training_state(epoch, step, wandb=None, reason='periodic', requested_step=None):
    if not is_main_process():
        return

    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir='../checkpoints',
        scaler=scaler,
    )
    Logger(
        f'Checkpoint saved at micro-step {step} '
        f'(reason={reason}, requested_step={requested_step or step})'
    )
    model.train()
    del state_dict


def apply_optimizer_update(epoch, step, valid_label_tokens, lr):
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    grad_norm_tensor = grad_norm if torch.is_tensor(grad_norm) else torch.tensor(grad_norm)
    if not torch.isfinite(grad_norm_tensor).all():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"non_finite_grad_norm: {build_failure_context(epoch, step, valid_label_tokens, lr, extra=f'grad_norm={grad_norm_tensor.item()}')}"
        )

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def validate_subset_args():
    if args.max_train_samples < 0:
        raise ValueError("--max_train_samples 不能小于 0")
    if not 0 < args.train_sample_ratio <= 1.0:
        raise ValueError("--train_sample_ratio 必须满足 0 < train_sample_ratio <= 1.0")
    if args.from_resume == 1 and (
        args.max_train_samples > 0 or args.train_sample_ratio < 1.0
    ):
        raise ValueError(
            "subset run 只支持 --from_resume 0：subset 数据顺序属于本次短跑实验配置，不能和历史 resume checkpoint 混用。"
        )


def maybe_build_subset(train_ds):
    original_samples = len(train_ds)
    if original_samples == 0:
        raise ValueError(f"SFT 数据集为空，无法训练：{args.data_path}")

    subset_config_enabled = args.max_train_samples > 0 or args.train_sample_ratio < 1.0
    subset_samples = original_samples

    if subset_config_enabled:
        candidate_sizes = []
        if args.train_sample_ratio < 1.0:
            candidate_sizes.append(int(original_samples * args.train_sample_ratio))
        if args.max_train_samples > 0:
            candidate_sizes.append(args.max_train_samples)

        subset_samples = min(candidate_sizes) if candidate_sizes else original_samples
        subset_samples = max(1, min(original_samples, subset_samples))

        if subset_samples < original_samples:
            if args.train_subset_mode == 'random':
                subset_generator = torch.Generator().manual_seed(args.train_subset_seed)
                subset_indices = torch.randperm(
                    original_samples, generator=subset_generator
                )[:subset_samples].tolist()
            else:
                subset_indices = list(range(subset_samples))
            train_ds = Subset(train_ds, subset_indices)

    effective_ratio = subset_samples / original_samples
    Logger(
        "SFT subset config: "
        f"original_train_samples={original_samples}, "
        f"subset_train_samples={subset_samples}, "
        f"train_sample_ratio={args.train_sample_ratio}, "
        f"effective_subset_ratio={effective_ratio:.6f}, "
        f"max_train_samples={args.max_train_samples}, "
        f"train_subset_seed={args.train_subset_seed}, "
        f"train_subset_mode={args.train_subset_mode}, "
        f"subset_config_enabled={int(subset_config_enabled)}, "
        f"subset_applied={int(subset_samples < original_samples)}, "
        f"from_weight={args.from_weight}, "
        f"from_weight_is_pretrain={int(args.from_weight == 'pretrain')}, "
        f"from_resume={args.from_resume}, "
        f"from_resume_is_zero={int(args.from_resume == 0)}"
    )
    return train_ds


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    last_valid_label_tokens = 0
    effective_backward_count = 0
    pending_save_step = None
    optimizer.zero_grad(set_to_none=True)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        valid_label_tokens = int((labels != -100).sum().item())
        if valid_label_tokens == 0:
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'skipped_no_supervision, valid_label_tokens: 0'
            )
            if step % args.save_interval == 0 and step != iters and is_main_process():
                if effective_backward_count == 0:
                    save_training_state(
                        epoch,
                        step,
                        wandb=wandb,
                        reason='periodic_no_pending_grad',
                        requested_step=step,
                    )
                elif pending_save_step is None:
                    pending_save_step = step
                    Logger(
                        f'Checkpoint save deferred from micro-step {step} '
                        f'until next optimizer update boundary'
                    )
            del input_ids, labels
            continue

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            logits_loss = res.loss
            aux_loss = res.aux_loss if res.aux_loss is not None else input_ids.new_zeros((), dtype=torch.float32)
            total_loss = logits_loss + aux_loss

        ensure_finite_tensor('logits_loss', logits_loss, epoch, step, valid_label_tokens, lr)
        ensure_finite_tensor('aux_loss', aux_loss, epoch, step, valid_label_tokens, lr)
        ensure_finite_tensor('loss', total_loss, epoch, step, valid_label_tokens, lr)

        loss = total_loss / args.accumulation_steps

        scaler.scale(loss).backward()
        effective_backward_count += 1
        last_valid_label_tokens = valid_label_tokens

        if effective_backward_count == args.accumulation_steps:
            apply_optimizer_update(epoch, step, last_valid_label_tokens, lr)
            effective_backward_count = 0
            if pending_save_step is not None and is_main_process():
                save_training_state(
                    epoch,
                    step,
                    wandb=wandb,
                    reason='deferred_periodic',
                    requested_step=pending_save_step,
                )
                pending_save_step = None

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = total_loss.item()
            current_aux_loss = aux_loss.item()
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if step % args.save_interval == 0 and step != iters and is_main_process():
            if effective_backward_count == 0:
                save_training_state(
                    epoch,
                    step,
                    wandb=wandb,
                    reason='periodic',
                    requested_step=step,
                )
            elif pending_save_step is None:
                pending_save_step = step
                Logger(
                    f'Checkpoint save deferred from micro-step {step} '
                    f'until next optimizer update boundary'
                )

        del input_ids, labels, res, logits_loss, aux_loss, total_loss, loss

    if last_step > start_step and effective_backward_count > 0:
        lr = optimizer.param_groups[-1]['lr']
        apply_optimizer_update(epoch, last_step, last_valid_label_tokens, lr)
        effective_backward_count = 0

    if last_step > start_step and is_main_process():
        save_reason = 'end_of_epoch'
        requested_step = last_step
        if pending_save_step is not None:
            save_reason = 'end_of_epoch_deferred_periodic'
            requested_step = pending_save_step
        save_training_state(
            epoch,
            last_step,
            wandb=wandb,
            reason=save_reason,
            requested_step=requested_step,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--max_train_samples", type=int, default=0, help="训练样本上限（0=不限制）")
    parser.add_argument("--train_sample_ratio", type=float, default=1.0, help="训练样本比例（1.0=全量）")
    parser.add_argument("--train_subset_seed", type=int, default=42, help="subset 抽样随机种子")
    parser.add_argument("--train_subset_mode", type=str, default="random", choices=["random", "head"], help="subset 取样模式（random=确定性随机抽样，head=截取前 N 条）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    validate_subset_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_ds = maybe_build_subset(train_ds)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
