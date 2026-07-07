from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = EXPERIMENT_DIR / "tiny_pretrain.jsonl"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} 出现 NaN 或 inf")


def require_finite_gradients(model: torch.nn.Module) -> None:
    missing = []
    checked = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        require_finite_tensor(f"gradient:{name}", param.grad)
        checked += 1
    if checked == 0:
        raise RuntimeError("没有检查到任何有效梯度")
    if missing:
        print(f"梯度为空参数数量: {len(missing)}，示例: {missing[:3]}")
    print(f"梯度有限性检查: 通过，已检查 {checked} 个参数")


def read_raw_texts(data_path: Path) -> list[str]:
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{data_path}:{line_no} 缺少 text 字段")
            rows.append(str(row["text"]))
    return rows


def print_first_sample(tokenizer, dataset: PretrainDataset, raw_texts: list[str]) -> None:
    input_ids, labels = dataset[0]
    ids = input_ids.tolist()
    label_values = labels.tolist()
    minus_100_positions = [idx for idx, value in enumerate(label_values) if value == -100]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\n=== 第一条样本可观察信息 ===")
    print(f"原始文本: {raw_texts[0]}")
    print(f"input_ids: {ids}")
    print(f"tokens: {tokens}")
    print(f"decode(skip_special_tokens=False): {tokenizer.decode(ids, skip_special_tokens=False)}")
    print(f"labels: {label_values}")
    print(f"label 中 -100 的位置: {minus_100_positions}")
    print(
        "PAD token id 与 -100 的区别: "
        f"PAD token id={tokenizer.pad_token_id} 是输入序列中的真实 padding token；"
        "-100 只出现在 labels 中，表示 cross_entropy(ignore_index=-100) 忽略这些位置。"
    )


def build_config(tokenizer, seq_len: int) -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=seq_len,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        flash_attn=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind 极小预训练闭环实验")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 torch.cuda.is_available() 为 False")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=== 实验入口 ===")
    print(f"仓库根目录: {ROOT}")
    print(f"数据文件: {args.data_path}")
    print(f"tokenizer 路径: {args.tokenizer_path}")
    print(f"产物目录: {args.output_dir}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("\n=== tokenizer ===")
    print(f"tokenizer 类: {type(tokenizer).__name__}")
    print(f"BOS: {tokenizer.bos_token!r}, id={tokenizer.bos_token_id}")
    print(f"EOS: {tokenizer.eos_token!r}, id={tokenizer.eos_token_id}")
    print(f"PAD: {tokenizer.pad_token!r}, id={tokenizer.pad_token_id}")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("当前 tokenizer.pad_token_id 为 None，PretrainDataset 无法构造 padding labels")

    raw_texts = read_raw_texts(args.data_path)
    dataset = PretrainDataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)
    print("\n=== dataset ===")
    print(f"PretrainDataset 类: {type(dataset).__name__}")
    print(f"JSONL 样本数: {len(raw_texts)}")
    print(f"Dataset 样本数: {len(dataset)}")
    print(f"max_seq_len: {args.max_seq_len}；对应 MiniMindConfig.max_position_embeddings")
    print_first_sample(tokenizer, dataset, raw_texts)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    input_ids, labels = next(iter(loader))
    print("\n=== DataLoader batch ===")
    print(f"input_ids.shape: {tuple(input_ids.shape)}, dtype={input_ids.dtype}, device={input_ids.device}")
    print(f"labels.shape: {tuple(labels.shape)}, dtype={labels.dtype}, device={labels.device}")
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    print(f"移动到训练设备后: input_ids.device={input_ids.device}, labels.device={labels.device}")

    config = build_config(tokenizer, args.max_seq_len)
    print("\n=== 随机初始化模型配置 ===")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"head_dim: {config.head_dim}")
    print(f"vocab_size: {config.vocab_size}")
    print("权重来源: 随机初始化，本轮不加载任何预训练权重")

    model = MiniMindForCausalLM(config).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    watched_name = "model.embed_tokens.weight"
    watched_param = dict(model.named_parameters())[watched_name]
    before_update = watched_param.detach().clone()

    print("\n=== forward / loss / backward / step ===")
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits
    loss = outputs.loss + outputs.aux_loss
    print(f"logits.shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.8f}")
    require_finite_tensor("loss", loss.detach())
    require_finite_tensor("logits", logits.detach())

    loss.backward()
    require_finite_gradients(model)
    optimizer.step()
    require_finite_tensor(watched_name, watched_param.detach())

    after_update = watched_param.detach().clone()
    max_abs_delta = (after_update - before_update).abs().max().item()
    print(f"观察参数: {watched_name}")
    print(f"更新前均值: {before_update.float().mean().item():.10f}")
    print(f"更新后均值: {after_update.float().mean().item():.10f}")
    print(f"最大绝对变化: {max_abs_delta:.10f}")
    if not math.isfinite(max_abs_delta) or max_abs_delta <= 0:
        raise RuntimeError(f"{watched_name} 未发生可观测更新")
    print("参数更新检查: 通过")

    weight_path = args.output_dir / "tiny_pretrain_state_dict.pth"
    resume_path = args.output_dir / "tiny_pretrain_resume_checkpoint.pth"
    step = 1
    torch.save(model.state_dict(), weight_path)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "seed": args.seed,
            "config": config.to_dict(),
            "watched_name": watched_name,
        },
        resume_path,
    )
    print("\n=== 保存产物 ===")
    print(f"仅 state_dict 权重: {weight_path}")
    print(f"resume checkpoint: {resume_path}")

    print("\n=== resume 验证 ===")
    restored_model = MiniMindForCausalLM(config).to(device)
    restored_optimizer = AdamW(restored_model.parameters(), lr=args.lr)
    checkpoint = torch.load(resume_path, map_location=device)
    restored_model.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_param = dict(restored_model.named_parameters())[checkpoint["watched_name"]]
    same_param = torch.allclose(restored_param.detach(), watched_param.detach(), atol=0, rtol=0)
    optimizer_state_entries = len(restored_optimizer.state_dict()["state"])
    print(f"恢复 step: {checkpoint['step']}")
    print(f"关键参数一致性: {same_param}")
    print(f"optimizer state 条目数: {optimizer_state_entries}")
    if checkpoint["step"] != step:
        raise RuntimeError("resume checkpoint 中的 step 与预期不一致")
    if not same_param:
        raise RuntimeError("resume 后关键参数与保存前不一致")
    if optimizer_state_entries == 0:
        raise RuntimeError("resume 后 optimizer state 为空")
    print("resume 验证: 通过")
    print("\n闭环结果: JSONL -> tokenizer -> PretrainDataset -> DataLoader -> forward -> loss -> backward -> optimizer.step -> 保存权重 -> resume 验证 已完成")


if __name__ == "__main__":
    main()
