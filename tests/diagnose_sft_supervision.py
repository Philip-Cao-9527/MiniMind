#!/usr/bin/env python3
import argparse
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.lm_dataset import SFTDataset, post_processing_chat, pre_processing_chat
from trainer.trainer_utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="诊断 SFT 数据监督 token 是否因截断丢失")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "sft_t2t_mini.jsonl"),
        help="SFT JSONL 数据路径",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=str(PROJECT_ROOT / "model"),
        help="本地 tokenizer 路径",
    )
    parser.add_argument("--max_seq_len", type=int, default=384, help="与训练一致的最大序列长度")
    parser.add_argument("--steps", type=int, default=10000, help="审计前多少个 micro-step")
    parser.add_argument("--seed", type=int, default=42, help="与训练一致的随机种子")
    parser.add_argument(
        "--focus_steps",
        type=int,
        nargs="*",
        default=[980, 1060, 1880, 2960, 4800, 7580, 8220, 8840],
        help="重点检查的 micro-step 列表（1-based）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "out"),
        help="诊断输出保存目录，默认保存到 tests/out",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="可选：显式指定诊断输出文件路径；指定后优先于 output_dir",
    )
    return parser.parse_args()


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_output_path(args):
    if args.output_path:
        return Path(args.output_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = (
        "diagnose_sft_supervision"
        f"-seq{args.max_seq_len}"
        f"-steps{args.steps}"
        f"-seed{args.seed}"
        f"-{timestamp}.log"
    )
    return Path(args.output_dir) / filename


def generate_labels(input_ids, bos_id, eos_id):
    labels = [-100] * len(input_ids)
    spans = []
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            supervised_end = min(end + len(eos_id), len(input_ids))
            if start < supervised_end:
                spans.append((start, supervised_end))
            for j in range(start, supervised_end):
                labels[j] = input_ids[j]
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels, spans


def pad_to_length(input_ids, labels, pad_id, max_seq_len):
    padded_input_ids = list(input_ids)
    padded_labels = list(labels)
    pad_len = max_seq_len - len(padded_input_ids)
    if pad_len > 0:
        padded_input_ids.extend([pad_id] * pad_len)
        padded_labels.extend([-100] * pad_len)
    return padded_input_ids, padded_labels


def analyze_prompt(input_ids_full, bos_id, eos_id, pad_id, max_seq_len):
    full_labels, full_spans = generate_labels(input_ids_full, bos_id, eos_id)
    full_supervision = sum(label != -100 for label in full_labels)

    original_input_ids = input_ids_full[:max_seq_len]
    original_labels, _ = generate_labels(original_input_ids, bos_id, eos_id)
    original_input_ids, original_labels = pad_to_length(
        original_input_ids,
        original_labels,
        pad_id,
        max_seq_len,
    )
    original_supervision = sum(label != -100 for label in original_labels)

    repaired_window_start = max(0, len(input_ids_full) - max_seq_len)
    repaired_window_end = repaired_window_start + min(len(input_ids_full), max_seq_len)
    repaired_input_ids = input_ids_full[repaired_window_start:repaired_window_end]
    repaired_labels = full_labels[repaired_window_start:repaired_window_end]
    repaired_input_ids, repaired_labels = pad_to_length(
        repaired_input_ids,
        repaired_labels,
        pad_id,
        max_seq_len,
    )
    repaired_supervision = sum(label != -100 for label in repaired_labels)

    last_span = full_spans[-1] if full_spans else None
    last_span_in_original = False
    last_span_in_repaired = False
    if last_span is not None:
        span_start, span_end = last_span
        last_span_in_original = span_start < max_seq_len and span_end > 0
        last_span_in_repaired = span_start < repaired_window_end and span_end > repaired_window_start

    lost_due_to_prefix_truncation = full_supervision > 0 and original_supervision == 0

    return {
        "raw_token_length": len(input_ids_full),
        "truncated_token_length": min(len(input_ids_full), max_seq_len),
        "full_supervision_tokens": full_supervision,
        "original_supervision_tokens": original_supervision,
        "repaired_supervision_tokens": repaired_supervision,
        "full_spans": full_spans,
        "last_span": last_span,
        "last_span_in_original": last_span_in_original,
        "last_span_in_repaired": last_span_in_repaired,
        "lost_due_to_prefix_truncation": lost_due_to_prefix_truncation,
        "original_window": (0, min(len(input_ids_full), max_seq_len)),
        "repaired_window": (repaired_window_start, repaired_window_end),
    }


def build_prompt(dataset, raw_sample):
    conversations = pre_processing_chat(raw_sample["conversations"])
    prompt = dataset.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)
    return prompt


def format_span(span):
    if span is None:
        return "None"
    return f"[{span[0]}, {span[1]})"


def main():
    args = parse_args()
    output_path = build_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        tee = TeeWriter(sys.stdout, output_file)
        with redirect_stdout(tee):
            print(f"output_path: {output_path}")

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
            focus_steps = sorted(set(step for step in args.focus_steps if step >= 1))
            focus_step_set = set(focus_steps)
            max_steps = min(args.steps, len(dataset))

            setup_seed(args.seed)
            indices = torch.randperm(len(dataset)).tolist()

            zero_supervision_steps = []
            repaired_zero_supervision_steps = []
            focus_results = {}

            for step in range(1, max_steps + 1):
                dataset_index = indices[step - 1]
                raw_sample = dataset.samples[dataset_index]
                prompt = build_prompt(dataset, raw_sample)
                input_ids_full = tokenizer(prompt).input_ids
                analysis = analyze_prompt(
                    input_ids_full=input_ids_full,
                    bos_id=dataset.bos_id,
                    eos_id=dataset.eos_id,
                    pad_id=tokenizer.pad_token_id,
                    max_seq_len=args.max_seq_len,
                )
                analysis["step"] = step
                analysis["dataset_index"] = int(dataset_index)

                if analysis["original_supervision_tokens"] == 0:
                    zero_supervision_steps.append(step)
                if analysis["repaired_supervision_tokens"] == 0:
                    repaired_zero_supervision_steps.append(step)

                if step in focus_step_set:
                    focus_results[step] = analysis

            toy_logits = torch.zeros(8, tokenizer.vocab_size, dtype=torch.float32)
            toy_labels_all_ignore = torch.full((8,), -100, dtype=torch.long)
            toy_labels_valid = torch.tensor([0, 1, 2, 3, -100, 4, 5, 6], dtype=torch.long)
            toy_loss_all_ignore = F.cross_entropy(
                toy_logits,
                toy_labels_all_ignore,
                ignore_index=-100,
                reduction="mean",
            )
            toy_loss_valid = F.cross_entropy(
                toy_logits,
                toy_labels_valid,
                ignore_index=-100,
                reduction="mean",
            )

            hit_focus_steps = [step for step in focus_steps if step in zero_supervision_steps]
            missed_focus_steps = [step for step in focus_steps if step not in zero_supervision_steps]

            print("=== SFT 监督诊断摘要 ===")
            print(f"data_path: {args.data_path}")
            print(f"tokenizer_path: {args.tokenizer_path}")
            print(f"seed: {args.seed}")
            print(f"max_seq_len: {args.max_seq_len}")
            print(f"steps_audited: {max_steps}")
            print(f"zero_supervision_steps_original: {len(zero_supervision_steps)}")
            print(f"zero_supervision_step_list_original: {json.dumps(zero_supervision_steps, ensure_ascii=False)}")
            print(f"zero_supervision_steps_repaired: {len(repaired_zero_supervision_steps)}")
            print(f"zero_supervision_step_list_repaired: {json.dumps(repaired_zero_supervision_steps, ensure_ascii=False)}")
            print(f"focus_steps: {json.dumps(focus_steps, ensure_ascii=False)}")
            print(f"focus_steps_hit_zero_supervision: {json.dumps(hit_focus_steps, ensure_ascii=False)}")
            print(f"focus_steps_missed_zero_supervision: {json.dumps(missed_focus_steps, ensure_ascii=False)}")
            print()

            print("=== CPU toy CrossEntropy 验证 ===")
            print(
                "all_ignore_labels_mean_loss: "
                f"{toy_loss_all_ignore.item()} | isfinite={torch.isfinite(toy_loss_all_ignore).item()}"
            )
            print(
                "mixed_valid_labels_mean_loss: "
                f"{toy_loss_valid.item()} | isfinite={torch.isfinite(toy_loss_valid).item()}"
            )
            print()

            print("=== 原始前缀截断行为 vs 修复后尾部窗口行为 ===")
            for step in focus_steps:
                info = focus_results.get(step)
                if info is None:
                    print(f"step={step}: 超出本次审计范围")
                    continue
                print(
                    f"step={step} | dataset_index={info['dataset_index']} | "
                    f"raw_token_length={info['raw_token_length']} | "
                    f"truncated_token_length={info['truncated_token_length']}"
                )
                print(
                    f"  full_supervision_tokens={info['full_supervision_tokens']} | "
                    f"original_supervision_tokens={info['original_supervision_tokens']} | "
                    f"repaired_supervision_tokens={info['repaired_supervision_tokens']}"
                )
                print(
                    f"  original_window={info['original_window']} | "
                    f"repaired_window={info['repaired_window']}"
                )
                print(
                    f"  last_assistant_span={format_span(info['last_span'])} | "
                    f"last_span_in_original={int(info['last_span_in_original'])} | "
                    f"last_span_in_repaired={int(info['last_span_in_repaired'])}"
                )
                print(
                    f"  lost_due_to_prefix_truncation={int(info['lost_due_to_prefix_truncation'])}"
                )

            if zero_supervision_steps:
                print()
                print("=== 结论边界提示 ===")
                print("1. 命中 zero supervision 只能支持“全 ignore labels 是本次 NaN 的主要候选机制”。")
                print("2. 数据诊断本身不能单独证明所有 NaN 都只由这一机制导致。")
                print("3. 训练侧仍需显式跳过 zero supervision batch，并在非有限 loss/梯度时立即失败。")


if __name__ == "__main__":
    main()
