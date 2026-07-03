#!/usr/bin/env python3
"""Validate key constraints in a MiniMind task prompt template or generated prompt."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    message: str
    needles: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        if self.needles and not all(needle in text for needle in self.needles):
            return False
        if self.any_of and not any(needle in text for needle in self.any_of):
            return False
        return True


def read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"UTF-8 读取失败：{exc}") from exc
    except OSError as exc:
        raise ValueError(f"文件读取失败：{exc}") from exc


def validate_rules(text: str) -> list[str]:
    rules = [
        Rule("缺少 MiniMind 仓库路径。", needles=("/home/harry/projects/MiniMind",)),
        Rule("缺少必读文件要求。", needles=("AGENTS.md", "README.md", "docs/minimind-roadmap.md")),
        Rule("缺少 MiniMind 项目边界。", needles=("个人 LLM 源码理解", "上游 MiniMind", "Learn MiniMind")),
        Rule("缺少事实分层要求。", needles=("本机已验证事实", "本地代码事实", "上游事实", "后续计划")),
        Rule("缺少硬件边界。", needles=("RTX 5060 Laptop", "8GB")),
        Rule("缺少禁止环境重装建议。", needles=("不要建议重装", "CUDA", "PyTorch")),
        Rule("缺少默认验证命令。", needles=("git status --short --branch", "direnv exec . python -V")),
        Rule("缺少 UTF-8 / 解析校验要求。", needles=("UTF-8", "解析", "编译")),
        Rule("缺少模型任务验证边界。", needles=("模型", "数据", "训练", "推理", "smoke test")),
        Rule("缺少最终总结结构。", needles=("文件改动清单", "验证命令", "证据路径", "未验证边界")),
        Rule("仍残留通用花括号占位符。", any_of=("{{", "}}")),
    ]
    missing = [rule.message for rule in rules[:-1] if not rule.matches(text)]
    if "{{" in text or "}}" in text:
        missing.append(rules[-1].message)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="检查项目任务 prompt 模板或生成物是否保留关键执行约束。"
    )
    parser.add_argument("prompt_path", help="目标 prompt-template.md 或生成出的 prompt 文件路径")
    args = parser.parse_args(argv)

    try:
        text = read_utf8(Path(args.prompt_path))
    except ValueError as exc:
        print(f"失败：{exc}")
        return 1

    missing = validate_rules(text)
    if missing:
        print("失败：项目任务 prompt 缺少以下关键约束：")
        for item in missing:
            print(f"- {item}")
        return 1

    print("通过：项目任务 prompt 关键约束检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
