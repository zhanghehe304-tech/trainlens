"""命令行入口：``trainlens demo``。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from . import __version__
from .demo import run_demo


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="trainlens", description="训练过程记录与报告生成")
    parser.add_argument("--version", action="version", version=f"trainlens {__version__}")

    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="训练一个零依赖的示例并生成报告")
    demo.add_argument("--out", default="out", help="报告输出目录（默认 out/）")
    demo.add_argument("--epochs", type=int, default=10, help="训练轮数（默认 10）")
    demo.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    demo.add_argument("--quiet", action="store_true", help="只输出报告路径")

    args = parser.parse_args(argv)

    if args.command == "demo":
        path = run_demo(out_dir=args.out, epochs=args.epochs, seed=args.seed, quiet=args.quiet)
        print(Path(path).resolve())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
