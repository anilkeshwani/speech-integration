#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU regression tests with matrix selectors.")
    parser.add_argument("--tokenizer", action="append", default=[], help="Tokenizer selector (repeatable).")
    parser.add_argument("--approach", action="append", default=[], help="Approach selector (repeatable).")
    parser.add_argument("--matrix-id", action="append", default=[], help="Matrix id selector (repeatable).")
    parser.add_argument(
        "--tier",
        choices=["smoke", "full", "all"],
        default="smoke",
        help="Test tier marker expression.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include rows currently disabled in tests/regression/matrix.yaml.",
    )
    parser.add_argument("pytest_args", nargs="*", help="Additional args forwarded to pytest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    marker_parts = ["gpu and regression and heavy"]
    if args.tier == "smoke":
        marker_parts.append("tier_smoke")
    elif args.tier == "full":
        marker_parts.append("tier_full")
    marker_expr = " and ".join(marker_parts)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        "tests/regression",
        "-m",
        marker_expr,
    ]
    for tokenizer in args.tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for approach in args.approach:
        cmd.extend(["--approach", approach])
    for matrix_id in args.matrix_id:
        cmd.extend(["--matrix-id", matrix_id])
    if args.include_disabled:
        cmd.append("--include-disabled")
    cmd.extend(args.pytest_args)

    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
