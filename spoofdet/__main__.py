"""python -m spoofdet {train,evaluate,replay}."""

from __future__ import annotations

import sys

from spoofdet.evaluate import main as evaluate_main
from spoofdet.replay import main as replay_main
from spoofdet.train import main as train_main

COMMANDS = {
    "train": train_main,
    "evaluate": evaluate_main,
    "replay": replay_main,
}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command in ("-h", "--help"):
        print("usage: python -m spoofdet {train,evaluate,replay} [options]")
        return 0
    if command not in COMMANDS:
        print(f"unknown command {command}", file=sys.stderr)
        print("usage: python -m spoofdet {train,evaluate,replay}", file=sys.stderr)
        return 2
    sys.argv = [f"spoofdet {command}", *sys.argv[2:]]
    COMMANDS[command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
