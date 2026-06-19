from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import add_common_args, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attention every K decode steps; skipped steps remove attention delta.")
    add_common_args(parser, "periodic_refresh")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
