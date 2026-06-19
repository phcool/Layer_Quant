from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import add_common_args, calibrated_schedule, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a fixed attention replay schedule from hidden-state delta statistics, then measure it without dynamic CPU sync."
    )
    add_common_args(parser, "calibrated_gating")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    args = parser.parse_args()
    schedule = calibrated_schedule(args)
    run_experiment(args, schedule=schedule)


if __name__ == "__main__":
    main()
