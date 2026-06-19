from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import add_common_args, layerwise_schedule_from_args, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Use different attention refresh intervals per attention layer.")
    add_common_args(parser, "layerwise_schedule")
    args = parser.parse_args()
    schedule = layerwise_schedule_from_args(args)
    run_experiment(args, schedule=schedule)


if __name__ == "__main__":
    main()
