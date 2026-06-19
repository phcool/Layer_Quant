from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the low-frequency attention experiment suite.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--output-root", default="results/low_freq_attention")
    args = parser.parse_args()

    common = [
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--decode-steps",
        str(args.decode_steps),
        "--warmup-steps",
        str(args.warmup_steps),
    ]
    if args.quality:
        common.append("--quality")

    jobs = []
    for interval in [2, 4, 8]:
        jobs.append(
            [
                args.python,
                "scripts/low_freq_attention/periodic_refresh/run.py",
                *common,
                "--interval",
                str(interval),
                "--output-dir",
                f"{args.output_root}/periodic_refresh/ctx{args.sequence_length}_interval{interval}",
            ]
        )
        jobs.append(
            [
                args.python,
                "scripts/low_freq_attention/correction_cache/run.py",
                *common,
                "--interval",
                str(interval),
                "--correction-decay",
                "1.0",
                "--output-dir",
                f"{args.output_root}/correction_cache/ctx{args.sequence_length}_interval{interval}_reuse",
            ]
        )
    jobs.append(
        [
            args.python,
            "scripts/low_freq_attention/correction_cache/run.py",
            *common,
            "--interval",
            "4",
            "--correction-decay",
            "0.95",
            "--output-dir",
            f"{args.output_root}/correction_cache/ctx{args.sequence_length}_interval4_decay095",
        ]
    )
    jobs.append(
        [
            args.python,
            "scripts/low_freq_attention/layerwise_schedule/run.py",
            *common,
            "--interval",
            "4",
            "--layer-intervals",
            "7:1,18:2,29:4,40:4",
            "--output-dir",
            f"{args.output_root}/layerwise_schedule/ctx{args.sequence_length}_7x1_18x2_29x4_40x4",
        ]
    )
    jobs.append(
        [
            args.python,
            "scripts/low_freq_attention/layerwise_schedule/run.py",
            *common,
            "--interval",
            "4",
            "--layer-intervals",
            "7:4,18:4,29:2,40:1",
            "--output-dir",
            f"{args.output_root}/layerwise_schedule/ctx{args.sequence_length}_7x4_18x4_29x2_40x1",
        ]
    )
    for keep_fraction in [0.25, 0.5, 0.75]:
        jobs.append(
            [
                args.python,
                "scripts/low_freq_attention/calibrated_gating/run.py",
                *common,
                "--keep-fraction",
                str(keep_fraction),
                "--max-skip",
                "4",
                "--output-dir",
                f"{args.output_root}/calibrated_gating/ctx{args.sequence_length}_keep{int(keep_fraction * 100)}_maxskip4",
            ]
        )

    for job in jobs:
        run(job)


if __name__ == "__main__":
    main()
