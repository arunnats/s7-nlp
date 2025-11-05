#!/usr/bin/env python3
"""Generate test summaries using all 8 trained summarizer models"""

import os
import subprocess
from pathlib import Path

STRATEGIES = 8
DATA_DIR = "output"
MODEL_DIR = "models/summarizers"
OUTPUT_DIR = "test_summaries"

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    for strategy in range(1, STRATEGIES + 1):
        print(f"\n{'='*60}")
        print(f"Generating summaries for Strategy {strategy}")
        print('='*60)

        src_file = f"{DATA_DIR}/strategy_{strategy}/test.src.txt"
        model_file = f"{MODEL_DIR}/summarizer_strategy_{strategy}_step_20000.pt"
        output_file = f"{OUTPUT_DIR}/summaries_strategy_{strategy}.txt"

        cmd = [
            "onmt_translate",
            "-model", model_file,
            "-src", src_file,
            "-output", output_file,
            "-gpu", "0",
            "-batch_size", "32",
            "-beam_size", "5",
            "-min_length", "25",
            "-max_length", "256"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"✓ Saved to {output_file}")
        else:
            print(f"ERROR in strategy {strategy}")

    print("\n" + "="*60)
    print("All test summaries generated!")
    print("="*60)

if __name__ == "__main__":
    main()