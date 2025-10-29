"""Quick test harness for tools/chunker.py

Runs the chunker on `rulingbr/sample-5.json` and prints a one-line preview
for the first document.
"""
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "rulingbr" / "sample-5.json"
OUT = ROOT / "tmp_sample_with_chunks.jsonl"


def run():
    cmd = [sys.executable, str(ROOT / "tools" / "chunker.py"), "--input", str(SAMPLE), "--out", str(OUT)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    with open(OUT, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        if not first:
            print("No output produced")
            return
        doc = json.loads(first)
        chunks = doc.get("chunks", [])
        print("Document id preview (first 8 chunks lengths):", [len(c.split()) for c in chunks])


if __name__ == "__main__":
    run()
