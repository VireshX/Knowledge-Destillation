
import json
from pathlib import Path

INPUT_FILE = Path("teacher_dataset_new.jsonl")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    rows = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print("Total examples:", len(rows))

for index, example in enumerate(rows, start=1):
    print(f"\n--- Example {index} ---")
    print("Instruction:", example.get("instruction", ""))
    print("Input:", example.get("input", "")[:200])
    print("Output preview:", example.get("output", "")[:350])
