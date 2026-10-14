
import json
from pathlib import Path

train_path = Path("train.jsonl")
example_path = Path("grievance_training_example.json")

# Load the new example
with example_path.open("r", encoding="utf-8") as f:
    example = json.load(f)

# Convert it to the existing dataset format
new_record = {
    "instruction": example["instruction"],
    "input": "",
    "output": example["response"]
}

# Validate the existing dataset before changing it
with train_path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

if not all(
    isinstance(record, dict)
    and {"instruction", "input", "output"}.issubset(record)
    for record in records
):
    raise ValueError("Existing dataset has an unexpected format.")

# Prevent accidental duplicate insertion
if any(
    record["instruction"] == new_record["instruction"]
    for record in records
):
    print("This example already exists. No changes made.")
else:
    with train_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    print("Grievance example added successfully.")
    print("Training examples:", len(records) + 1)
