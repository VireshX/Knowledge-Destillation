
import json
from pathlib import Path

INPUT_FILE = Path("data/raw/teacher_dataset.jsonl")
OUTPUT_FILE = Path("teacher_dataset_clean.jsonl")

REQUIRED_FIELDS = ["instruction", "input", "output"]
MIN_OUTPUT_LENGTH = 150

valid_examples = []
invalid_count = 0
seen_records = set()
duplicate_count = 0

with INPUT_FILE.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            example = json.loads(line)
        except json.JSONDecodeError:
            print(f"Line {line_number}: Invalid JSON")
            invalid_count += 1
            continue

        if not isinstance(example, dict):
            print(f"Line {line_number}: Record is not a JSON object")
            invalid_count += 1
            continue

        valid_structure = True

        for key in REQUIRED_FIELDS:
            value = example.get(key)

            if not isinstance(value, str) or not value.strip():
                print(f"Line {line_number}: Invalid field '{key}'")
                valid_structure = False

        if not valid_structure:
            invalid_count += 1
            continue

        output = example["output"].strip()
        normalized_output = output.lower().strip()

        bad_outputs = {
            "null",
            "{}",
            "[]",
            "design solution json",
        }

        if (
            len(output) < MIN_OUTPUT_LENGTH
            or normalized_output in bad_outputs
        ):
            print(
                f"Line {line_number}: Output too short or unusable "
                f"({len(output)} characters)"
            )
            invalid_count += 1
            continue

        record_key = json.dumps(
            example,
            sort_keys=True,
            ensure_ascii=False
        )

        if record_key in seen_records:
            duplicate_count += 1
            continue

        seen_records.add(record_key)
        valid_examples.append(example)

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    for example in valid_examples:
        file.write(
            json.dumps(example, ensure_ascii=False) + "\n"
        )

print("\n--- DATASET QUALITY REPORT ---")
print("Usable examples:", len(valid_examples))
print("Invalid or low-quality records:", invalid_count)
print("Exact duplicates removed:", duplicate_count)
print("Clean dataset:", OUTPUT_FILE)
