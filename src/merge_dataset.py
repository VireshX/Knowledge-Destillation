
import json
from pathlib import Path

ORIGINAL_FILE = Path("teacher_dataset_clean.jsonl")
NEW_FILE = Path("teacher_dataset_new.jsonl")
OUTPUT_FILE = Path("teacher_dataset_final.jsonl")

MIN_OUTPUT_LENGTH = 200

all_examples = []
seen_instructions = set()

duplicate_count = 0
invalid_count = 0


def load_examples(file_path):
    examples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                print(f"{file_path}, line {line_number}: Invalid JSON")
                continue

            if isinstance(example, dict):
                examples.append(example)
            else:
                print(f"{file_path}, line {line_number}: Not a JSON object")

    return examples


for file_path in [ORIGINAL_FILE, NEW_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    examples = load_examples(file_path)

    for example in examples:
        required_fields = ["instruction", "input", "output"]

        if any(
            not isinstance(example.get(key), str)
            or not example[key].strip()
            for key in required_fields
        ):
            invalid_count += 1
            continue

        output = example["output"].strip()

        if len(output) < MIN_OUTPUT_LENGTH:
            invalid_count += 1
            continue

        if output.lower() in {
            "null",
            "{}",
            "[]",
            "design solution json",
        }:
            invalid_count += 1
            continue

        instruction_key = example["instruction"].strip().lower()

        if instruction_key in seen_instructions:
            duplicate_count += 1
            continue

        seen_instructions.add(instruction_key)
        all_examples.append(example)


with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    for example in all_examples:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")


print("\n--- FINAL DATASET REPORT ---")
print("Final usable examples:", len(all_examples))
print("Duplicates removed:", duplicate_count)
print("Invalid examples skipped:", invalid_count)
print("Output file:", OUTPUT_FILE)
