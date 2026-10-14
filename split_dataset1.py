
import json
import random
from pathlib import Path

INPUT_FILE = Path("teacher_dataset_final.jsonl")
TRAIN_FILE = Path("train.jsonl")
VALIDATION_FILE = Path("validation.jsonl")

random.seed(42)

with INPUT_FILE.open("r", encoding="utf-8") as file:
    examples = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# Shuffle examples in a repeatable way
random.shuffle(examples)

# Use approximately 80% for training
split_index = int(len(examples) * 0.8)

train_data = examples[:split_index]
validation_data = examples[split_index:]


def save_jsonl(file_path, data):
    with file_path.open("w", encoding="utf-8") as file:
        for example in data:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


save_jsonl(TRAIN_FILE, train_data)
save_jsonl(VALIDATION_FILE, validation_data)

print("--- DATASET SPLIT REPORT ---")
print("Total examples:", len(examples))
print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Training file:", TRAIN_FILE)
print("Validation file:", VALIDATION_FILE)
