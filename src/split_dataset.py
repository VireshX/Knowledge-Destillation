
import json
import random
from pathlib import Path

INPUT_FILE = Path("teacher_dataset_clean.jsonl")
TRAIN_FILE = Path("data/splits/train.jsonl")
VALIDATION_FILE = Path("data/splits/validation.jsonl")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Load clean dataset
with INPUT_FILE.open("r", encoding="utf-8") as file:
    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(dataset) < 2:
    raise ValueError(
        "At least 2 examples are required for splitting."
    )

# Shuffle reproducibly
random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(dataset)

# Split dataset
split_index = int(len(dataset) * TRAIN_RATIO)
split_index = max(1, min(split_index, len(dataset) - 1))

train_data = dataset[:split_index]
validation_data = dataset[split_index:]


def save_jsonl(file_path, examples):
    with file_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


save_jsonl(TRAIN_FILE, train_data)
save_jsonl(VALIDATION_FILE, validation_data)

print("--- DATASET SPLIT REPORT ---")
print("Total examples:", len(dataset))
print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Training file:", TRAIN_FILE)
print("Validation file:", VALIDATION_FILE)
