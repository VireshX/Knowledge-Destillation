
import json
import random
from pathlib import Path

INPUT_FILE = Path("teacher_dataset_clean.jsonl")

TARGET_SIZES = [500, 1000, 2500, 5000, 10000]
RANDOM_SEED = 42

# Load the validated dataset
with INPUT_FILE.open("r", encoding="utf-8") as file:
    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# Shuffle reproducibly for future experiments
random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(dataset)

available = len(dataset)

print("Available examples:", available)
print("\n--- DATASET SIZE EXPERIMENT PLAN ---")

for target_size in TARGET_SIZES:
    actual_size = min(target_size, available)

    print(
        f"Target: {target_size:>5} | "
        f"Available subset: {actual_size:>5}"
    )

    # Create files only when the full target size exists
    if available >= target_size:
        output_file = Path(
            f"dataset_{target_size}.jsonl"
        )

        with output_file.open(
            "w", encoding="utf-8"
        ) as file:
            for example in dataset[:target_size]:
                file.write(
                    json.dumps(
                        example, ensure_ascii=False
                    ) + "\n"
                )

        print("Saved:", output_file)

print("\nExperiment preparation complete.")
