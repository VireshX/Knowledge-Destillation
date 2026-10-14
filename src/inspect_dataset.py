
import json
from pathlib import Path

INPUT_FILE = Path("teacher_dataset_clean.jsonl")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print("Total examples:", len(dataset))

for index, example in enumerate(dataset, start=1):
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    print(f"\n{'=' * 60}")
    print(f"EXAMPLE {index}")
    print("=" * 60)

    print("\nINSTRUCTION:")
    print(instruction)

    print("\nINPUT:")
    print(input_text[:400])

    print("\nOUTPUT PREVIEW:")
    print(output_text[:700])

    print("\nCharacter counts:")
    print("Input:", len(input_text))
    print("Output:", len(output_text))
