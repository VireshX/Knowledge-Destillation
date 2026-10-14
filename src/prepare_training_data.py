
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_example(example):
    instruction = example["instruction"].strip()
    extra_input = example.get("input", "").strip()
    output = example["output"].strip()

    user_message = instruction
    if extra_input:
        user_message += "\n\nRequirements:\n" + extra_input

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

for filename in ["data/splits/train.jsonl", "data/splits/validation.jsonl"]:
    path = Path(filename)
    output_path = Path(filename.replace(".jsonl", "_formatted.jsonl"))

    count = 0
    with path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as target:

        for line in source:
            example = json.loads(line)
            formatted = format_example(example)
            target.write(json.dumps(
                {"text": formatted},
                ensure_ascii=False,
            ) + "\n")
            count += 1

    print(f"{output_path}: {count} examples")

print("Training data formatting complete.")
