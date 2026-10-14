
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 512
OUTPUT_DIR = "student_lora_checkpoints"
ADAPTER_DIR = "student_lora_adapter"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to("cuda")


def load_examples(filename):
    rows = []

    with open(filename, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            example = json.loads(line)
            text = example["text"]

            marker = "<|im_start|>assistant\n"
            start = text.rfind(marker)

            if start == -1:
                raise ValueError(
                    f"Assistant marker missing: {filename}, line {line_number}"
                )

            prompt = text[: start + len(marker)]
            answer = text[start + len(marker):]

            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]

            answer_ids = tokenizer(
                answer, add_special_tokens=False
            )["input_ids"]

            # Keep the answer ending token in the target sequence.
            eos_id = tokenizer.eos_token_id
            if not answer_ids or answer_ids[-1] != eos_id:
                answer_ids.append(eos_id)

            # Reserve room for at least one answer token.
            prompt_ids = prompt_ids[: MAX_LENGTH - 1]
            remaining = MAX_LENGTH - len(prompt_ids)
            answer_ids = answer_ids[:remaining]

            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids

            if not any(label != -100 for label in labels):
                print(f"Skipping example without answer: {filename}:{line_number}")
                continue

            rows.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    return rows


train_rows = load_examples("train_formatted.jsonl")
val_rows = load_examples("validation_formatted.jsonl")

if not train_rows:
    raise RuntimeError("No usable training examples found.")

if not val_rows:
    raise RuntimeError("No usable validation examples found.")


class SimpleCollator:
    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = []
        labels = []
        attention_masks = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )
            labels.append(item["labels"] + [-100] * pad_len)
            attention_masks.append(
                [1] * len(item["input_ids"]) + [0] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
        }


print("Training examples:", len(train_rows))
print("Validation examples:", len(val_rows))

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_rows,
    eval_dataset=val_rows,
    data_collator=SimpleCollator(),
)

print("Starting LoRA training...")
trainer.train()

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print(f"Training complete. Adapter saved to {ADAPTER_DIR}/")
