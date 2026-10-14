
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "student_lora_adapter"
OUTPUT_FILE = "student_evaluation.json"

PROMPTS = [
    {
        "task": "College Library Management",
        "prompt": (
            "Design a college library management interface. Include "
            "book search, availability status, issue and return dates, "
            "and overdue-book alerts."
        ),
    },
    {
        "task": "Student Fee Payment Portal",
        "prompt": (
            "Design a student fee payment portal. Include fee details, "
            "payment status, pending dues, receipts, and payment history."
        ),
    },
    {
        "task": "Examination Timetable",
        "prompt": (
            "Design a college examination timetable interface. Include "
            "exam dates, times, rooms, filters, and personal schedules."
        ),
    },
]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model and LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda")

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

results = []

for item in PROMPTS:
    messages = [{"role": "user", "content": item["prompt"]}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to("cuda")

    print(f"\nEvaluating: {item['task']}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print(response)

    results.append({
        "task": item["task"],
        "prompt": item["prompt"],
        "response": response,
    })

Path(OUTPUT_FILE).write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")
