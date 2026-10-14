
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/student_lora_adapter"
OUTPUT_FILE = "comparison_results.json"

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
            "exam dates, times, rooms, day/week/month filters, and "
            "personal schedules."
        ),
    },
]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_base():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to("cuda")

def generate(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = result[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

results = []

# Evaluate the base model first.
print("\nLoading base model...")
base_model = load_base()

for item in PROMPTS:
    print("Base:", item["task"])
    response = generate(base_model, item["prompt"])
    results.append({
        "task": item["task"],
        "prompt": item["prompt"],
        "base_response": response,
        "student_response": None,
    })

del base_model
torch.cuda.empty_cache()

# Load a fresh model instance with the trained adapter.
print("\nLoading fine-tuned student...")
student_base = load_base()
student_model = PeftModel.from_pretrained(
    student_base,
    ADAPTER_DIR,
)
student_model.eval()

for result in results:
    print("Student:", result["task"])
    result["student_response"] = generate(
        student_model,
        result["prompt"],
    )
    result["responses_identical"] = (
        result["base_response"] == result["student_response"]
    )

Path(OUTPUT_FILE).write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nSaved comparison to {OUTPUT_FILE}")
for result in results:
    print(
        result["task"],
        "| Identical:",
        result["responses_identical"],
    )
