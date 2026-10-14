
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "student_lora_adapter"
INPUT_FILE = "unseen_test_prompts.json"
OUTPUT_FILE = "unseen_comparison_results.json"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

prompts = json.loads(
    Path(INPUT_FILE).read_text(encoding="utf-8")
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = "cuda"

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    ).to(device)
    model.eval()
    return model

def generate_response(model, prompt):
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

results = []

print("GPU:", torch.cuda.get_device_name(0))
print("\nLoading base model...")

base_model = load_base_model()

for index, item in enumerate(prompts, start=1):
    print(f"Base model: {index}/{len(prompts)} - {item['task']}")

    response = generate_response(
        base_model, item["prompt"]
    )

    results.append({
        "task": item["task"],
        "prompt": item["prompt"],
        "base_response": response,
        "student_response": None
    })

del base_model
torch.cuda.empty_cache()

print("\nLoading fine-tuned student model...")

student_base = load_base_model()
student_model = PeftModel.from_pretrained(
    student_base, ADAPTER_DIR
)
student_model.eval()

for index, item in enumerate(results, start=1):
    print(f"Student model: {index}/{len(results)} - {item['task']}")

    item["student_response"] = generate_response(
        student_model, item["prompt"]
    )

    item["responses_identical"] = (
        item["base_response"] == item["student_response"]
    )

Path(OUTPUT_FILE).write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"\nSaved results to {OUTPUT_FILE}")
print("Total prompts evaluated:", len(results))
print(
    "Different response pairs:",
    sum(not item["responses_identical"] for item in results),
    "out of",
    len(results)
)
