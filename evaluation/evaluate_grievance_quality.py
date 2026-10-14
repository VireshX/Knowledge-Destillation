
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/student_lora_adapter"
OUTPUT_FILE = "grievance_quality_comparison.json"
MAX_NEW_TOKENS = 600

PROMPT = (
    "Design a college complaint management system that helps students "
    "report campus problems and monitor how the administration handles them. "
    "Describe complaint categories, the submission form, unique tracking IDs, "
    "status updates, and resolution history. Do not invent policies or rules "
    "that are not specified in the prompt."
)

FEATURES = {
    "Complaint categories": [
        "complaint categories", "academic", "facilities"
    ],
    "Submission form": [
        "submission form", "submit a complaint", "form"
    ],
    "Unique tracking ID": [
        "tracking id", "tracking number", "unique tracking"
    ],
    "Status updates": [
        "status updates", "under review", "in progress", "resolved"
    ],
    "Resolution history": [
        "resolution history", "history of resolutions",
        "actions taken"
    ],
}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [{"role": "user", "content": PROMPT}]
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

def generate(model):
    model.eval()
    device = next(model.parameters()).device
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

def feature_checks(response):
    text = response.lower()
    return {
        feature: any(term in text for term in terms)
        for feature, terms in FEATURES.items()
    }

results = {}

print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
print("Generating base response...")
results["base"] = generate(base_model)

del base_model
torch.cuda.empty_cache()

print("\nLoading student model...")
student_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
student_model = PeftModel.from_pretrained(
    student_base, ADAPTER_DIR
)
print("Generating student response...")
results["student"] = generate(student_model)

comparison = {
    "prompt": PROMPT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "base": {
        "response": results["base"],
        "feature_checks": feature_checks(results["base"]),
    },
    "student": {
        "response": results["student"],
        "feature_checks": feature_checks(results["student"]),
    },
    "manual_review_needed": [
        "Unsupported claims or invented policies",
        "Whether status workflow is adequately explained",
        "Whether resolution history is sufficiently detailed",
        "Whether each response appears truncated",
    ],
}

Path(OUTPUT_FILE).write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nFEATURE CHECKS (keyword-based only)")
for feature in FEATURES:
    print(
        f"{feature}: "
        f"Base={comparison['base']['feature_checks'][feature]}, "
        f"Student={comparison['student']['feature_checks'][feature]}"
    )

print(f"\nFull responses saved to {OUTPUT_FILE}")
print("Review the saved responses manually before drawing conclusions.")
