
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/student_lora_adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

print("Loading trained LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = model.to("cuda")
model.eval()

messages = [
    {
        "role": "user",
        "content": (
            "Design a student attendance dashboard for a college. "
            "Include attendance percentages, subject-wise records, "
            "low-attendance alerts, and a clean user interface."
        ),
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGenerating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print("\nTrained student response:\n")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
