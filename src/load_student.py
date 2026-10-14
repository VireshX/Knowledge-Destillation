
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading student model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model = model.to("cuda")
model.eval()

print("Model loaded successfully!")
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)

prompt = "Design a simple student management dashboard."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

print("\nStudent model response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
