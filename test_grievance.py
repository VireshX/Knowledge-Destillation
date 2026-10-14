
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./student_lora_adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

print("Loading trained LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

prompt = """Create a college complaint management system that helps students
report campus problems and monitor how the administration handles them.

Describe the complaint categories, submission form, unique tracking IDs,
status updates, and resolution history."""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

print("\nPreparing input...")
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

input_ids = encoded["input_ids"].to(model.device)
attention_mask = encoded["attention_mask"].to(model.device)

print("\nGenerating grievance portal response...\n")

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=350,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = output_ids[0][input_ids.shape[-1]:]
response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("=" * 60)
print("TRAINED STUDENT MODEL RESPONSE")
print("=" * 60)
print(response)
print("=" * 60)
