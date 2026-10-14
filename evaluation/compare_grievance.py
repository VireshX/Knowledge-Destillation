
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/student_lora_adapter"

PROMPT = """Create a college complaint management system that helps students
report campus problems and monitor how the administration handles them.

Describe the complaint categories, submission form, unique tracking IDs,
status updates, and resolution history."""

MAX_NEW_TOKENS = 600

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [{"role": "user", "content": PROMPT}]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

def generate_response(model):
    model.eval()
    device = next(model.parameters()).device

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

print("\nGenerating BASE response...")
base_response = generate_response(base_model)

del base_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nLoading fresh base model for student adapter...")
student_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

student_model = PeftModel.from_pretrained(student_base, ADAPTER_PATH)

print("\nGenerating FINE-TUNED STUDENT response...")
student_response = generate_response(student_model)

print("\n" + "=" * 65)
print("BASE MODEL RESPONSE")
print("=" * 65)
print(base_response)

print("\n" + "=" * 65)
print("FINE-TUNED STUDENT RESPONSE")
print("=" * 65)
print(student_response)

print("\n" + "=" * 65)
print("FEATURE CHECK")
print("=" * 65)

features = {
    "Complaint categories": ["category", "academic", "facilities"],
    "Submission form": ["submission form", "submit", "form"],
    "Unique tracking ID": ["tracking id", "tracking number", "unique id"],
    "Status updates": ["status update", "under review", "in progress", "resolved"],
    "Resolution history": ["resolution history", "history of", "actions taken"]
}

for name, keywords in features.items():
    base_found = any(k in base_response.lower() for k in keywords)
    student_found = any(k in student_response.lower() for k in keywords)
    print(f"{name}: Base={base_found}, Student={student_found}")

print("\nNote: Keyword checks are only a rough indicator, not a quality score.")
