
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "student_lora_adapter"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt = (
    "Design a student attendance dashboard for a college. "
    "Include attendance percentages, subject-wise records, "
    "low-attendance alerts, and a clean user interface."
)

messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to("cuda")


def generate(model):
    model.eval()

    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = result[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


print("\nLoading independent base model...")
base_model = load_base_model()

print("\n=== BASE MODEL ===")
base_response = generate(base_model)
print(base_response)

# Free GPU memory before loading the second model.
del base_model
torch.cuda.empty_cache()

print("\nLoading a fresh model for the LoRA adapter...")
student_base = load_base_model()
student_model = PeftModel.from_pretrained(
    student_base,
    ADAPTER_DIR,
)

print("\n=== FINE-TUNED STUDENT ===")
student_response = generate(student_model)
print(student_response)

print("\n=== BASIC COMPARISON ===")
print("Base response characters:", len(base_response))
print("Student response characters:", len(student_response))
print("Responses exactly identical:", base_response == student_response)
