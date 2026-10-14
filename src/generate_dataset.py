
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

MODEL_NAME = "qwen2.5:3b"
API_URL = "http://localhost:11434/api/generate"

# Save new examples separately to protect the original dataset
OUTPUT_FILE = Path("teacher_dataset_new.jsonl")

prompts = [
    "Design a college student dashboard with attendance tracking.",
    "Design a student profile page with academic performance.",
    "Design a faculty dashboard for managing student records.",
    "Design a mobile-friendly attendance management interface.",
    "Design an admin dashboard with student statistics and charts.",
    "Design a student fee payment and fee history page.",
    "Design a college examination timetable interface.",
    "Design a student assignment submission dashboard.",
    "Design a faculty course and class management interface.",
    "Design a college library book search and issue tracking page.",
    "Design a student result and grade report dashboard.",
    "Design a college admission application tracking interface.",
    "Design a student leave request management page.",
    "Design a placement and internship tracking dashboard.",
    "Design a college notice board and announcements page.",
    "Design a hostel room allocation management interface.",
    "Design a student timetable with upcoming classes.",
    "Design a faculty attendance reporting dashboard.",
    "Design a college event registration interface.",
    "Design an accessible student support services dashboard."
]


def generate_example(task, max_retries=3):
    prompt = f"""
You are an expert UI/UX designer for college management systems.

Task: {task}

Create a detailed, practical design specification for this exact task.

The output must explain:
1. Page layout and component placement.
2. Important UI components and displayed information.
3. Color palette and typography.
4. User interactions and validation.
5. Responsive behavior on mobile and desktop.
6. Accessibility and error handling.

Use specific details relevant to the task.
Explain how the interface should behave.
Avoid generic filler, repeated sentences, and incomplete answers.
The output must be at least 200 characters long.

Return only a valid JSON object with exactly these keys:
{{
  "instruction": "A short task instruction",
  "input": "Specific requirements for the design",
  "output": "A detailed design specification"
}}

All three values must be non-empty strings.
Do not use Markdown code fences.
Do not add text outside the JSON object.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.5,
            "num_predict": 800
        }
    }

    request_data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request, timeout=300
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            example = json.loads(result["response"])

            if not isinstance(example, dict):
                raise ValueError("Response is not a JSON object")

            for key in ["instruction", "input", "output"]:
                value = example.get(key)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Invalid field: {key}")

            output = example["output"].strip()

            bad_outputs = {
                "null",
                "{}",
                "[]",
                "design solution json"
            }

            if output.lower() in bad_outputs:
                raise ValueError("Placeholder output generated")

            if len(output) < 200:
                raise ValueError(
                    f"Output too short: {len(output)} characters"
                )

            return example

        except urllib.error.HTTPError as error:
            details = error.read().decode(
                "utf-8", errors="replace"
            )
            print(f"Attempt {attempt}: HTTP {error.code}")
            print("Ollama details:", details[:300])

        except (
            urllib.error.URLError,
            json.JSONDecodeError,
            ValueError,
            KeyError
        ) as error:
            print(f"Attempt {attempt} failed: {error}")

        if attempt < max_retries:
            time.sleep(attempt * 2)

    return None


# Load only this script's previous generated examples
dataset = []

if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid existing JSONL line.")

existing_tasks = {
    item.get("instruction", "").strip().lower()
    for item in dataset
}

successful = 0

for index, task in enumerate(prompts, start=1):
    print(f"\nProcessing task {index}/{len(prompts)}: {task}")

    example = generate_example(task)

    if example is None:
        print("Generation failed after retries.")
        continue

    task_key = example["instruction"].strip().lower()

    if task_key in existing_tasks:
        print("Duplicate instruction skipped.")
        continue

    dataset.append(example)
    existing_tasks.add(task_key)

    # Save after each successful example
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    successful += 1
    print("Example generated and saved.")

print("\n--- GENERATION REPORT ---")
print("New examples generated this run:", successful)
print("Total examples in new file:", len(dataset))
print("Output file:", OUTPUT_FILE)
