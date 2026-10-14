
import json
import urllib.request
import urllib.error

# Step 1: Select the teacher model
MODEL_NAME = "qwen2.5:3b"

# Step 2: Define the domain-specific prompt
prompt = """
Design a modern college management dashboard.

Requirements:
1. Student records
2. Attendance tracking
3. Responsive UI layout
4. Clean and professional design
5. Accessibility and data privacy

Generate a useful, detailed design specification.
"""

# Step 3: Prepare the request payload
payload = {
    "model": MODEL_NAME,
    "prompt": prompt + """
Return a JSON object with exactly these keys:
{
  "domain": "college_dashboard",
  "requirements": [],
  "components": [],
  "color_palette": [],
  "responsive_design": [],
  "accessibility": []
}

Rules:
- Return only valid JSON.
- Use arrays of descriptive strings for each list.
- Do not include Markdown code fences.
- Do not invent actual student personal data.
""",
    "stream": False,
    "format": "json"
}

# Step 4: Configure the Ollama API request
url = "http://localhost:11434/api/generate"

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

# Step 5: Call the teacher model
try:
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Step 6: Extract the generated answer
    teacher_output = result["response"]

    # Step 7: Validate the generated JSON
    design_data = json.loads(teacher_output)

    required_keys = [
        "domain",
        "requirements",
        "components",
        "color_palette",
        "responsive_design",
        "accessibility"
    ]

    for key in required_keys:
        if key not in design_data:
            raise ValueError(f"Missing required key: {key}")

    # Step 8: Display the structured output
    print("\n--- TEACHER LLM JSON OUTPUT ---\n")
    print(json.dumps(design_data, indent=2, ensure_ascii=False))

    # Step 9: Save the output to a JSON file
    with open(
        "teacher_output.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            design_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nOutput saved to teacher_output.json")

except urllib.error.URLError as error:
    print("Could not connect to Ollama.")
    print("Check that Ollama is installed and running.")
    print(error)

except (json.JSONDecodeError, ValueError) as error:
    print("The teacher output failed JSON validation.")
    print(error)

except TimeoutError:
    print("The request timed out. Try again.")
