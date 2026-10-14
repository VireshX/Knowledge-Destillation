
import json
from pathlib import Path

INPUT_FILE = "comparison_results.json"
OUTPUT_FILE = "human_evaluation_template.json"

CRITERIA = [
    "requirement_coverage",
    "factual_correctness",
    "clarity_and_usefulness",
    "completeness",
]


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    results = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    evaluation = []

    for item in results:
        evaluation.append({
            "task": item["task"],
            "prompt": item["prompt"],
            "base_response": item["base_response"],
            "student_response": item["student_response"],
            "base_scores": {
                criterion: None for criterion in CRITERIA
            },
            "student_scores": {
                criterion: None for criterion in CRITERIA
            },
            "review_notes": "",
        })

    Path(OUTPUT_FILE).write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_FILE}")
    print("Rate every criterion from 1 to 5.")
    print("1 = Poor, 2 = Weak, 3 = Adequate,")
    print("4 = Good, 5 = Excellent")


if __name__ == "__main__":
    main()
