
import json
from pathlib import Path

INPUT_FILE = "human_evaluation_template.json"
OUTPUT_FILE = "final_evaluation_results.json"

CRITERIA = [
    "requirement_coverage",
    "factual_correctness",
    "clarity_and_usefulness",
    "completeness",
]


def average(scores):
    values = [scores[c] for c in CRITERIA]

    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not 1 <= v <= 5 for v in values):
        raise ValueError("All scores must be numbers from 1 to 5.")

    return round(sum(values) / len(values), 2)


def main():
    data = json.loads(
        Path(INPUT_FILE).read_text(encoding="utf-8")
    )

    report = []

    for item in data:
        base_avg = average(item["base_scores"])
        student_avg = average(item["student_scores"])

        report.append({
            "task": item["task"],
            "base_average": base_avg,
            "student_average": student_avg,
            "difference": round(student_avg - base_avg, 2),
        })

        print(f"\nTask: {item['task']}")
        print(f"Base Model: {base_avg}/5")
        print(f"Student Model: {student_avg}/5")
        print(f"Difference: {round(student_avg - base_avg, 2)}")

    overall_base = round(
        sum(r["base_average"] for r in report) / len(report), 2
    )
    overall_student = round(
        sum(r["student_average"] for r in report) / len(report), 2
    )

    summary = {
        "task_results": report,
        "overall_base_average": overall_base,
        "overall_student_average": overall_student,
        "overall_difference": round(
            overall_student - overall_base, 2
        ),
        "note": (
            "Preliminary subjective ratings on three tasks; "
            "not a statistically validated benchmark."
        ),
    }

    Path(OUTPUT_FILE).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Overall Results ---")
    print(f"Base Model Average: {overall_base}/5")
    print(f"Student Model Average: {overall_student}/5")
    print(f"Difference: {round(overall_student - overall_base, 2)}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
