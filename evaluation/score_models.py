
import json
from pathlib import Path

INPUT_FILE = "comparison_results.json"
OUTPUT_FILE = "evaluation_scores.json"

# Expected features for each task
REQUIRED_FEATURES = {
    "College Library Management": {
        "Book Search": ["search"],
        "Availability Status": ["availability", "available"],
        "Issue Dates": ["issue", "issued", "checkout", "checked out"],
        "Return Dates": ["return", "due date"],
        "Overdue Alerts": ["overdue", "alert"],
    },
    "Student Fee Payment Portal": {
        "Fee Details": ["fee", "amount"],
        "Payment Status": ["payment status", "pending", "completed"],
        "Pending Dues": ["pending dues", "pending payments", "outstanding"],
        "Receipts": ["receipt"],
        "Payment History": ["payment history", "past transactions"],
    },
    "Examination Timetable": {
        "Exam Dates": ["exam date", "dates"],
        "Exam Times": ["exam time", "times"],
        "Exam Rooms": ["room", "location"],
        "Day/Week/Month Filters": ["day/week/month filter", "day/week/month", "days/weeks/months"],
        "Personal Schedule": ["personal schedule", "my schedule"],
    },
}


def evaluate_response(task, response):
    text = response.lower()
    features = REQUIRED_FEATURES[task]

    feature_results = {}

    for feature, keywords in features.items():
        found = any(keyword in text for keyword in keywords)
        feature_results[feature] = found

    matched = sum(feature_results.values())
    total = len(features)

    score = round((matched / total) * 10, 2)

    return {
        "score": score,
        "matched_features": matched,
        "total_features": total,
        "feature_results": feature_results,
    }


def main():
    results = json.loads(
        Path(INPUT_FILE).read_text(encoding="utf-8")
    )

    evaluations = []

    for item in results:
        task = item["task"]

        base = evaluate_response(
            task, item["base_response"]
        )

        student = evaluate_response(
            task, item["student_response"]
        )

        evaluations.append({
            "task": task,
            "base": base,
            "student": student,
        })

        print(f"\nTask: {task}")
        print(f"Base Score: {base['score']}/10")
        print(f"Student Score: {student['score']}/10")

    Path(OUTPUT_FILE).write_text(
        json.dumps(evaluations, indent=2),
        encoding="utf-8"
    )

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
