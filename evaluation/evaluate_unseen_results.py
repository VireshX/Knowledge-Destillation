
import json
from pathlib import Path

INPUT_FILE = "evaluation/unseen_comparison_results.json"
OUTPUT_FILE = "evaluation/unseen_feature_scores.json"

REQUIRED_FEATURES = {
    "Hostel Management": [
        ["room allocation", "room assignment"],
        ["vacancy", "available room"],
        ["check-in", "check in"],
        ["check-out", "check out"],
        ["maintenance request", "maintenance"],
    ],
    "College Transport": [
        ["bus route", "routes"],
        ["stop", "stops"],
        ["schedule", "timetable"],
        ["student registration", "register"],
        ["transport fee", "fee status"],
    ],
    "Canteen Ordering": [
        ["menu", "food items"],
        ["availability", "available"],
        ["cart", "shopping cart"],
        ["order tracking", "track order"],
        ["payment status", "payment"],
    ],
    "Student Attendance Analytics": [
        ["subject-wise", "by subject", "subject"],
        ["monthly trend", "monthly", "month"],
        ["shortage warning", "shortage", "low attendance"],
        ["downloadable report", "download report", "export"],
    ],
    "Faculty Management": [
        ["faculty profile", "faculty details"],
        ["assigned subject", "subjects"],
        ["teaching schedule", "schedule"],
        ["leave request", "leave"],
    ],
    "Placement Portal": [
        ["company profile", "company details"],
        ["eligibility", "eligible"],
        ["application deadline", "deadline"],
        ["interview schedule", "interview"],
    ],
    "Assignment Submission": [
        ["instructions", "assignment details"],
        ["deadline", "due date"],
        ["file upload", "upload"],
        ["submission status", "submitted"],
        ["feedback", "comments"],
    ],
    "Student Grievance": [
        ["complaint category", "category"],
        ["submit complaint", "complaint form", "complaint"],
        ["tracking id", "tracking number"],
        ["status update", "status"],
        ["resolution history", "resolution"],
    ],
    "Laboratory Booking": [
        ["time slot", "available slot"],
        ["equipment", "equipment list"],
        ["booking approval", "approval"],
        ["cancellation", "cancel"],
    ],
    "Event Registration": [
        ["event details", "event description"],
        ["seat availability", "available seats"],
        ["registration deadline", "deadline"],
        ["confirmation receipt", "confirmation"],
    ],
    "Scholarship Portal": [
        ["eligibility", "eligible"],
        ["document upload", "upload documents"],
        ["application status", "status"],
        ["deadline", "due date"],
        ["award notification", "notification"],
    ],
    "Alumni Portal": [
        ["alumni profile", "alumni details"],
        ["graduation", "graduated"],
        ["networking event", "networking"],
        ["mentorship", "mentor"],
        ["contact preference", "contact"],
    ],
    "Course Registration": [
        ["prerequisite", "prerequisites"],
        ["available seats", "seat availability"],
        ["timetable conflict", "schedule conflict", "conflict"],
        ["enrollment status", "enrollment"],
        ["add-drop", "add/drop"],
    ],
    "Inventory Management": [
        ["item quantity", "stock quantity", "quantity"],
        ["stock alert", "low stock", "stock"],
        ["purchase request", "purchase"],
        ["supplier", "vendor"],
        ["issue record", "issue history"],
    ],
    "Campus Health Appointment": [
        ["available slot", "time slot"],
        ["appointment booking", "book appointment"],
        ["cancellation", "cancel"],
        ["visit status", "appointment status", "status"],
    ],
    "Internship Tracker": [
        ["company details", "company"],
        ["application stage", "application status", "stage"],
        ["interview date", "interview"],
        ["offer status", "offer"],
        ["mentor feedback", "feedback"],
    ],
    "Classroom Scheduling": [
        ["room capacity", "capacity"],
        ["available time slot", "time slot"],
        ["course assignment", "course"],
        ["scheduling conflict", "conflict"],
    ],
    "Research Project Portal": [
        ["project proposal", "proposal"],
        ["supervisor assignment", "supervisor"],
        ["milestone deadline", "milestone"],
        ["progress report", "progress"],
    ],
    "Certificate Request": [
        ["certificate type", "types of certificate"],
        ["application form", "form"],
        ["processing status", "status"],
        ["collection date", "collection"],
    ],
    "College Noticeboard": [
        ["announcement", "notice"],
        ["publication date", "published"],
        ["expiry date", "expires", "expiry"],
        ["department filter", "department"],
    ],
}


def score_response(response, feature_groups):
    text = response.lower()
    matched = 0
    details = {}

    for index, keywords in enumerate(feature_groups, start=1):
        found = any(keyword in text for keyword in keywords)
        details[f"feature_{index}"] = {
            "matched": found,
            "keywords": keywords,
        }
        matched += int(found)

    total = len(feature_groups)
    score = round(matched / total * 10, 2)

    return {
        "score_out_of_10": score,
        "matched_features": matched,
        "total_features": total,
        "details": details,
    }


def main():
    results = json.loads(
        Path(INPUT_FILE).read_text(encoding="utf-8")
    )

    report = []

    for item in results:
        task = item["task"]
        features = REQUIRED_FEATURES.get(task)

        if not features:
            print(f"Warning: no feature rules for {task}")
            continue

        base = score_response(item["base_response"], features)
        student = score_response(item["student_response"], features)

        report.append({
            "task": task,
            "base": base,
            "student": student,
            "difference": round(
                student["score_out_of_10"] - base["score_out_of_10"],
                2
            ),
        })

        print(
            f"{task}: Base {base['score_out_of_10']}/10 | "
            f"Student {student['score_out_of_10']}/10"
        )

    if not report:
        raise ValueError("No results were scored.")

    base_avg = round(
        sum(row["base"]["score_out_of_10"] for row in report)
        / len(report), 2
    )
    student_avg = round(
        sum(row["student"]["score_out_of_10"] for row in report)
        / len(report), 2
    )

    output = {
        "tasks_evaluated": len(report),
        "base_average": base_avg,
        "student_average": student_avg,
        "difference": round(student_avg - base_avg, 2),
        "results": report,
        "note": (
            "Keyword-based feature coverage only. It does not verify "
            "correctness, completeness, or overall response quality."
        ),
    }

    Path(OUTPUT_FILE).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n--- Summary ---")
    print("Tasks evaluated:", len(report))
    print("Base average:", base_avg, "/10")
    print("Student average:", student_avg, "/10")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
