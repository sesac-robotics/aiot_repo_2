"""제공자 오류 라벨을 피드백 문구에 연결합니다. 안내 문구의 연결과 실제 프레임별 정답 라벨링은 별개입니다.

Generate reviewable English feedback for every original QEVD label.
Original raw annotations and free-text feedback are never modified."""

import hashlib
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parent
# category, English example. These templates are not expert-validated targets.
PLANK = {
    "arms facing inward": (
        "error_candidate",
        "Match your forearm direction to the reference plank position.",
    ),
    "arms facing inward and legs and hips on the floor": (
        "error_candidate",
        "Match your forearm direction to the reference and lift your hips and legs into the demonstrated plank position.",
    ),
    "arms facing inward and on knees": (
        "variant_or_error",
        "Check your forearm direction. If a full plank is intended, lift your knees to match the reference; otherwise keep the intended knee-supported variation.",
    ),
    "arms too wide": (
        "error_candidate",
        "Bring your forearms closer together to match the reference plank position.",
    ),
    "average": (
        "ambiguous",
        "Continue with a controlled plank and compare your alignment with the reference; this label does not specify a particular error.",
    ),
    "butt sinking": (
        "error_candidate",
        "Lift your hips toward the straight body line shown in the reference.",
    ),
    "head down": (
        "error_candidate",
        "Bring your head into line with your torso, following the reference neck position.",
    ),
    "head up": (
        "error_candidate",
        "Avoid lifting your head away from your body line; follow the reference neck position.",
    ),
    "high hips": (
        "error_candidate",
        "Lower your hips toward the straight body line shown in the reference.",
    ),
    "legs and hips on the floor": (
        "error_candidate",
        "Lift your hips and legs from the floor to match the demonstrated plank position.",
    ),
    "on knees": (
        "variant_or_error",
        "If a full plank is intended, lift your knees to match the reference; retain knee support if that is the intended variation.",
    ),
    "shaking": (
        "error_candidate",
        "Reset your position and use a controlled hold; pause if you cannot maintain the demonstrated alignment.",
    ),
    "shoulders shrugged up toward the ears": (
        "error_candidate",
        "Keep your shoulders from rising toward your ears and follow the reference shoulder position.",
    ),
    "stopping early": (
        "timing_candidate",
        "Complete the planned hold duration while you can maintain the reference position; pause if needed.",
    ),
}
PUSH = {
    "90 degrees": (
        "reference_attribute",
        "Maintain the demonstrated push-up depth and controlled arm movement.",
    ),
    "butt sinking": (
        "error_candidate",
        "Lift your hips to keep your torso aligned during the push-up.",
    ),
    "child pose cheat": (
        "variant_or_error",
        "Return to the demonstrated push-up position when ready; keep rest positions separate from active repetitions.",
    ),
    "deep": (
        "reference_deviation",
        "Match the demonstrated lower position instead of descending beyond the intended range.",
    ),
    "early stopping": (
        "timing_candidate",
        "Complete the planned repetitions while maintaining the reference form; pause if needed.",
    ),
    "elevated hips": (
        "error_candidate",
        "Lower your hips toward the straight body line shown in the reference push-up.",
    ),
    "fast": (
        "timing_candidate",
        "Slow the movement to match the controlled tempo in the reference.",
    ),
    "head down": (
        "error_candidate",
        "Keep your head aligned with your torso instead of dropping it during the push-up.",
    ),
    "head moving in an out of frame": (
        "visibility",
        "Adjust the camera so your head and full body remain visible throughout the push-up.",
    ),
    "insufficient": (
        "ambiguous",
        "Compare the full repetition with the reference; the source label does not identify a specific joint correction.",
    ),
    "legs on the floor": (
        "variant_or_error",
        "If a full push-up is intended, lift your legs into the reference position; check whether a supported variation is intended.",
    ),
    "moving head": (
        "error_candidate",
        "Keep your head aligned with your torso as your body moves through the push-up.",
    ),
    "narrow": (
        "reference_deviation",
        "For this reference variation, move your hands slightly farther apart to match the demonstrated spacing.",
    ),
    "not visible": (
        "visibility",
        "Adjust the camera so your full body is visible before assessing push-up form.",
    ),
    "shallow": (
        "error_candidate",
        "Lower your body to the demonstrated depth before pressing back up.",
    ),
    "shoulder-width": (
        "reference_attribute",
        "Maintain the hand spacing shown in the reference push-up.",
    ),
    "side to side": (
        "error_candidate",
        "Keep your body centered and reduce side-to-side movement during each repetition.",
    ),
    "sloppy": (
        "ambiguous",
        "Reset and follow the reference movement with control; this label does not identify a specific joint error.",
    ),
    "very wide": (
        "reference_deviation",
        "For this reference variation, bring your hands closer together to match the demonstrated spacing.",
    ),
    "waiting position": (
        "phase_attribute",
        "Begin the next repetition from the reference starting position when ready.",
    ),
    "wide": (
        "variant_or_error",
        "Check your hand spacing against the intended push-up variation before changing it.",
    ),
}
SQUAT = {
    "90 degrees": (
        "reference_attribute",
        "Maintain the squat depth demonstrated in the reference variation.",
    ),
    "back not straight": (
        "error_candidate",
        "Keep your torso controlled and match the back alignment demonstrated in the reference squat.",
    ),
    "hold": (
        "phase_attribute",
        "Maintain the demonstrated bottom hold if it is part of the planned repetition.",
    ),
    "insufficient": (
        "ambiguous",
        "Compare the full squat with the reference; this label does not specify a particular joint correction.",
    ),
    "knees over toes": (
        "source_conflict",
        "Knee-position annotation needs review before a corrective cue can be assigned.",
    ),
    "narrow": (
        "reference_deviation",
        "For this reference squat variation, move your feet slightly farther apart to match the demonstrated stance.",
    ),
    "no obvious issue": (
        "normal_attribute",
        "Maintain the reference squat form and controlled movement.",
    ),
    "not visible": (
        "visibility",
        "Adjust the camera so your full body is visible before assessing squat form.",
    ),
    "over 90 degrees": (
        "reference_deviation",
        "Match the intended reference squat depth; a deeper squat is not automatically an error in every variation.",
    ),
    "plie": (
        "variant_or_error",
        "Confirm the intended squat variation before changing a wide stance or outward-pointing toes.",
    ),
    "shallow": (
        "error_candidate",
        "Lower into the demonstrated squat depth before standing back up.",
    ),
    "shoulder-width": (
        "reference_attribute",
        "Maintain the foot spacing demonstrated in the reference squat.",
    ),
    "starting late": (
        "timing_candidate",
        "Begin the movement with the planned cue or reference timing when ready.",
    ),
    "wide": (
        "reference_deviation",
        "For this reference squat variation, bring your feet closer together to match the demonstrated stance.",
    ),
}
YTW = {
    "arms too low": (
        "error_candidate",
        "Raise your arms to the height demonstrated for the current Y, T, or W position.",
    ),
    "bent legs": (
        "reference_deviation",
        "Match the leg position demonstrated in the intended Y-T-W variation.",
    ),
    "hands flat": (
        "reference_deviation",
        "Match your hand and thumb orientation to the reference Y-T-W position.",
    ),
    "not low enough": (
        "ambiguous",
        "Match the lower position demonstrated for the current Y-T-W phase; confirm which body segment the label refers to before assigning a joint-specific cue.",
    ),
    "thumbs pointed down": (
        "reference_deviation",
        "Rotate your thumbs to the direction demonstrated in the reference Y-T-W position.",
    ),
    "thumbs pointed sideway": (
        "reference_deviation",
        "Adjust your thumb direction to match the reference Y-T-W position.",
    ),
    "too fast": (
        "timing_candidate",
        "Slow down and complete each Y, T, and W position at the demonstrated tempo.",
    ),
    "too slow": (
        "timing_candidate",
        "Match the planned Y-T-W tempo while keeping each position controlled.",
    ),
    "torso parallel to the ground, thumbs pointed up": (
        "variant_or_error",
        "Confirm the intended Y-T-W variation and match its demonstrated torso and thumb positions.",
    ),
}
TABLES = {
    "elbow plank": PLANK,
    "pushups": PUSH,
    "squats": SQUAT,
    "standing ytw": YTW,
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    raw_paths = [
        ROOT / "raw/coach/fine_grained_labels.json",
        ROOT / "raw/coach/feedbacks_short_clips.json",
        ROOT / "raw/annotations/fine_grained_labels_with_worker_ids.json",
    ]
    before = {str(p.relative_to(ROOT)): sha(p) for p in raw_paths}
    annotations = json.loads(raw_paths[0].read_text())
    original_feedback = json.loads(raw_paths[1].read_text())
    original_by_id = {r["video_path"]: r for r in original_feedback}
    labels = sorted({l for r in annotations for l in r["labels"]})
    descriptions = {}
    for r in annotations:
        for l, d in zip(r["labels"], r["labels_descriptive"]):
            descriptions.setdefault(l, set()).add(d)
    mapping = {}
    error_categories = {
        "error_candidate",
        "reference_deviation",
        "timing_candidate",
    }
    for i, label in enumerate(labels):
        exercise, posture = label.split(" - ", 1)
        if posture == "wrong order" and exercise.startswith("standing "):
            category, feedback = (
                "sequence_candidate",
                "Perform the positions in the intended Y, then T, then W order.",
            )
            group = "ytw"
        else:
            category, feedback = TABLES[exercise][posture]
            group = (
                "ytw"
                if exercise.startswith("standing ")
                else {
                    "squats": "squats",
                    "pushups": "pushups",
                    "elbow plank": "elbow_plank",
                }[exercise]
            )
        mapping[label] = {
            "label": label,
            "name_en": posture,
            "exercise_prefix": exercise,
            "exercise_group": group,
            "original_descriptions": sorted(descriptions[label]),
            "category": category,
            "feedback_en": feedback,
            "feedback_origin": "assistant_authored_label_mapping",
            "review_status": "draft",
            "expert_verified": False,
            "is_confirmed_error": False,
        }
        if label == "squats - knees over toes":
            mapping[label][
                "review_note"
            ] = "The short label says knees over toes, but the original description says knees do not go past toes. Preserve the source and do not infer a correction from the short label."
        elif category in {"variant_or_error", "reference_deviation"}:
            mapping[label][
                "review_note"
            ] = "Reference-variation-specific. This attribute is not a universal form error."
        elif category == "ambiguous":
            mapping[label][
                "review_note"
            ] = "Source label is underspecified; no precise biomechanical correction is asserted."
        assert feedback.isascii()
    records = []
    review = []
    for r in annotations:
        entries = [mapping[l] for l in r["labels"]]
        if any(e["category"] == "visibility" for e in entries):
            selected = [e for e in entries if e["category"] == "visibility"]
            rule = "visibility_first"
        elif any(
            e["category"] in error_categories | {"sequence_candidate"}
            for e in entries
        ):
            selected = [
                e
                for e in entries
                if e["category"] in error_categories | {"sequence_candidate"}
            ]
            rule = "specific_cues_before_attributes"
        else:
            selected = entries
            rule = "all_available_reference_or_review_cues"
        feedbacks = list(dict.fromkeys(e["feedback_en"] for e in selected))
        records.append(
            {
                "video_path": r["video_path"],
                "feedbacks": feedbacks,
                "split": r["split"],
            }
        )
        review.append(
            {
                "video_path": r["video_path"],
                "split": r["split"],
                "labels": r["labels"],
                "selected_labels": [e["label"] for e in selected],
                "selection_rule": rule,
                "per_label_feedbacks": [
                    {
                        k: e[k]
                        for k in [
                            "label",
                            "category",
                            "feedback_en",
                            "review_status",
                        ]
                    }
                    for e in entries
                ],
                "original_free_text_feedback_available": r["video_path"]
                in original_by_id,
                "feedback_status": "draft",
                "expert_verified": False,
            }
        )
    write(D / "manifests/feedback_label_mapping.json", list(mapping.values()))
    write(D / "coach/feedbacks_short_clips.json", records)
    write(D / "manifests/feedback_review.json", review)
    assert len(records) == 18652 and len(mapping) == 63
    assert {r["video_path"] for r in records} == {
        r["video_path"] for r in annotations
    }
    assert all(
        set(r) == {"video_path", "feedbacks", "split"} and r["feedbacks"]
        for r in records
    )
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in raw_paths}
    check = {
        "status": "passed",
        "source_raw_files_unchanged": True,
        "source_sha256": before,
        "mapped_labels": len(mapping),
        "covered_clips": len(records),
        "coverage_percent": 100,
        "original_feedback_clips_preserved": len(original_feedback),
        "ytw_clips_with_generated_feedback": sum(
            any(l.startswith("standing ") for l in r["labels"])
            for r in annotations
        ),
        "category_counts": dict(
            Counter(e["category"] for e in mapping.values())
        ),
        "selection_rule_counts": dict(
            Counter(r["selection_rule"] for r in review)
        ),
        "feedback_status": "draft; not expert-validated ground truth",
        "source_conflict_labels": ["squats - knees over toes"],
    }
    write(D / "checks/validation_report.json", check)
    md = [
        "# QEVD English feedback mapping",
        "",
        "63 original label names are preserved. English feedback templates cover all 18,652 local clips, including all 3,597 YTW clips. Original raw annotations and the 7,107 original feedback records are unchanged.",
        "",
        "Templates are authored examples, not expert-validated ground truth or newly observed video diagnoses. No model training or new pose inference was performed.",
        "",
        "## Files",
        "",
        "- `coach/feedbacks_short_clips.json`: same three-field schema as QEVD original feedback.",
        "- `manifests/feedback_label_mapping.json`: exact original label -> category -> English feedback.",
        "- `manifests/feedback_review.json`: per-clip label associations and selection rules.",
        "- `checks/validation_report.json`: coverage and raw source hashes.",
        "",
        "## Cue selection",
        "",
        "Visibility cues take priority when the subject is not visible. Specific error/timing/sequence cues take priority over generic positive attributes. Every original label-to-feedback association is retained in the review sidecar, even when not selected for display.",
        "",
        "Do not classify every label as an error: spacing, depth, phase and exercise variations can be context-specific. The source `squats - knees over toes` conflicts with its original description; it is flagged for review and does not produce a knees-behind-toes instruction.",
        "",
        "## Label dictionary",
        "",
        "| Original label | Category | English feedback |",
        "|---|---|---|",
    ]
    for e in mapping.values():
        md.append(f"|{e['label']}|{e['category']}|{e['feedback_en']}|")
    (D / "README.md").write_text("\n".join(md) + "\n")
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
