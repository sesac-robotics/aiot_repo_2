"""직접 검토한 추가 반복을 원본·참여자 분할을 유지한 점수 초안으로 저장한다."""

import copy
import json
import pathlib

import audit
from recovery_prepare import ROOT


def run(batch: int = 0) -> dict:
    """검토 결정의 구간·등급·중복을 검증하고 별도 QEVD 형식 자료를 만든다.

    Args:
        batch: 0 이상 검토 묶음 번호. 준비·직접 결정 파일과 대응한다.

    Returns:
        dict: 새 반복 수와 네 항목이 모두 관측된 초안 수. 전문가 정답이
        아니며 같은 참여자의 반복은 독립 참여자 수에 더하지 않는다.

    Raises:
        OSError: 원본 또는 검토 근거 파일 접근 실패.
        ValueError: 해시·프레임·분할·등급·구간 중복 계약 위반.
    """
    prepared = {
        r["id"]: r
        for r in audit.read_json(ROOT / f"full_repetition_batch_{batch}.json")[
            "records"
        ]
    }
    decisions = audit.read_json(
        ROOT / f"full_repetition_manual_decisions_{batch}.json"
    )
    prior = []
    for path in [
        audit.BASE
        / "output/three-exercise-comparison-20261022/score_annotations.json",
        audit.BASE
        / "output/three-exercise-comparison-20261022/additional_score_annotations.json",
        ROOT / "additional_score_annotations.json",
    ]:
        prior.extend(audit.read_json(path))
    for path in ROOT.glob("full_repetition_score_annotations_*.json"):
        if path.name != f"full_repetition_score_annotations_{batch}.json":
            prior.extend(audit.read_json(path))
    rows = []
    native = {}
    for decision in decisions:
        source = prepared[decision["id"]]
        if (
            source["exercise"] not in ("pushups", "squats", "shoulder")
            or source["split"] != "train"
        ):
            raise ValueError(
                "Only reviewed training exercise records are supported"
            )
        receipt = source["sheets"]["primary"]
        first, last = decision["start_frame"], decision["end_frame"]
        sha = receipt["video_sha256"]
        if sha != audit.digest(pathlib.Path(source["video_path"])):
            raise ValueError("Source hash changed")
        if not receipt["start_frame"] <= first < last <= receipt["end_frame"]:
            raise ValueError("Decision outside reviewed window")
        for old in prior + rows:
            if old["video_sha256"] != sha:
                continue
            if old["person"] != source["person"] or old["split"] != "train":
                raise ValueError("Source person or role mismatch")
            interval = old["segment"]
            if not (
                last < interval["start_frame"] or first > interval["end_frame"]
            ):
                raise ValueError(
                    "Repeated interval overlaps existing annotation"
                )
        row = {
            key: copy.deepcopy(source[key])
            for key in [
                "id",
                "exercise",
                "video_path",
                "video_sha256",
                "person",
                "worker_id",
                "split",
                "labels",
                "labels_descriptive",
                "provenance",
            ]
        }
        row["schema_version"] = "qevd-compatible-assessment-v1"
        grades = decision["levels"]
        if len(grades) != 4 or any(
            type(g) is not int or g not in (0, 1, 2)
            for g in grades
            if g is not None
        ):
            raise ValueError("Invalid four-item grades")
        row["id"] += f"_direct_rep{batch}"
        row["segment"] = dict(
            start_frame=first, end_frame=last, fps=receipt["fps"]
        )
        row["required_input_views"] = {
            "pushups": ["side"],
            "squats": ["side", "synchronized_front"],
            "shoulder": ["front"],
        }[source["exercise"]]
        if source["exercise"] == "squats":
            row["front_path"] = source["front_path"]
            row["front_sha256"] = source["front_sha256"]
        row["evidence_images"] = [
            receipt["evidence_image"],
            *(
                [decision["boundary_image"]]
                if "boundary_image" in decision
                else []
            ),
            *(
                [source["sheets"]["front"]["evidence_image"]]
                if source["exercise"] == "squats"
                else []
            ),
        ]
        for path in row["evidence_images"]:
            if not pathlib.Path(path).is_file():
                raise ValueError("Missing review evidence")
        definitions = {
            "pushups": (
                ["trunk_line", "descent_range", "return_extension", "support"],
                [40, 30, 20, 10],
            ),
            "squats": (
                [
                    "descent_range",
                    "return_extension",
                    "knee_foot",
                    "trunk_balance",
                ],
                [30, 20, 30, 20],
            ),
            "shoulder": (
                [
                    "overhead_reach",
                    "lower_return",
                    "bilateral_timing",
                    "frontal_trunk_hold",
                ],
                [30, 25, 25, 20],
            ),
        }
        criteria, weights = definitions[source["exercise"]]
        row["assessment"] = dict(
            criterion_ids=criteria,
            criterion_weights=weights,
            criterion_levels=grades,
            score=(
                None
                if None in grades
                else sum(w * g / 2 for w, g in zip(weights, grades))
            ),
            rubric_version="project-core-form-v1",
            expert_validated=False,
            label_source="assistant_direct_temporal_sheet_review",
            review_method="24장 시간순 표본 직접 검토. 스쿼트는 동기화 정면·측면. 연속 재생·전문가 검토 아님."
            + (
                " 푸시업 13장 경계판 추가 확인."
                if "boundary_image" in decision
                else ""
            ),
            note_ko=decision["note_ko"],
            unit="observable_project_core_points_0_100",
        )
        row["eligible_for_total_score_training"] = None not in grades
        row["manual_unobservable_items"] = decision.get(
            "manual_unobservable_items", []
        )
        row["attempt_complete"] = decision.get("attempt_complete", True)
        if not row["attempt_complete"]:
            row["assessment"]["score"] = None
            row["eligible_for_total_score_training"] = False
        row["eligible_for_item_training"] = True
        row["score_is_expert_ground_truth"] = False
        row["qevd_video_path"] = (
            "./external/" + source["provenance"]["dataset"] + "/" + sha + ".mp4"
        )
        rows.append(row)
        native[row["qevd_video_path"]] = dict(
            video_path=row["qevd_video_path"],
            labels=row["labels"],
            labels_descriptive=row["labels_descriptive"],
            split="train",
        )
    audit.save_json(
        ROOT / f"full_repetition_score_annotations_{batch}.json", rows
    )
    audit.save_json(
        ROOT / f"full_repetition_fine_grained_labels_{batch}.json",
        list(native.values()),
    )
    return dict(
        new_repetitions=len(rows),
        full_score_drafts=sum(
            r["eligible_for_total_score_training"] for r in rows
        ),
        expert_validated=False,
    )


if __name__ == "__main__":
    import sys

    print(json.dumps(run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)))
