"""70개 보강 검토의 직접 결정만 QEVD 기본 JSON·독립 점수 주석으로 저장한다."""

import copy
import json
import pathlib

import audit
from recovery_prepare import ROOT


def run() -> dict:
    """직접 결정의 구간·원본·배점을 검증하고 새 주석 묶음을 저장한다.

    Returns:
        dict: 직접 검토·라벨·완전/부분 총점 수. 후보·검토판·프레임 수는
        학습 수에 포함하지 않는다. 기존 공통 라벨·모델은 수정하지 않는다.

    Raises:
        OSError: 원본·근거·기존 분할·출력 접근 실패.
        ValueError: 프레임 범위·참여자·배점·원본 해시·중복 구간 계약 위반.
    """
    prepared = {
        r["id"]: r
        for r in audit.read_json(ROOT / "seventy_batch_preparation.json")[
            "records"
        ]
    }
    decisions = audit.read_json(ROOT / "seventy_manual_decisions.json")
    originals = {
        pathlib.Path(r["video_path"]).stem: r
        for r in audit.read_json(
            audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
        )
    }
    workers = {
        pathlib.Path(r["video_path"]).stem: r["worker_id"]
        for r in audit.read_json(
            audit.BASE
            / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
        )
    }
    rubric_files = audit.read_json(
        audit.BASE / "output/label-review-20261021/all-five-v2/rubrics.json"
    )
    definitions = {
        e: (r["version"], v)
        for r in rubric_files[:2]
        for e, v in r["definitions"].items()
    }
    prior = []
    for name in ["score_annotations.json", "additional_score_annotations.json"]:
        prior.extend(
            audit.read_json(
                audit.BASE / "output/three-exercise-comparison-20261022" / name
            )
        )
    current = {r["id"]: r for r in prior}
    rows = []
    basic = {}
    ranges = {}
    for decision in decisions:
        source = prepared[decision["id"]]
        receipt = source["preparation"]
        sha = receipt["video_sha256"]
        if sha != audit.digest(pathlib.Path(source["path"])):
            raise ValueError("Source changed")
        first, last = decision["start_frame"], decision["end_frame"]
        if not receipt["start_frame"] <= first < last <= receipt["end_frame"]:
            raise ValueError("Decision outside reviewed window")
        if any(not (last < a or first > b) for a, b in ranges.get(sha, [])):
            raise ValueError("Repeated source interval overlaps")
        ranges.setdefault(sha, []).append((first, last))
        e = source["exercise"]
        qevd = source["id"].startswith("qevd_")
        if qevd:
            native = copy.deepcopy(originals[pathlib.Path(source["path"]).stem])
            worker = workers[pathlib.Path(source["path"]).stem]
            if (
                source["person"] != "qevd_" + str(worker)
                or native["split"] == "test"
            ):
                raise ValueError("Native source role mismatch")
        else:
            old = current[source["source_id"]]
            if (
                source["person"] != old["person"]
                or source["split"] != old["split"]
            ):
                raise ValueError("Canonical source role changed")
            prior_range = old["segment"]
            if not (
                last < prior_range["start_frame"]
                or first > prior_range["end_frame"]
            ):
                raise ValueError(
                    "New repetition overlaps prior labeled repetition"
                )
            worker = old["worker_id"]
            native = dict(
                video_path="./external/MultiView/" + sha + ".mp4",
                labels=old["labels"],
                labels_descriptive=old["labels_descriptive"],
                split="train",
            )
        version, rubric = definitions[e]
        levels = decision["levels"]
        weights = rubric["weights"]
        if (
            len(levels) != 4
            or sum(weights) != 100
            or any(
                type(g) is not int or g not in (0, 1, 2)
                for g in levels
                if g is not None
            )
        ):
            raise ValueError("Invalid grades or weights")
        score = (
            None
            if None in levels
            else sum(w * g / 2 for w, g in zip(weights, levels))
        )
        row = dict(
            schema_version="qevd-compatible-assessment-v1",
            id=source["id"] + "_batch1",
            exercise=e,
            video_path=source["path"],
            qevd_video_path=native["video_path"],
            video_sha256=sha,
            person=source["person"],
            worker_id=worker,
            split=source["split"],
            labels=native["labels"],
            labels_descriptive=native["labels_descriptive"],
            segment=dict(start_frame=first, end_frame=last, fps=receipt["fps"]),
            required_input_views=["front"],
            direct_view_verified=True,
            assessment=dict(
                criterion_ids=[i["id"] for i in rubric["items"]],
                criterion_weights=weights,
                criterion_levels=levels,
                score=score,
                rubric_version=version,
                unit="observable_project_core_points_0_100",
                label_source="assistant_direct_temporal_sheet_review",
                review_method="24장 시간순 표본 직접 검토. 연속 재생·전 프레임·전문가 검토 아님.",
                note_ko=decision["note_ko"],
                expert_validated=False,
            ),
            provenance=dict(
                dataset="QEVD" if qevd else "MultiView",
                provider_strings_not_converted_to_scores=True,
                original_split=native["split"],
            ),
            evidence_images=[receipt["evidence_image"]],
            eligible_for_item_training=True,
            eligible_for_total_score_training=score is not None,
        )
        rows.append(row)
        basic[native["video_path"]] = native
    audit.save_json(ROOT / "additional_score_annotations.json", rows)
    audit.save_json(
        ROOT / "additional_fine_grained_labels.json", list(basic.values())
    )
    audit.save_json(
        ROOT / "additional_development_splits.json",
        [
            dict(
                id=r["id"],
                person=r["person"],
                split=r["split"],
                video_path=r["qevd_video_path"],
            )
            for r in rows
        ],
    )
    report = dict(
        directly_reviewed_overviews=len(prepared),
        new_draft_segments=len(rows),
        full_score_drafts=sum(
            r["assessment"]["score"] is not None for r in rows
        ),
        partial_score_drafts=sum(
            r["assessment"]["score"] is None for r in rows
        ),
        native_strings_preserved=True,
        selected_new_intervals_do_not_overlap=True,
        source_participant_roles_preserved=True,
        training_count_not_yet_updated=True,
    )
    audit.save_json(ROOT / "additional_label_verification.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
