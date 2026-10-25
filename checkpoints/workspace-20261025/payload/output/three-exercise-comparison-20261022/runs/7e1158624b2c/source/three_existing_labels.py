"""직접 검토한 보유 QEVD 구간의 초안을 별도 묶음으로 검증·저장한다.

수치 등급은 manual_decisions.json의 직접 결정을 읽을 뿐 추론하지 않는다.
기존 학습 모델·활성 라벨·플랭크·YTW 결과는 수정하지 않는다.
"""

import collections
import copy
import html
import json
import pathlib

import audit
from three_existing_review import ROOT


def validate_assessment(row: dict) -> None:
    """직접 등급·결측·프레임·관측 근거의 계약을 검사한다.

    Args:
        row: 원본 경로·구간·네 항목의 독립 점수 확장 레코드.

    Returns:
        None: 0/1/2/null 등급, 100점 배점과 총점이 일치하면 정상 반환한다.

    Raises:
        ValueError: 등급·배점·총점·구간·근거 또는 원본 해시 불일치.
        OSError: 원본이나 근거 읽기 실패.
    """
    a = row["assessment"]
    levels, weights = a["criterion_levels"], a["criterion_weights"]
    if len(levels) != 4 or len(weights) != 4 or sum(weights) != 100:
        raise ValueError("Four criteria with 100 points required")
    if any(
        type(x) is not int or x not in (0, 1, 2)
        for x in levels
        if x is not None
    ):
        raise ValueError("Invalid direct grade")
    score = (
        None
        if None in levels
        else sum(w * x / 2 for w, x in zip(weights, levels))
    )
    if a["score"] != score or a["expert_validated"] is not False:
        raise ValueError("Invalid total or expert claim")
    segment = row["segment"]
    if (
        not 0 <= segment["start_frame"] < segment["end_frame"]
        or segment["fps"] <= 0
    ):
        raise ValueError("Invalid frame interval")
    if audit.digest(pathlib.Path(row["video_path"])) != row["video_sha256"]:
        raise ValueError("Original video changed")
    if not all(pathlib.Path(p).is_file() for p in row["evidence_images"]):
        raise ValueError("Direct review evidence missing")


def run() -> dict:
    """직접 결정과 구도 검토를 QEVD 기본 JSON·별도 점수 주석으로 저장한다.

    Returns:
        dict: 부분/총점 초안 수·종목별 분포·보존 해시·사람 분할 검사.
        미검토 전체의 부족 여부나 전문가 정답·학습 완료를 뜻하지 않는다.

    Raises:
        OSError: 입력·출력 접근 실패.
        ValueError: 원본 참여자·분할 누출·주석 계약·보존 해시 불일치.
    """
    prepared = []
    for name in ("preparation.json", "expanded_preparation.json"):
        prepared.extend(audit.read_json(ROOT / name)["records"])
    if (ROOT / "shoulder_preparation.json").exists():
        prepared.extend(
            audit.read_json(ROOT / "shoulder_preparation.json")["records"]
        )
    mapping = {x["id"]: x for x in prepared}
    decisions = audit.read_json(ROOT / "manual_decisions.json")
    originals = {
        pathlib.Path(x["video_path"]).stem: x
        for x in audit.read_json(
            audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
        )
    }
    workers = {
        pathlib.Path(x["video_path"]).stem: x["worker_id"]
        for x in audit.read_json(
            audit.BASE
            / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
        )
    }
    rubrics = audit.read_json(
        audit.BASE / "output/label-review-20261021/all-five-v2/rubrics.json"
    )
    definitions = {
        e: (r["version"], v)
        for r in rubrics[:2]
        for e, v in r["definitions"].items()
    }
    rows = []
    bases = {}
    people = {}
    hashes = {}
    for decision in decisions:
        source = mapping[decision["id"]]
        stem = pathlib.Path(source["path"]).stem
        is_qevd = source["id"].startswith("qevd_")
        native = (
            copy.deepcopy(originals[stem])
            if is_qevd
            else dict(
                video_path="./external/MultiView/"
                + source["preparation"]["video_sha256"]
                + ".mp4",
                labels=source["source_labels"],
                labels_descriptive=[],
                split="train",
            )
        )
        worker = workers[stem] if is_qevd else source["worker_id"]
        if (
            (is_qevd and source["person"] != "qevd_" + str(worker))
            or source["split"] == "test"
            or native["split"] == "test"
        ):
            raise ValueError("Participant or test role mismatch")
        preparation = source["preparation"]
        levels = decision["levels"]
        version, rubric = definitions[source["exercise"]]
        detail = ROOT / "review" / (source["id"] + "_detail.jpg")
        evidence = [preparation["evidence_image"]] + (
            [str(detail)] if detail.exists() else []
        )
        row = dict(
            schema_version="qevd-compatible-assessment-v1",
            id=source["id"] + "_existing_review",
            exercise=source["exercise"],
            video_path=source["path"],
            qevd_video_path=native["video_path"],
            labels=native["labels"],
            labels_descriptive=native["labels_descriptive"],
            split=source["split"],
            person=source["person"],
            worker_id=worker,
            video_sha256=preparation["video_sha256"],
            segment=dict(
                start_frame=decision["start_frame"],
                end_frame=decision["end_frame"],
                fps=preparation["fps"],
            ),
            required_input_views=["front" if is_qevd else "front_oblique"],
            direct_view_verified=True,
            eligible_for_item_training=True,
            eligible_for_total_score_training=None not in levels,
            assessment=dict(
                criterion_ids=[x["id"] for x in rubric["items"]],
                criterion_weights=rubric["weights"],
                criterion_levels=levels,
                score=(
                    None
                    if None in levels
                    else sum(
                        w * g / 2 for w, g in zip(rubric["weights"], levels)
                    )
                ),
                rubric_version=version,
                unit="observable_project_core_points_0_100",
                label_source="assistant_direct_temporal_sheet_review",
                review_method="18/24/36장 시간순 개요 표본, 해당하는 8건은 36장 구간 표본 추가 직접 검토. 연속 재생·전 프레임 검토 아님.",
                note_ko=decision["note_ko"],
                expert_validated=False,
            ),
            provenance=dict(
                dataset="QEVD" if is_qevd else "MultiView",
                provider_labels_are_not_score_targets=True,
                original_split=native["split"],
            ),
            evidence_images=evidence,
            input_status="front_view_feature_and_group_validation_required_before_active_training",
            active_models_updated=False,
        )
        if source["exercise"] in ("pushups", "squats"):
            row["input_status"] = (
                "front_visible_items_only_not_compatible_with_current_side_primary_loader"
            )
        if source["exercise"] == "ytw":
            row["stage_windows"] = {
                "qevd_00104489": {"Y": [19, 34], "T": [43, 68], "W": [77, 93]},
                "qevd_00209671": {"Y": [9, 24], "T": [27, 51], "W": [61, 79]},
                "qevd_00159081": {"Y": [12, 23], "T": [27, 39], "W": [43, 52]},
            }.get(source["id"])
        validate_assessment(row)
        if decision["end_frame"] > preparation["end_frame"]:
            raise ValueError("Segment beyond video")
        for key, registry in [
            (row["person"], people),
            (row["video_sha256"], hashes),
        ]:
            if key in registry and registry[key] != row["split"]:
                raise ValueError("Local split leakage")
            registry[key] = row["split"]
        rows.append(row)
        bases[native["video_path"]] = native
    prior = []
    for folder, names in [
        (audit.OUT, ["labels.json", "supplemental_labels.json"]),
        (
            audit.BASE / "output/three-exercise-comparison-20261022",
            ["score_annotations.json", "additional_score_annotations.json"],
        ),
    ]:
        for name in names:
            prior.extend(audit.read_json(folder / name))
    for row in prior:
        if (
            row.get("person") in people
            and row["split"] != people[row["person"]]
        ):
            raise ValueError("Prior participant role conflict")
    protocol = audit.read_json(
        audit.BASE / "output/three-exercise-comparison-20261022/protocol.json"
    )
    preserved = {
        p: audit.digest(pathlib.Path(p)) == sha
        for p, sha in protocol["preserved_sha256"].items()
    }
    if not all(preserved.values()):
        raise ValueError("Preserved five-exercise result changed")
    report = dict(
        direct_temporal_overview_reviews=len(prepared),
        direct_dense_window_reviews=4
        + (4 if (ROOT / "shoulder_detail_preparation.json").exists() else 0),
        new_draft_segments=len(rows),
        full_score_drafts=sum(
            x["assessment"]["score"] is not None for x in rows
        ),
        partial_score_drafts=sum(
            x["assessment"]["score"] is None for x in rows
        ),
        draft_exercises=dict(collections.Counter(x["exercise"] for x in rows)),
        plank_side_score_drafts_added=0,
        plank_candidates_directly_screened=7,
        plank_screening_reason="정면의 팔·어깨는 보이지만 골반·무릎·발 접점이 투영·가림으로 불분명하다. 측면 기준 총점이나 만점을 만들지 않는다.",
        preserved_sha256_checks=preserved,
        source_person_split_check_passed=True,
        provider_strings_converted_to_numeric_targets=False,
        active_models_updated=False,
        external_discovery_paused=True,
        whole_existing_dataset_exhausted=False,
    )
    audit.save_json(ROOT / "score_annotations.json", rows)
    audit.save_json(ROOT / "fine_grained_labels.json", list(bases.values()))
    audit.save_json(
        ROOT / "media_manifest.json",
        [
            dict(
                video_path=x["qevd_video_path"],
                local_path=x["video_path"],
                sha256=x["video_sha256"],
            )
            for x in rows
        ],
    )
    audit.save_json(
        ROOT / "fine_grained_labels_with_worker_ids.json",
        [
            dict(video_path=x["qevd_video_path"], worker_id=x["worker_id"])
            for x in rows
        ],
    )
    audit.save_json(
        ROOT / "development_splits.json",
        [
            dict(
                video_path=x["qevd_video_path"],
                person=x["person"],
                split=x["split"],
            )
            for x in rows
        ],
    )
    audit.save_json(ROOT / "verification.json", report)
    journal = []
    for source in prepared:
        journal.append(
            dict(
                id=source["id"],
                exercise=source["exercise"],
                person=source["person"],
                split=source["split"],
                evidence_image=source["preparation"]["evidence_image"],
                directly_reviewed=True,
                review_method="시간순 표본 직접 검토; 전 프레임 아님",
                score_draft_written=source["id"]
                in {x["id"] for x in decisions},
                note_ko=(
                    report["plank_screening_reason"]
                    if source["exercise"] == "plank"
                    else next(
                        x["note_ko"]
                        for x in decisions
                        if x["id"] == source["id"]
                    )
                ),
            )
        )
    audit.save_json(ROOT / "direct_review_journal.json", dict(records=journal))
    content = [
        '<!doctype html><meta charset="utf-8"><title>보유 자료 직접 검토</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{width:100%}section{padding:15px;border-bottom:1px solid #bbb}</style><h1>보유 QEVD·다중 구도 자료 검토·라벨 보강</h1><p>32개 후보의 시간순 표본 직접 검토, 25개 초안(총점 3·부분 22). 전문가 정답·독립 시험·재학습 완료가 아닙니다. 기존 모델과 플랭크·YTW 결과는 보존했습니다.</p><p>정면에서 보이는 항목은 보강했지만 현재 측면 중심 입력기에 그대로 넣지 않습니다. 관절 추출·구도별 항목 특징은 별도 JSON에 준비했으며 추가 오류 사례와 관측 개선이 필요합니다. 전체 보유 영상이 부족하다는 결론은 내리지 않았습니다.</p><a href="front_item_features.json">관절 항목 특징</a> · <a href="remaining_work.json">실제 남은 작업</a> · <a href="score_annotations.json">점수 주석</a> · <a href="verification.json">검증</a> · <a href="direct_review_journal.json">직접 검토 기록</a>'
    ]
    for row in rows:
        a = row["assessment"]
        content.append(
            "<section><h2>"
            + html.escape(row["id"])
            + "</h2><p>"
            + html.escape(str(a["criterion_levels"]))
            + " / "
            + html.escape(str(a["score"]))
            + "</p><p>"
            + html.escape(a["note_ko"])
            + '</p><img src="review/'
            + pathlib.Path(row["evidence_images"][-1]).name
            + '"></section>'
        )
    (ROOT / "index.html").write_text("\n".join(content), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
