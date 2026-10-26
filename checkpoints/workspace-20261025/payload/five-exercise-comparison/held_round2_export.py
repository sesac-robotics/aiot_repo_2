"""직접 초안을 QEVD 기본 JSON과 항목 점수 확장 주석으로 연결·감사한다."""

import copy
import json
import pathlib

import audit
from allfive_round1_inputs import DEFINITIONS
from held_round2 import OUT


def export() -> dict:
    """원본·참여자·분할·구간 중복을 검사하고 직접 초안을 내보낸다.

    Returns:
        dict: 원본별 기본 네 필드 JSON 수와 구간별 점수 초안 수.
        합계만 자동 계산하며 새로운 직접 등급이나 전문가 정답을 만들지 않는다.

    Raises:
        ValueError: 기존 원본 역할·주석 겹침·직접 초안 구간 또는 등급 오류.
        OSError: 원본·기존 주석·내보내기 파일 접근 실패.
    """
    sources = {
        r["id"]: r
        for r in audit.read_json(OUT / "dense_preparation.json")["records"]
    }
    sources.update(
        {
            r["id"]: r
            for r in audit.read_json(OUT / "held_plank_preparation.json")[
                "records"
            ]
        }
    )
    old_inputs = audit.read_json(
        OUT.parent / "allfive_round1/candidate_input_manifest.json"
    )
    people = {r["person"]: r["split"] for r in old_inputs}
    source_roles = {s: r["split"] for r in old_inputs for s in r["sources"]}
    history = list(audit.read_json(audit.OUT / "labels.json"))
    for folder in (
        OUT.parent,
        audit.BASE / "output/three-exercise-comparison-20261022",
        OUT.parent / "allfive_round1",
    ):
        for path in folder.glob("*score_annotations*.json"):
            rows = audit.read_json(path)
            if isinstance(rows, list):
                history.extend(rows)
    intervals = {}
    for r in history:
        if r.get("segment") and r.get("video_sha256"):
            intervals.setdefault(r["video_sha256"], []).append(
                (
                    r["segment"]["start_frame"],
                    r["segment"]["end_frame"],
                    r["id"],
                )
            )
    for name in ("youtube_segment_plan.json", "plank_side_export_plan.json"):
        for r in audit.read_json(audit.OUT / name)["records"]:
            intervals.setdefault(r["source_sha256"], []).append(
                (r["start_frame"], r["end_frame_exclusive"] - 1, r["id"])
            )
    native = {
        pathlib.Path(r["video_path"]).stem: r
        for r in audit.read_json(
            audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
        )
    }
    native_workers = {
        pathlib.Path(r["video_path"]).stem: r["worker_id"]
        for r in audit.read_json(
            audit.BASE
            / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
        )
    }
    registry = copy.deepcopy(
        audit.read_json(
            OUT.parent / "allfive_round1/external_worker_registry.json"
        )
    )
    basics = {}
    worker_rows = {}
    extended = []
    decisions = audit.read_json(OUT / "manual_decisions.json")
    for d in decisions["records"]:
        r = sources[d["id"]]
        sha = r["sha256"]
        first, last = d["frames"]
        if (
            people.get(r["person"], "train") != "train"
            or source_roles.get(sha, "train") != "train"
        ):
            raise ValueError("Existing source or person is not training-only")
        if (
            r["split"] != "train"
            or audit.digest(pathlib.Path(r["path"])) != sha
        ):
            raise ValueError("Held source role or fingerprint changed")
        for a, b, identifier in intervals.get(sha, []):
            if not (last < a or first > b):
                raise ValueError(
                    f"Overlapping source intervals: {d['id']} and {identifier}"
                )
        intervals.setdefault(sha, []).append((first, last, d["id"]))
        levels = d["levels"]
        if len(levels) != 4 or any(
            type(g) is not int or g not in (0, 1, 2)
            for g in levels
            if g is not None
        ):
            raise ValueError("Four direct ordinal grades required")
        exercise = r["exercise"]
        ids, weights = DEFINITIONS[exercise]
        complete = d["attempt_complete"] and None not in levels
        total = (
            sum(w * g / 2 for w, g in zip(weights, levels))
            if complete
            else None
        )
        segment = dict(start_frame=first, end_frame=last, fps=r["fps"])
        extended.append(
            dict(
                id=d["id"] + "_held2_score",
                source_id=d["id"],
                exercise=exercise,
                video_path=r["path"],
                video_sha256=sha,
                person=r["person"],
                split="train",
                segment=segment,
                required_input_views=d["views"],
                assessment=dict(
                    criterion_ids=ids,
                    criterion_weights=weights,
                    criterion_levels=levels,
                    total_score=total,
                ),
                attempt_complete=d["attempt_complete"],
                eligible_for_total_score_training=complete,
                label_source="assistant_direct_temporal_sample_nonexpert_draft",
                note_ko=d["note_ko"],
                expert_validated=False,
                rule_generated_label=False,
                evidence=d.get(
                    "review_evidence", r["preparation"]["evidence_image"]
                ),
            )
        )
        if d["id"].startswith("qevd_"):
            key = pathlib.Path(r["path"]).stem
            basic = copy.deepcopy(native[key])
            worker = native_workers[key]
            if basic["split"] == "test" or r["person"] != "qevd_" + str(worker):
                raise ValueError("Native QEVD worker or test role differs")
        else:
            if r["person"] not in registry:
                registry[r["person"]] = max(registry.values()) + 1
            worker = registry[r["person"]]
            basic = dict(
                video_path="./external/held/" + sha + ".mp4",
                labels=r.get("source_labels", ["elbow plank"]),
                labels_descriptive=[],
                split="train",
            )
        basic = {
            k: basic[k]
            for k in ("video_path", "labels", "labels_descriptive", "split")
        }
        basics[sha] = basic
        worker_rows[sha] = dict(basic, worker_id=worker)
    audit.save_json(OUT / "score_annotations.json", extended)
    audit.save_json(OUT / "fine_grained_labels.json", list(basics.values()))
    audit.save_json(
        OUT / "fine_grained_labels_with_worker_ids.json",
        list(worker_rows.values()),
    )
    audit.save_json(OUT / "external_worker_registry.json", registry)
    report = dict(
        direct_score_drafts=len(extended),
        original_records=len(basics),
        same_source_intervals_nonoverlapping=True,
        native_four_fields_preserved=True,
        original_and_person_roles_preserved=True,
        expert_reference=False,
        independent_final_test=False,
        external_discovery=False,
        manual_decisions_sha256=audit.digest(OUT / "manual_decisions.json"),
    )
    audit.save_json(OUT / "annotation_export_audit.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(export(), ensure_ascii=False))
