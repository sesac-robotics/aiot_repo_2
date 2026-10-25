"""직접 검토·관절 품질을 통과한 새 플랭크 초안 한 개를 개발 자료에 추가한다."""

import copy
import shutil

import audit
import benchmark


def run() -> dict:
    """동결 직접 초안과 공통 입력을 확인한 뒤 보충 라벨 한 개만 추가한다.

    Returns:
        dict: 추가 여부·라벨 출처·분할과 감사 결과 경로. 새 초안은 AI의
        비전문가 직접 검토이며 규칙 결과·전문가 정답·독립 시험이 아니다.

    Raises:
        ValueError: 직접 초안·원본·캐시·관측 계약이 불충분할 때.
        OSError: 입력·백업·결과 접근 실패.
    """
    identifier = "youtube_chris_basic"
    drafts = audit.read_json(audit.OUT / "youtube_direct_labels.json")
    draft = next(r for r in drafts["reviews"] if r["id"] == identifier)
    export = next(
        r
        for r in audit.read_json(audit.OUT / "youtube_segment_exports.json")[
            "records"
        ]
        if r["id"] == identifier
    )
    if (
        draft["draft_levels"] != [2, 2, 2, 2]
        or draft["total_score"] != 100
        or draft["rule_generated_label"]
    ):
        raise ValueError(
            "Expected independently recorded direct experimental draft"
        )
    manifest = dict(
        id=identifier,
        exercise="plank",
        video_sha256=export["video_sha256"],
        segment=dict(
            start_frame=0, end_frame=export["frames"] - 1, fps=export["fps"]
        ),
    )
    _, _, holds = benchmark.load_input(manifest)
    if holds:
        raise ValueError("Common input held: " + ";".join(holds))
    path = audit.OUT / "supplemental_labels.json"
    labels = audit.read_json(path)
    lineage_path = (
        audit.OUT / "youtube_segments/youtube_chris_basic_lineage.json"
    )
    audit.save_json(
        lineage_path,
        dict(
            original_video_path=export["source_path"],
            original_sha256=export["source_sha256"],
            derived_video_path=export["video_path"],
            derived_sha256=export["video_sha256"],
            spatial_crop_xyxy=export["crop_xyxy"],
            original_start_frame=export["start_frame"],
            original_end_frame_exclusive=export["end_frame_exclusive"],
            mapping="source_frame = clip_frame + original_start_frame",
            frame_numbers_preserved=False,
            codec="FFV1",
            pixel_checks=export["pixel_checks"],
            score_generated=False,
        ),
    )
    added = not any(r["id"] == identifier for r in labels)
    if added:
        snapshot = audit.OUT / "snapshots/before_youtube_cc_plank"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in [
            "supplemental_labels.json",
            "labels.json",
            "audit.json",
            "comparison.json",
            "development_manifest.json",
        ]:
            shutil.copyfile(audit.OUT / name, snapshot / name)
        template = next(
            r
            for r in labels
            if r["exercise"] == "plank" and "640237" in r["id"]
        )
        row = copy.deepcopy(template)
        row.update(
            id=identifier,
            video_path=export["video_path"],
            video_sha256=export["video_sha256"],
            person="youtube_chris_ellerby_hemmings",
            label_group="youtube_chris_ellerby_hemmings_all_sources",
            performance_group=export["source_group"],
            segment=manifest["segment"],
            note_ko=draft["note_ko"],
            source_usage_status="YouTube Creative Commons Attribution license (reuse allowed) metadata and original creator checked; local noncommercial educational development",
            source_url=export["source_url"],
            source_terms_url="https://www.youtube.com/t/creative_commons",
            source_notice=None,
            evidence_images=[
                str(audit.OUT / "review/youtube_chris_basic.jpg"),
                str(audit.OUT / "review/youtube_chris_basic_f720.png"),
            ],
            origin_label_file=str(path),
            subject_selection="단일 수행자의 전신 측면 사선 유지 구간. 다른 영상·시점을 합치지 않음.",
            review_method="원본 전체 48개 개요 표본과 29~32초의 36개 시간순 표본 및 f720 원본 크기 장면 직접 검토. 전체 프레임 전문가 검토 아님.",
            review_number=1,
            source_original_video_path=export["source_path"],
            source_original_sha256=export["source_sha256"],
            spatial_crop_xyxy=export["crop_xyxy"],
            lineage_file=str(lineage_path),
            source_credit=export["creator"],
            source_title="Forearm Plank",
            source_original_start_frame=export["start_frame"],
            source_original_end_frame_exclusive=export["end_frame_exclusive"],
            direct_label_file=str(audit.OUT / "youtube_direct_labels.json"),
            direct_label_sha256=audit.digest(
                audit.OUT / "youtube_direct_labels.json"
            ),
        )
        labels.append(row)
        audit.save_json(path, labels)
    else:
        existing = next(r for r in labels if r["id"] == identifier)
        existing["lineage_file"] = str(lineage_path)
        audit.save_json(path, labels)
    report = audit.audit()
    result = dict(
        id=identifier,
        added_in_this_invocation=added,
        integrated=True,
        new_relative_to_snapshot=not any(
            r["id"] == identifier
            for r in audit.read_json(
                audit.OUT
                / "snapshots/before_youtube_cc_plank/supplemental_labels.json"
            )
        ),
        split="train",
        expert_validated=False,
        independent_final_test=False,
        label_source="assistant_direct_temporal_visual_nonexpert_draft",
        rule_result_used_as_label=False,
        audit_path=str(audit.OUT / "audit.json"),
    )
    audit.save_json(audit.OUT / "youtube_plank_integration.json", result)
    return result


if __name__ == "__main__":
    print(run())
