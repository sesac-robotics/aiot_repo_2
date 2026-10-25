"""직접 초안과 공통 관측을 모두 통과한 측면 플랭크만 개발 자료에 추가한다."""

import copy
import shutil

import audit


def run() -> dict:
    """규칙 출력과 무관하게 적격 직접 초안·계보를 보존하고 감사한다.

    Returns:
        dict: 추가·보류 ID, 백업 경로와 감사 결과. 동일 원본의 정상·오류는
        같은 train 참여자 그룹이며 새로운 독립 시험으로 취급하지 않는다.

    Raises:
        ValueError: 동결 초안 해시 또는 구조 감사 실패.
        OSError: 파일·백업·계보 저장 실패.
    """
    direct_path = audit.OUT / "plank_side_direct_labels.json"
    direct = audit.read_json(direct_path)
    joints = audit.read_json(audit.OUT / "plank_side_joint_results.json")
    if audit.digest(direct_path) != joints["direct_labels_sha256"]:
        raise ValueError("Direct drafts changed")
    exports = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_side_export_plan_exports.json"
        )["records"]
    }
    candidates = {r["id"]: r for r in direct["reviews"]}
    labels_path = audit.OUT / "supplemental_labels.json"
    labels = audit.read_json(labels_path)
    snapshot = audit.OUT / "snapshots/before_plank_side_candidates"
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for name in [
            "supplemental_labels.json",
            "labels.json",
            "audit.json",
            "development_manifest.json",
            "comparison.json",
        ]:
            shutil.copyfile(audit.OUT / name, snapshot / name)
    template = next(r for r in labels if r["id"] == "youtube_chris_basic")
    added, held = [], []
    for result in joints["results"]:
        identifier = result["id"]
        if not result["common_input_passed"]:
            held.append(dict(id=identifier, reasons=result["holds"]))
            continue
        if any(r["id"] == identifier for r in labels):
            continue
        draft, export = candidates[identifier], exports[identifier]
        if draft["rule_generated_label"] or any(
            x is None for x in draft["draft_levels"]
        ):
            raise ValueError("Direct complete draft required")
        lineage = audit.OUT / "youtube_segments" / f"{identifier}_lineage.json"
        audit.save_json(
            lineage,
            dict(
                original_video_path=export["source_path"],
                original_sha256=export["source_sha256"],
                derived_video_path=export["video_path"],
                derived_sha256=export["video_sha256"],
                original_start_frame=export["start_frame"],
                original_end_frame_exclusive=export["end_frame_exclusive"],
                spatial_crop_xyxy=export["crop_xyxy"],
                frame_numbers_preserved=False,
                mapping="source_frame = clip_frame + original_start_frame",
                pixel_checks=export["pixel_checks"],
                codec="FFV1",
                score_generated=False,
            ),
        )
        row = copy.deepcopy(template)
        row.update(
            id=identifier,
            video_path=export["video_path"],
            video_sha256=export["video_sha256"],
            person="youtube_channel_" + export["person_group"],
            label_group="youtube_channel_"
            + export["person_group"]
            + "_all_sources",
            performance_group=export["source_group"],
            segment=result["segment"],
            criterion_levels=draft["draft_levels"],
            score=draft["total_score"],
            note_ko=draft["note_ko"],
            source_url=export["source_url"],
            source_credit=export["creator"],
            source_title=audit.read_json(
                audit.OUT
                / f"sources/youtube_{export['source_id']}.metadata.json"
            )["title"],
            source_original_video_path=export["source_path"],
            source_original_sha256=export["source_sha256"],
            source_original_start_frame=export["start_frame"],
            source_original_end_frame_exclusive=export["end_frame_exclusive"],
            spatial_crop_xyxy=export["crop_xyxy"],
            lineage_file=str(lineage),
            evidence_images=[draft["evidence_sheet"]],
            direct_label_file=str(direct_path),
            direct_label_sha256=audit.digest(direct_path),
            review_method="원본 개요 48개·해당 구간 36개 시간순 표본과 원본 크기 중앙 장면 직접 검토. 비전문가 AI 초안이며 규칙 정답 아님.",
            subject_selection="단일 측면 수행자. Avid 벽 인쇄 인물은 고정 ROI로 제외. 같은 원본 시범은 동일 train 그룹.",
            independent_final_test=False,
        )
        labels.append(row)
        added.append(identifier)
    audit.save_json(labels_path, labels)
    report = audit.audit()
    if not report["passed"]:
        raise ValueError("Integrated structural audit failed")
    result = dict(
        added_in_this_run=added,
        held=held,
        snapshot=str(snapshot),
        audit_passed=report["passed"],
        rule_used_as_label=False,
        expert_validated=False,
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "plank_side_integration.json", result)
    return result


if __name__ == "__main__":
    print(run())
