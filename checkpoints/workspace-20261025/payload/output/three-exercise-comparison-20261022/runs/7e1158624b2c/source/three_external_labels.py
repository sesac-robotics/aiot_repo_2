"""직접 작성한 외부 영상 등급 결정을 QEVD 점수 확장으로 연결한다.

스크립트는 등급을 추론하지 않는다. 제공된 검토 결정·배점을 합산하며
원본을 처음부터 디코딩해 정확한 연속 구간을 새 MP4로 만든다. 원본 해시와
참여자 그룹은 구간 사본마다 유지하여 중복·분할 누출을 방지한다.
"""

import pathlib
import subprocess

import cv2

import audit
import external_review
import three_protocol
from resources import check_resources


def run(decisions_name: str = "external_label_decisions.json") -> dict:
    """직접 검토 결정의 프레임 구간을 추출하고 QEVD 호환 주석에 추가한다.

    Args:
        decisions_name: 직접 작성한 검토 결정 JSON의 출력 폴더 상대 이름.

    Returns:
        dict: 새 주석 ID와 구간 사본·원본 연결. null 항목은 총점도 null이다.
        영상 파생 처리는 직접 검토 또는 전문가 검증을 의미하지 않는다.

    Raises:
        OSError: 검토 결정·원본·결과 파일 접근 실패.
        ValueError: 재사용 조건·구간·직접 검토 근거·출력 프레임 수 불일치.
        subprocess.SubprocessError: 영상 인코딩 실패 또는 120초 제한 초과.
        RuntimeError: 온도·전원·메모리 보호 위반.
    """
    root = three_protocol.OUT
    decisions = audit.read_json(root / decisions_name)
    sources = {
        r["id"]: r
        for r in audit.read_json(root / "youtube_download_receipts.json")[
            "records"
        ]
    }
    for filename in [
        "external_discovery_next.json",
        "external_discovery_errors.json",
        "external_discovery_faults.json",
    ]:
        if (root / filename).exists():
            sources.update(
                {
                    r["id"]: r
                    for r in audit.read_json(root / filename)["records"]
                }
            )
    rows = audit.read_json(root / "additional_score_annotations.json")
    registry = audit.read_json(root / "external_worker_registry.json")
    written = []
    for item in decisions:
        check_resources()
        source = sources[item["source_id"]]
        if (
            not source["reusable_metadata_gate"]
            or not item["directly_reviewed"]
            or source["evaluation_role"].startswith("prospective")
        ):
            raise ValueError("Source reuse or direct review missing")
        if source["video_sha256"] != audit.digest(
            pathlib.Path(source["video_path"])
        ):
            raise ValueError("Source changed")
        for evidence in item["evidence_images"]:
            if not pathlib.Path(evidence).is_file():
                raise ValueError("Review evidence missing")
        first, last = item["start_frame"], item["end_frame"]
        if not 0 <= first < last:
            raise ValueError("Invalid frame interval")
        capture = cv2.VideoCapture(
            source["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        fps = capture.get(5)
        capture.release()
        clip = root / f'sources/clips/{item["id"]}_{first}_{last}.mp4'
        clip.parent.mkdir(exist_ok=True)
        if not clip.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    source["video_path"],
                    "-an",
                    "-vf",
                    f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS",
                    "-frames:v",
                    str(last - first + 1),
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(clip),
                ],
                check=True,
                timeout=120,
            )
        cap = cv2.VideoCapture(
            str(clip), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        count = 0
        try:
            while cap.grab():
                count += 1
            output_fps = cap.get(5)
        finally:
            cap.release()
        if count != last - first + 1 or abs(output_fps - fps) > 0.1:
            raise ValueError("Exact clip frame mapping failed")
        clip_sheet = external_review.contact_sheet(
            clip, root / f'review/input_{item["id"]}.jpg', 0, None, 36
        )
        person = "youtube_channel_" + source["channel_id"]
        if person not in registry:
            registry[person] = max(registry.values()) + 1
        exercise = item["exercise"]
        levels = item["levels"]
        weights = (
            [40, 30, 20, 10] if exercise == "pushups" else [30, 25, 25, 20]
        )
        ids = (
            ["trunk_line", "descent_range", "return_extension", "support"]
            if exercise == "pushups"
            else [
                "overhead_reach",
                "lower_return",
                "bilateral_timing",
                "frontal_trunk_hold",
            ]
        )
        row = dict(
            schema_version=three_protocol.SCHEMA,
            id=item["id"],
            exercise=exercise,
            video_path=str(clip),
            video_sha256=audit.digest(clip),
            source_original_sha256=source["video_sha256"],
            source_original_path=source["video_path"],
            source_original_segment=dict(
                start_frame=first, end_frame=last, fps=fps
            ),
            person=person,
            worker_id=registry[person],
            split=(
                "validation"
                if source["evaluation_role"] == "development_validation"
                else "train"
            ),
            labels=[],
            labels_descriptive=[],
            segment=dict(start_frame=0, end_frame=count - 1, fps=output_fps),
            assessment=dict(
                criterion_ids=ids,
                criterion_weights=weights,
                criterion_levels=levels,
                score=(
                    None
                    if None in levels
                    else sum(
                        w * g / 2 for w, g in zip(weights, levels, strict=True)
                    )
                ),
                rubric_version=(
                    "project-core-form-v1"
                    if exercise == "pushups"
                    else "project-other-three-core-v1"
                ),
                unit="observable_project_core_points_0_100",
                label_source="assistant_direct_temporal_sheet_review",
                review_method="명시한 시간순 표본을 직접 검토한 비전문가 초안. 원본 제공자 문자열 또는 규칙·모델 출력으로 등급을 만들지 않음.",
                note_ko=item["note_ko"],
                expert_validated=False,
            ),
            provenance=dict(
                dataset="YouTube",
                source_url=source["source_url"],
                creator=source["creator"],
                license=source["license"],
                channel_id=source["channel_id"],
                original_provider_strings_not_score_targets=True,
                identity_grouping="conservative_channel_group_not_verified_unique_physical_person",
                frame_mapping="ffmpeg sequential decode from source frame zero and exact trim; local frame zero maps to original start_frame",
            ),
            evidence_images=item["evidence_images"],
            direct_view_verified=True,
            required_input_views=item["required_input_views"],
            eligible_for_item_training=True,
            eligible_for_total_score_training=None not in levels,
            independent_final_test=False,
            reviewed_before_new_Heavy_and_model_predictions=True,
        )
        row["evaluation_role"] = "development_" + row["split"]
        row["attempt_status"] = item.get(
            "attempt_status", "complete_repetition"
        )
        row["input_clip_review_sheet_prepared"] = clip_sheet["evidence_image"]
        three_protocol.validate_record(row)
        rows = [r for r in rows if r["id"] != row["id"]] + [row]
        written.append(
            dict(id=row["id"], original_frame_count=count, clip=str(clip))
        )
        audit.save_json(root / "additional_score_annotations.json", rows)
        audit.save_json(root / "external_worker_registry.json", registry)
    report = dict(
        records=written, script_inferred_grades=False, expert_validated=False
    )
    audit.save_json(root / "external_label_merge_verification.json", report)
    return report


if __name__ == "__main__":
    print(len(run()["records"]))
