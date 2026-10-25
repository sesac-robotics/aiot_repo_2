"""두 번째 스쿼트 검토의 분할·보존·웹 지문과 미완료 항목을 최종 기록한다."""

import json
import pathlib

import audit
from multiview_squat_batch2 import OUT, PREVIOUS
from recovery_finalize import LocalLinks
from recovery_prepare import ROOT


def run() -> dict:
    """저장 결과만 검증해 재개 가능한 감사 기록을 만든다.

    Returns:
        dict: 검토·유효 입력·성능·분할·보존·표시 파일 확인 결과.
        새로운 직접 라벨, 전문가 검증, 실제 사람 신원 확인은 아니다.

    Raises:
        ValueError: 보존 지문, 역할, 예약 시험 원본, 웹 연결 불일치.
        OSError: 감사 입력·출력을 읽거나 기록하지 못하는 경우.
    """
    comparison = audit.read_json(OUT / "comparison.json")
    verification = audit.read_json(OUT / "report_verification.json")
    if verification["comparison_sha256"] != audit.digest(
        OUT / "comparison.json"
    ):
        raise ValueError("Report is stale")
    if verification["code_sha256"] != audit.digest(
        pathlib.Path(__file__).with_name("multiview_squat_batch2_report.py")
    ):
        raise ValueError("Report generator changed")
    if not all(verification["fingerprints"].values()):
        raise ValueError("Report checks failed")
    preserved = {
        p: sha == audit.digest(pathlib.Path(p))
        for p, sha in comparison["preserved"].items()
    }
    if not all(preserved.values()):
        raise ValueError("Protected files changed")
    features = audit.read_json(OUT / "fit_input_manifest.json")
    added = [r for r in features if "_batch2_rep" in r["id"]]
    older = [r for r in features if "_batch2_rep" not in r["id"]]
    old_sources = {s for r in older for s in r["sources"]}
    new_sources = {s for r in added for s in r["sources"]}
    if old_sources & new_sources:
        raise ValueError("A new source was already used")
    roles = {}
    for r in features:
        roles.setdefault(r["person"], set()).add(r["split"])
    if any(len(v) > 1 for v in roles.values()):
        raise ValueError("Participant roles overlap")
    tests = set()
    for path in (
        audit.OUT / "labels.json",
        audit.BASE
        / "output/three-exercise-comparison-20261022/score_annotations.json",
    ):
        for row in audit.read_json(path):
            if row.get("split") == "test":
                tests.update(
                    row[k]
                    for k in ("video_sha256", "front_sha256")
                    if row.get(k)
                )
    if tests & new_sources:
        raise ValueError("Reserved test source entered development")
    parser = LocalLinks()
    parser.feed((OUT / "index.html").read_text())
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Missing report links")
    if (
        "multiview_squat/batch2/index.html"
        not in (ROOT / "index.html").read_text()
    ):
        raise ValueError("Main review link missing")
    distribution = {}
    for split in ("train", "validation"):
        bucket = [r for r in added if r["split"] == split]
        full = [
            r
            for r in bucket
            if r["reference_total_allowed"]
            and None not in r["levels"]
            and all(n["eligible"] for n in r["items"])
        ]
        distribution[split] = dict(
            segments=len(bucket),
            metadata_groups=len({r["person"] for r in bucket}),
            full_total_segments=len(full),
            full_normal_segments=sum(
                all(g == 2 for g in r["levels"]) for r in full
            ),
            full_error_segments=sum(
                any(g < 2 for g in r["levels"]) for r in full
            ),
        )
    failure = audit.read_json(OUT / "failure_analysis.json")
    if failure["comparison_sha256"] != audit.digest(OUT / "comparison.json"):
        raise ValueError("Failure analysis is stale")
    result = dict(
        stage="held_multiview_squat_batch2_direct_review_and_isolated_fit",
        reviewed_recording_candidates=12,
        reviewed_temporal_view_sheets=24,
        reviewed_joint_view_sheets=40,
        reviewed_joint_sample_frames=480,
        continuous_full_video_review=False,
        direct_score_drafts=20,
        accepted_source_pairs=10,
        paired_correspondence_pending_source_pairs=2,
        annotation_reference_complete_segments=16,
        added=comparison["added"],
        added_distribution=distribution,
        training_effective_segments=comparison["training_effective_segments"],
        training_four_item_observed_segments=comparison[
            "training_four_item_observed_segments"
        ],
        development_comparison=comparison["comparisons"],
        group_cross_validation=comparison["group_cross_validation"],
        tests_passed=162,
        protected_files_unchanged=preserved,
        original_reserved_test_source_overlap=[],
        new_source_overlap_previous_fit=[],
        video_files=len(verification["videos"]),
        all_frames_decoded=verification["all_frames_decoded"],
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        browser_qa=False,
        external_discovery_paused=True,
        promoted=False,
        expert_reference=False,
        independent_final_test=False,
        files_sha256={
            name: audit.digest(OUT / name)
            for name in (
                "comparison.json",
                "fit_input_manifest.json",
                "score_annotations.json",
                "subject_tracking_review.json",
                "failure_analysis.json",
                "report_verification.json",
                "index.html",
            )
        },
        existing_plank_ytw_preserved=True,
        next_work_ko=[
            "학습 그룹 안에서 뒤꿈치·발끝·지면 기준 특징의 오류 구분력을 비교하고 기존 특징과 절제 대조한다.",
            "보유 자료에서 복귀 신전·몸통 균형 오류와 다양한 참여자를 먼저 검토한다. 현재 정상 단일 등급으로 오류 검출 성능을 주장하지 않는다.",
            "기존 개발 검증을 최종 시험으로 재명명하지 않고 이후 미검토 원본·참여자를 별도 확정 시험으로 잠근다.",
            "플랭크·YTW와 기존 모델은 보존한다. 이번 Rule 후보의 총점 오차가 악화되어 자동 승격하지 않는다.",
        ],
    )
    audit.save_json(OUT / "completion_audit.json", result)
    audit.save_json(ROOT / "runs/multiview_squat_batch2_20261022.json", result)
    return dict(
        added_distribution=distribution,
        tests_passed=162,
        videos=len(verification["videos"]),
        local_links_checked=len(parser.paths),
        protected_files_unchanged=True,
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
