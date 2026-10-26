"""추가 실험의 검사·보존 해시·진행 상태와 복구용 소스 사본을 저장한다.

학습·추출·영상 직접 검토를 수행하지 않는다. 앞서 완료한 테스트 로그,
디코딩 기록과 별도로 기록된 직접 검토 범위를 명시하여 자동 검사를 구분한다.
"""

import datetime
import pathlib
import re
import shutil
import subprocess
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

import audit
from cpu_analysis_guard import check_resources


class References(HTMLParser):
    """실행 없이 HTML의 로컬 링크·영상·이미지 경로를 수집한다."""

    def __init__(self):
        """참조 문자열 목록을 초기화한다. 입력·실패 조건은 없다."""
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        """파서가 전달한 태그 속성의 href·src·poster 값을 저장한다.

        Args:
            tag: HTML 태그 이름. 물리 단위 없음.
            attrs: 속성 이름·값의 순서쌍 목록.

        Returns:
            None: 수집 목록만 변경한다. 별도 실패 조건은 없다.
        """
        self.paths.extend(
            value
            for key, value in attrs
            if key in ("href", "src", "poster") and value
        )


def run() -> dict:
    """저장 결과와 동결 기준선을 대조하고 검사·진행 기록을 갱신한다.

    Returns:
        dict: 해시·테스트 수·링크 검사·자원 상태·실험 제한이 있는 기록.
        테스트 수는 실제 저장된 unittest 성공 로그에서 읽는다.

    Raises:
        OSError: 결과·로그·스냅샷 파일 접근 실패.
        ValueError: 기준선 변경, 실패 테스트, 누락 링크 또는 예상 수량 불일치.
        RuntimeError: CPU 통계 보호 조건 위반.
    """
    now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ).isoformat()
    code = pathlib.Path(__file__).resolve().parent
    out = audit.OUT
    frozen = audit.read_json(out / "followup_freeze.json")
    baseline = {
        name: audit.digest(out / name) for name in frozen["baseline_hashes"]
    }
    if baseline != frozen["baseline_hashes"]:
        raise ValueError("Frozen label/model changed")
    rule_hash = audit.digest(code / "rule_score.py")
    if rule_hash != frozen["rule_sha256"]:
        raise ValueError("Common scoring rule changed")
    sources = audit.read_json(out / "audit.json")["source_json_sha256"]
    current = {
        name: audit.digest(audit.SOURCE / (name + ".json")) for name in sources
    }
    if sources != current:
        raise ValueError("Original six source JSON changed")
    test_log = pathlib.Path("/tmp/five-followup-tests.log").read_text()
    tests = re.search(r"Ran (\d+) tests", test_log)
    if not tests or not test_log.strip().endswith("OK"):
        raise ValueError("Successful test log required")
    overlay = audit.read_json(out / "followup_overlay_verification.json")[
        "records"
    ]
    if len(overlay) != 9 or sum(r["decoded_frames"] for r in overlay) != 152:
        raise ValueError("Nine decoded overlays/152 frames required")
    trial_models = [
        out / "followup_W_model/model.joblib",
        out / "followup_event_model/model.joblib",
        *sorted((out / "followup_plank_model").glob("*.joblib")),
    ]
    report = dict(
        updated_at=now,
        source_six_json_unchanged=True,
        source_six_json_sha256=current,
        frozen_common_labels_and_models_unchanged=True,
        frozen_hashes=baseline,
        common_rule_unchanged=True,
        common_rule_sha256=rule_hash,
        trial_model_sha256={
            str(p.relative_to(out)): audit.digest(p) for p in trial_models
        },
        unit_tests_passed=int(tests.group(1)),
        python_source_files=len(list(code.glob("*.py"))),
        compileall_passed=True,
        black_line_length=80,
        decoded_overlay_videos=9,
        decoded_overlay_frames=152,
        directly_viewed_latest_preview_images=9,
        preview_review_evidence="assistant visual tool calls before this finalizer; not inferred by script",
        local_browser_visual_check=False,
        heavy_new_GPU_extractions=15,
        explicit_existing_cache_aliases=1,
        heavy_new_CPU_extractions=2,
        new_GPU_and_CPU_runtime_not_mixed=True,
        original_source_groups_reserved_for_autonomous_test_reused_for_fitting=False,
        new_independent_all_five_final_tests=0,
        expert_reference_labels=0,
        all_five_complete=False,
        common_model_or_rule_replaced=False,
        resource_end=check_resources(),
        free_disk_bytes=shutil.disk_usage(out).free,
        git_branch=subprocess.check_output(
            [
                "git",
                "-C",
                str(audit.BASE / "aiot_repo_2"),
                "branch",
                "--show-current",
            ],
            text=True,
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "-C", str(audit.BASE / "aiot_repo_2"), "status", "--short"],
            text=True,
        ).splitlines(),
        limitations_ko=[
            "직접 등급은 시간순 표본을 본 AI의 비전문가 실험 초안이다.",
            "W 신규 조건부 비교는 4건이며 관측 가능한 0등급 사례가 없다. 자동 같은 시행 비교는 2/5이다.",
            "플랭크 오류 보강 후 총점 재학습·단조·3D·부분 항목 모델이 모두 개선되지 않았다.",
            "상지 사건 모델과 부분 항목 등급·전신 점수는 다른 출력이다.",
            "새 측면 Y 두 구간은 같은 참여자·원본이며 전환 구간은 운동 오류 라벨이 아니다.",
            "반복 개발 검증과 전문가 정답·실제 재촬영·5종 독립 최종 시험을 구분한다.",
        ],
    )
    path = out / "followup_iteration_verification.json"
    audit.save_json(path, report)
    links = {}
    for name in [
        "index.html",
        "unified_status.html",
        "followup_results.html",
        "autonomous_prospective.html",
        "upper_support_event.html",
        "upper_event_nested.html",
        "upper_event_review.html",
        "autonomous_runtime.html",
        "ytw_side_components.html",
    ]:
        parser = References()
        parser.feed((out / name).read_text())
        local = [
            unquote(urlsplit(p).path)
            for p in parser.paths
            if not urlsplit(p).scheme
            and not urlsplit(p).netloc
            and urlsplit(p).path
        ]
        missing = sorted({p for p in local if not (out / p).exists()})
        links[name] = dict(references=len(local), missing=missing)
    report["html_reference_checks"] = links
    if any(r["missing"] for r in links.values()):
        audit.save_json(path, report)
        raise ValueError("Missing local HTML references")
    audit.save_json(path, report)
    progress = audit.read_json(out / "progress.json")
    progress["latest_completed_stage"] = (
        "followup_data_W_plank_retraining_and_validation"
    )
    progress["work_in_progress"] = False
    progress["latest_followup_iteration"] = dict(
        updated_at=now,
        verification=str(path),
        web=str(out / "followup_results.html"),
        source_dispositions=str(out / "followup_source_dispositions.json"),
        full_plank_training_records=12,
        full_plank_training_person_groups=9,
        upper_event_training_records=21,
        upper_event_training_person_groups=18,
        upper_event_positive_person_groups=7,
        W_training_windows=22,
        W_training_person_groups=21,
        new_W_known_drafts=5,
        new_W_conditionally_eligible=4,
        new_W_automatic_same_attempt_pairable=2,
        new_Y_hinge_component_windows=2,
        new_Y_hinge_source_person_groups=1,
        CC_sources_downloaded=7,
        excluded_raws_deleted=2,
        excluded_raw_bytes_freed=15895145,
        canonical_records_unchanged=259,
        supplemental_records_unchanged=30,
        directly_screened_QEVD_clips=42,
        detailed_W_temporal_sheets=8,
        detailed_QEVD_upper_event_temporal_sheets=3,
        direct_review_not_full_video_coverage=True,
        all_five_completed=False,
        common_model_or_rule_replaced=False,
        remaining_issues=report["limitations_ko"],
    )
    progress["latest_checks"] = dict(
        verification=str(path),
        unit_tests=report["unit_tests_passed"],
        links_missing=0,
        all_five_complete=False,
    )
    progress["latest_counts_before_followup"] = progress.get(
        "latest_counts", {}
    )
    progress["latest_counts"] = dict(
        canonical_records=259,
        supplemental_records=30,
        separate_new_full_plank_drafts=3,
        separate_new_W_drafts_including_unknown=8,
        separate_new_upper_event_drafts=5,
        separate_new_Y_hinge_component_drafts=2,
        core_score_models=5,
        independently_validated_all_five_models=0,
        unit_tests=report["unit_tests_passed"],
        new_overlay_videos=9,
        expert_reference_labels=0,
        independent_all_five_final_tests=0,
    )
    replacements = {
        "plank_error_cases_and_more_people_before_model_fit": "plank_more_error_people_before_adopting_retrained_model",
        "real_recapture_consistency_and_controlled_end_to_end_latency": "real_recapture_consistency_not_yet_measured_controlled_runtime_already_measured",
    }
    progress["pending"] = [replacements.get(p, p) for p in progress["pending"]]
    audit.save_json(out / "progress.json", progress)
    snapshot = out / "snapshots/followup_iteration_20261021"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in [*code.glob("*.py"), *code.glob("*.sh")]:
        shutil.copy2(source, snapshot / source.name)
    for source in [
        path,
        out / "progress.json",
        out / "unified_registry.json",
        *out.glob("followup_*.json"),
        *out.glob("plank_*trial_results.json"),
        out / "first_cycle_trial_results.json",
    ]:
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2("/tmp/five-followup-tests.log", snapshot / "unit_tests.log")
    for model in trial_models:
        target = snapshot / model.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, target)
    return report


if __name__ == "__main__":
    print(run()["unit_tests_passed"])
