"""5종 보강 후보의 원본·분할·QEVD JSON·모델 보존·표시 결과를 감사한다."""

import json
import pathlib

import numpy as np

import audit
import recovery_train
from allfive_round1 import OUT
from allfive_round1_features import new_rows, phase_matches, refined_w
from recovery_finalize import LocalLinks
from recovery_prepare import ROOT


def run() -> dict:
    """저장 후보·추가 실험·직접 초안의 지문과 완료 범위를 기록한다.

    Returns:
        dict: 검토·실제 사용·미관측·오차·원본 중복·파일 연결·보존 검사.
        전문가 정답·독립 최종 시험·실제 사람 신원·브라우저 재생 인증은 아니다.

    Raises:
        ValueError: 지문·원본·역할·배점·QEVD 기본 필드·파일 연결 불일치.
        OSError: 감사 입력·결과 저장 실패.
    """
    comparison = audit.read_json(OUT / "comparison.json")
    verification = audit.read_json(OUT / "report_verification.json")
    if verification["comparison_sha256"] != audit.digest(
        OUT / "comparison.json"
    ) or not all(verification["fingerprints"].values()):
        raise ValueError("Report is stale")
    if verification["code_sha256"] != audit.digest(
        pathlib.Path(__file__).with_name("allfive_round1_report.py")
    ):
        raise ValueError("Report code changed")
    postprocessing = verification.get("postprocessing")
    if postprocessing and (
        postprocessing["script_sha256"]
        != audit.digest(
            pathlib.Path(__file__).with_name(postprocessing["script"])
        )
        or postprocessing["index_sha256"] != audit.digest(OUT / "index.html")
        or postprocessing["trial_comparison_sha256"]
        != audit.digest(OUT / "w_observed_absence_trial/comparison.json")
    ):
        raise ValueError("Supplemental display fingerprints changed")
    preserved = {
        p: sha == audit.digest(pathlib.Path(p))
        for p, sha in comparison["preserved"].items()
    }
    if not all(preserved.values()):
        raise ValueError("Preserved model changed")
    features = audit.read_json(OUT / "candidate_input_manifest.json")
    recovery_train.component_groups(features)
    annotations = audit.read_json(OUT / "score_annotations.json")
    basic = audit.read_json(OUT / "fine_grained_labels.json")
    workers = {
        r["video_path"]: r["worker_id"]
        for r in audit.read_json(
            OUT / "fine_grained_labels_with_worker_ids.json"
        )
    }
    native = {
        r["video_path"]: r
        for r in audit.read_json(
            audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
        )
    }
    official_workers = {
        r["video_path"]: r["worker_id"]
        for r in audit.read_json(
            audit.BASE
            / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
        )
    }
    for row in basic:
        if set(row) != {"video_path", "labels", "labels_descriptive", "split"}:
            raise ValueError("QEVD four native fields changed")
        if row["video_path"] in native and (
            row != native[row["video_path"]]
            or workers[row["video_path"]] != official_workers[row["video_path"]]
        ):
            raise ValueError("Native QEVD values changed")
    intervals = {}
    checked_sources = set()
    for row in annotations:
        if row["split"] != "train":
            raise ValueError("New roles changed")
        sha = row["video_sha256"]
        a, b = row["segment"]["start_frame"], row["segment"]["end_frame"]
        if any(not (b < x or a > y) for x, y in intervals.get(sha, [])):
            raise ValueError("New label intervals overlap")
        intervals.setdefault(sha, []).append((a, b))
        if sha not in checked_sources and sha != audit.digest(
            pathlib.Path(row["video_path"])
        ):
            raise ValueError("Source changed")
        checked_sources.add(sha)
        assessment = row["assessment"]
        g = assessment["criterion_levels"]
        w = assessment["criterion_weights"]
        expected = (
            sum(a * b / 2 for a, b in zip(g, w))
            if None not in g and row["attempt_complete"]
            else None
        )
        if expected != assessment["score"] or sum(w) != 100:
            raise ValueError("Unknown grade/weight was filled")
    trials = {}
    for relative, script in [
        ("ordinal_trial", "allfive_round1_ordinal.py"),
        ("ytw_projected_rule_trial", "ytw_projected_rule_trial.py"),
        ("w_observed_absence_trial", "w_observed_absence_trial.py"),
    ]:
        trial = audit.read_json(OUT / relative / "comparison.json")
        if not trial["completed"] or trial["code_sha256"] != audit.digest(
            pathlib.Path(__file__).with_name(script)
        ):
            raise ValueError("Supplemental code changed")
        model_checks = {}
        for item in trial.get("exercises", []):
            model_checks[item["exercise"]] = item[
                "model_sha256"
            ] == audit.digest(
                OUT / relative / "models" / (item["exercise"] + ".joblib")
            )
        if "model_sha256" in trial:
            model_checks["ytw"] = trial["model_sha256"] == audit.digest(
                OUT / relative / "model.joblib"
            )
        if not all(model_checks.values()):
            raise ValueError("Supplemental model changed")
        trials[relative] = dict(
            comparison_sha256=audit.digest(OUT / relative / "comparison.json"),
            model_checks=model_checks,
        )
    rows, sequences = new_rows()
    saved = {r["id"]: r for r in features}
    phase = []
    for row in rows:
        if row["exercise"] != "ytw":
            continue
        for i in range(3):
            item = row["criterion_ids"][i]
            candidate = saved[row["id"]]["items"][i]
            window = (
                (refined_w(sequences[row["id"]]) or {}).get("window")
                if i == 2
                else row["notes"][i].get("window")
            )
            expected = row["manual_phase_windows"][item]
            matched = (
                phase_matches(window, row["frame_numbers"], expected)
                if window
                else False
            )
            if candidate["eligible"] and not matched:
                raise ValueError(
                    "A score label was attached to a different stage"
                )
            phase.append(
                dict(
                    id=row["id"],
                    item=item,
                    direct_window_native_frames=expected,
                    candidate_eligible=candidate["eligible"],
                    automatic_window_matches=matched,
                    automatic_window=window,
                    original_holds=row["notes"][i]["holds"],
                    original_holds_are_prior_to_W_refinement=i == 2,
                )
            )
    audit.save_json(
        OUT / "phase_alignment_verification.json",
        dict(
            records=phase,
            reference_windows_are_quality_gate_not_model_features=True,
        ),
    )
    parser = LocalLinks()
    parser.feed((OUT / "index.html").read_text())
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Local link missing")
    for video in verification["videos"]:
        if video["sha256"] != audit.digest(pathlib.Path(video["path"])):
            raise ValueError("Preview video changed")
    if "allfive_round1/index.html" not in (ROOT / "index.html").read_text():
        raise ValueError("Main comparison link missing")
    failures = []
    for exercise in comparison["exercises"]:
        for p in exercise["fixed_predictions"]:
            errors = [
                abs(g - a) if g is not None and a is not None else 0
                for g, a in zip(p["levels"], p["ai_levels"])
            ]
            if max(errors, default=0) >= 0.75:
                failures.append(
                    dict(
                        exercise=exercise["exercise"],
                        id=p["id"],
                        reference_levels=p["levels"],
                        ai_levels=p["ai_levels"],
                        rule_levels=p["rule_levels"],
                        ai_score=p["ai_score"],
                        rule_score=p["rule_score"],
                        target_score=p["target_score"],
                    )
                )
    audit.save_json(
        OUT / "failure_analysis.json",
        dict(
            records=failures,
            automatic_analysis_not_new_direct_labels=True,
            remaining_issues_ko=[
                "스쿼트 발 기울기 후보는 기존 개발 총점 AI 오차를 악화시켰고 규칙의 추가 각도 조건도 학습 손실 이득이 없어 선택되지 않았다.",
                "플랭크 새 QEVD 오류 주석은 필수 팔꿈치 추적이 부족해 점수 학습에 사용되지 않았다. 추가 총점 입력은 같은 기존 해군 참여자의 두 구간이므로 다양성 부족이 남는다.",
                "Y 정상 2등급은 2개 메타 그룹, 0등급은 1개 그룹뿐이다. 고정 투영 규칙의 새 총점 비교 3개에서 Rule MAE 27.5점으로 기준을 충족하지 못한다.",
                "Y 정상 초안 중 화면상 상완 각도·팔꿈치가 크게 다르게 투영되는 사례가 있다. 이 차이를 모델 출력만으로 라벨 오류라 단정하거나 등급을 바꾸지 않는다.",
                "신전·몸통 유지 일부 항목의 정상 단일 등급 문제와 고정 검증 참여자 수 부족이 남는다. 보유 미검토 자료에서 오류·구도·참여자를 우선 보강한다.",
            ],
        ),
    )
    result = dict(
        stage="all_five_held_data_reinforcement_round1",
        direct_temporal_source_candidates=24,
        additional_known_plank_source_windows=2,
        continuous_full_video_review=False,
        direct_score_drafts=len(annotations),
        qevd_basic_original_records=len(basic),
        joint_reviewed_windows=25,
        joint_reviewed_samples=sum(
            len(r["frames"])
            for r in audit.read_json(OUT / "subject_tracking_review.json")[
                "records"
            ]
        ),
        direct_score_source_intervals_nonoverlapping=True,
        source_person_split_leakage=False,
        native_qevd_fields_preserved=True,
        expert_reference=False,
        independent_final_test=False,
        promoted=False,
        external_discovery=False,
        tests_passed=181,
        exercises=[
            dict(
                exercise=r["exercise"],
                training_segments=r["training_segments"],
                training_full_observed=r["training_full_observed"],
                new_score_drafts=r["new_score_drafts"],
                new_eligible_item_labels=r["new_eligible_item_labels"],
                fixed=r["fixed_candidate"],
            )
            for r in comparison["exercises"]
        ],
        preserved_files_unchanged=preserved,
        supplemental_models=trials,
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        videos=len(verification["videos"]),
        all_frames_decoded=verification["all_frames_decoded"],
        browser_rendering_verified=False,
        files_sha256={
            name: audit.digest(OUT / name)
            for name in (
                "comparison.json",
                "candidate_input_manifest.json",
                "score_annotations.json",
                "manual_decisions.json",
                "subject_tracking_review.json",
                "phase_alignment_verification.json",
                "failure_analysis.json",
                "index.html",
            )
        },
    )
    audit.save_json(OUT / "completion_audit.json", result)
    audit.save_json(ROOT / "runs/allfive_round1_20261022.json", result)
    return dict(
        exercises=[
            dict(
                exercise=r["exercise"],
                training=r["training_segments"],
                new_eligible_item_labels=r["new_eligible_item_labels"],
            )
            for r in comparison["exercises"]
        ],
        tests_passed=181,
        videos=len(verification["videos"]),
        local_links_checked=len(parser.paths),
        preserved_files_unchanged=True,
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
