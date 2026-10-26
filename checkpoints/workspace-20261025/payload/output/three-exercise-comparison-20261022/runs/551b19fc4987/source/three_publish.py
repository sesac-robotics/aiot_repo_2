"""3종 비교의 항목 근거·개발 오차·관절 영상과 미충족 기준을 공개한다.

로컬 HTML만 생성하며 팀 서비스·DB·통신 계약을 변경하지 않는다. 이전
플랭크·YTW 산출물의 해시를 검증하고 전문가 정답·독립 시험과 구분한다.
"""

import argparse
import html
import pathlib
import shutil

import cv2
import numpy as np

import audit
import render
import three_pipeline
import three_protocol
import three_qevd_export
import three_split_audit
from resources import check_resources


def overlays() -> dict:
    """운동별 만점·비만점 개발 예제의 입력 관절과 점수를 MP4로 만든다.

    Returns:
        dict: 각 예제·동기 구도의 영상·프레임 수·미리보기·디코딩 검증.
        표시는 사람 제외 개발 예측 또는 고정 개발 검증 예측이며 최종 시험이 아니다.

    Raises:
        OSError: 캐시·원본 접근 또는 결과 저장 실패.
        ValueError: 영상 생성·완전 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    report = audit.read_json(root / "development_results.json")
    inputs = {r["id"]: r for r in three_pipeline.load()}
    receipt = dict(
        records=[],
        model_version=report["label_version"],
        independent_test=False,
    )
    for result in report["exercises"]:
        predictions = (
            result["validation_predictions"]
            or result["training_person_out_predictions"]
        )
        selected = []
        for full in (True, False):
            chosen = next(
                (
                    p
                    for p in predictions
                    if p["target_score"] is not None
                    and (p["target_score"] == 100) == full
                    and p["ai_score"] is not None
                    and p["rule_score"] is not None
                ),
                None,
            )
            if chosen:
                selected.append(chosen)
            else:
                partial = next(
                    (
                        p
                        for p in predictions
                        if p["target_score"] is not None
                        and (p["target_score"] == 100) == full
                    ),
                    None,
                )
                if partial:
                    selected.append(partial)
        for full in (True, False):
            scored = [
                p
                for p in predictions
                if p["target_score"] is not None
                and p["ai_score"] is not None
                and (p["target_score"] == 100) == full
            ]
            if scored:
                worst = max(
                    scored, key=lambda p: abs(p["ai_score"] - p["target_score"])
                )
                if worst["id"] not in {p["id"] for p in selected}:
                    selected.append(worst)
        for predicted in selected:
            row = inputs[predicted["id"]]
            for view, sequence, source in [
                (
                    "side" if row["exercise"] != "shoulder" else "primary",
                    row["primary"],
                    row["manifest"]["video_path"],
                ),
                *(
                    [("front", row["front"], row["manifest"]["front_path"])]
                    if row["front"] is not None
                    else []
                ),
            ]:
                check_resources()
                folder = (
                    root
                    / "overlays"
                    / report["label_version"]
                    / row["id"]
                    / view
                )
                folder.mkdir(parents=True, exist_ok=True)
                # 표시용 사본에는 채점에 사용한 경계 마스크 좌표만 저장한다.
                name = "primary" if row["exercise"] == "shoulder" else "side"
                np.savez_compressed(
                    folder / (name + ".npz"),
                    image_xy=sequence,
                    frame_numbers=row["frames"],
                )
                movie = render.overlay(
                    dict(row["manifest"], video_path=source),
                    dict(predicted, cache_folder=str(folder)),
                )
                cap = cv2.VideoCapture(
                    str(movie), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
                )
                decoded = 0
                preview = folder / "preview.jpg"
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if decoded == len(sequence) // 2:
                            if not cv2.imwrite(str(preview), frame):
                                raise ValueError("Preview write failed")
                        decoded += 1
                finally:
                    cap.release()
                if decoded != len(sequence):
                    raise ValueError("Overlay full decode count differs")
                receipt["records"].append(
                    dict(
                        id=row["id"],
                        exercise=row["exercise"],
                        view=view,
                        path=str(movie),
                        preview=str(preview),
                        decoded_frames=decoded,
                        expected_frames=len(sequence),
                        full_decode_passed=True,
                        prediction=predicted,
                    )
                )
                audit.save_json(root / "overlay_verification.json", receipt)
    return receipt


def publish() -> pathlib.Path:
    """보존 검증·자료 현황·미충족 기준을 JSON과 로컬 비교 화면으로 저장한다.

    Returns:
        pathlib.Path: index.html 절대 경로. 수치 미달을 완료로 바꾸지 않는다.

    Raises:
        OSError: 결과 접근·저장 실패.
        ValueError: 이전 산출물 해시 또는 데이터 분할이 변경된 경우.
    """
    root = three_protocol.OUT
    protocol = audit.read_json(root / "protocol.json")
    preserved = {
        path: audit.digest(pathlib.Path(path)) == expected
        for path, expected in protocol["preserved_sha256"].items()
    }
    if not all(preserved.values()):
        raise ValueError("Preserved five-exercise result changed")
    split = three_split_audit.run()
    export = three_qevd_export.export()
    report = audit.read_json(root / "development_results.json")
    runtime = (
        audit.read_json(root / "runtime_results.json")
        if (root / "runtime_results.json").exists()
        else None
    )
    video = (
        audit.read_json(root / "overlay_verification.json")
        if (root / "overlay_verification.json").exists()
        else dict(records=[])
    )
    if video.get("model_version") != report["label_version"]:
        video = dict(records=[])
    if runtime and runtime.get("model_version") != report["label_version"]:
        runtime = None
    gates = []
    for item in report["exercises"]:
        predictions = item["validation_predictions"]
        paired = item["validation_paired_comparison"]
        metrics = paired["metrics"]
        ai = metrics["ai_score"]
        rule = metrics["rule_score"]
        base = metrics["baseline_score"]
        checks = dict(
            training_people=item["train_people"] >= 8,
            training_segments=item["train_segments"] >= 20,
            fixed_development_validation_people=len(
                {p["person"] for p in predictions}
            )
            >= 4,
            fixed_development_validation_segments=len(predictions) >= 10,
            independent_final_exam=False,
            paired_development_MAE=all(
                m["MAE_points"] is not None and m["MAE_points"] <= 10
                for m in [ai, rule]
            ),
            paired_development_P90=all(
                m["P90_absolute_error_points"] is not None
                and m["P90_absolute_error_points"] <= 20
                for m in [ai, rule]
            ),
            normal_and_error_subgroup_MAE=all(
                group["records"] > 0
                and group["MAE_points"] is not None
                and group["MAE_points"] <= 15
                for method in (ai, rule)
                for group in method["groups"].values()
            ),
            development_assessable_fraction=all(
                item["validation_metrics"][k]["assessed_fraction"] is not None
                and item["validation_metrics"][k]["assessed_fraction"] >= 0.8
                for k in ["ai_score", "rule_score"]
            ),
            AI_beats_same_paired_train_mean=ai["MAE_points"] is not None
            and base["MAE_points"] is not None
            and ai["MAE_points"] < base["MAE_points"],
            error_learning_available_for_every_item=all(
                t["error_people"] >= 2 and len(t["grade_counts"]) >= 2
                for t in item["item_training"]
            ),
            same_input_repeat=all(
                p["same_input_repeat_max_delta_points"] <= 1e-6
                for p in predictions + item["training_person_out_predictions"]
            ),
        )
        timing = (
            next(
                (
                    r
                    for r in runtime["records"]
                    if r["exercise"] == item["exercise"]
                ),
                None,
            )
            if runtime
            else None
        )
        checks["runtime_measured"] = bool(timing and timing.get("repeats"))
        checks["guarded_end_to_end_factor"] = bool(
            timing
            and timing.get("repeats")
            and all(
                r["wall_time_per_input_second"] <= 2 for r in timing["repeats"]
            )
        )
        gates.append(
            dict(
                exercise=item["exercise"],
                checks=checks,
                complete=False,
                pending_or_failed=[k for k, v in checks.items() if not v],
                final_exam_metrics=None,
                engineering_targets_not_expert_standard=True,
            )
        )
    verification = dict(
        model_version=report["label_version"],
        preserved_sha256_checks=preserved,
        plank_and_YTW_preserved=True,
        split_audit_passed=split["passed"],
        json_export=export,
        exercises=gates,
        all_three_complete=False,
        expert_reference=False,
        independent_final_exam_completed=False,
    )
    audit.save_json(root / "acceptance.json", verification)
    run = root / "runs" / report["label_version"]
    (run / "source").mkdir(exist_ok=True)
    for path in pathlib.Path(__file__).parent.glob("three_*.py"):
        shutil.copy2(path, run / "source" / path.name)
    for path in [
        root / "acceptance.json",
        root / "qevd-json/score_annotations.json",
        root / "qevd-json/development_splits.json",
    ]:
        shutil.copy2(path, run / path.name)
    esc = lambda value: html.escape(str(value))
    number = lambda value: "판정 불가" if value is None else f"{value:.2f}"
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>3종 운동 점수 개발 비교</title><style>body{max-width:1180px;margin:auto;padding:24px;background:#101b24;color:#edf3f6;font:16px/1.65 sans-serif}section,details{background:#1b2b38;border-radius:10px;padding:18px;margin:16px 0}a{color:#8dd6ff}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #526370;padding:8px}.scroll{overflow-x:auto}video{max-width:100%;max-height:500px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.notice{border-left:5px solid #ffd17d;padding:16px;background:#302c23}</style><h1>푸시업 · 스쿼트 · 숄더 프레스</h1><p class="notice">MediaPipe Heavy는 관절 추출기이며 별도 AI 회귀 모델과 명시 규칙이 점수를 계산합니다. 라벨은 AI가 시간순 표본을 직접 검토한 비전문가 실험 초안입니다. 직접 정한 구간에서의 조건부 채점 비교이며 전체 영상 자동 구간 인식 완료를 뜻하지 않습니다. 독립 최종 시험·전문가 정확도 검증은 미완료입니다. 플랭크·YTW 기존 결과는 보존했습니다.</p>"""
    ]
    parts.append(
        f'<p>후보 버전 {esc(report["label_version"])} · JSON 연결 영상 {export["videos"]}개 / 구간 주석 {export["scored_segments"]}건. 중복 제외·부분 라벨을 포함하므로 모두 학습 가능한 수는 아닙니다.</p><p><a href="qevd-json/fine_grained_labels.json">QEVD 기본 JSON</a> · <a href="qevd-json/score_annotations.json">항목 점수 주석</a> · <a href="qevd-json/development_splits.json">실제 개발 분할</a> · <a href="acceptance.json">미충족 기준</a> · <a href="development_results.json">전체 비교 수치</a></p>'
    )
    parts.append(
        '<p><a href="iteration_comparison.json">동일 표본의 이전 후보 비교</a> · <a href="input_contract.json">입력·구도·단위 계약</a> · <a href="preserved-five/followup_results.html">보존한 기존 5종 결과</a></p><p class="notice">새 스쿼트 특징은 이전 후보보다 오차가 늘어 개선으로 채택하지 않았습니다. 현재 화면은 최신 실험 후보의 실패 사례까지 공개한 결과이며 기존 공통 모델을 교체하지 않았습니다.</p>'
    )
    names = dict(pushups="푸시업", squats="스쿼트", shoulder="숄더 프레스")
    for result, gate in zip(report["exercises"], gates, strict=True):
        parts.append(
            f'<section><h2>{names[result["exercise"]]}</h2><p>학습 {result["train_segments"]}개 구간 / 참여자 그룹 {result["train_people"]}개. 자료 집단: {esc(result["cohort"])}.</p>'
        )
        parts.append(
            '<div class="scroll"><table><tr><th>검증 방식</th><th>입력 / 공통 오차 표본</th><th>AI MAE</th><th>규칙 MAE</th><th>평균 기준선 MAE</th></tr>'
        )
        for title, key in [
            ("고정 개발 검증", "validation_paired_comparison"),
            (
                "참여자 전체 제외 개발 교차 검증",
                "training_person_out_paired_comparison",
            ),
        ]:
            comparison = result[key]
            m = comparison["metrics"]
            parts.append(
                f'<tr><td>{title}</td><td>{comparison["submitted"]} / {comparison["paired_records"]}</td><td>{number(m["ai_score"]["MAE_points"])}</td><td>{number(m["rule_score"]["MAE_points"])}</td><td>{number(m["baseline_score"]["MAE_points"])}</td></tr>'
            )
        parts.append(
            "</table></div><p>오차 단위: 100점 중 점. 공통 표본에서 세 방법을 비교합니다. 판정률·미판정 수는 전체 입력 기준으로 별도 JSON에 제공합니다. 재촬영 일관성은 검증하지 않았습니다.</p>"
        )
        parts.append(
            "<p>미충족·대기: "
            + esc(", ".join(gate["pending_or_failed"]))
            + "</p><details><summary>항목별 학습 수·오류 참여자 수·규칙 경계</summary><pre>"
            + esc(
                dict(
                    items=result["item_training"],
                    rules=result["rule_boundaries"],
                )
            )
            + "</pre></details>"
        )
        for clip in [
            r for r in video["records"] if r["exercise"] == result["exercise"]
        ]:
            path = pathlib.Path(clip["path"]).relative_to(root).as_posix()
            parts.append(
                f'<p>{esc(clip["id"])} · {esc(clip["view"])} · 개발 예측</p><video controls preload="metadata" src="{esc(path)}"></video>'
            )
        for p in (
            result["validation_predictions"]
            + result["training_person_out_predictions"]
        ):
            difference = (
                abs(p["ai_score"] - p["rule_score"])
                if p["ai_score"] is not None and p["rule_score"] is not None
                else None
            )
            parts.append(
                f'<details><summary>{esc(p["id"])} · 초안 {number(p["target_score"])} / AI {number(p["ai_score"])} / 규칙 {number(p["rule_score"])} / 차이 {number(difference)}</summary><pre>{esc(dict(target_levels=p["target_levels"],ai_levels=p.get("ai_levels"),rule_levels=p.get("rule_levels"),observations=p["observations"],automatic_within_manual_envelope=p["automatic_within_manual_envelope"]))}</pre></details>'
            )
        parts.append("</section>")
    parts.append(
        '<p><a href="runtime_results.json">새 관절 추출 포함 처리 시간</a> · <a href="overlay_verification.json">관절 영상 완전 디코딩 검증</a></p></html>'
    )
    target = root / "index.html"
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["overlays", "publish"])
    args = parser.parse_args()
    print(overlays() if args.mode == "overlays" else publish())
