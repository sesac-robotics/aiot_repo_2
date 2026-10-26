"""로컬 비교 결과를 점수·근거·관절 표시 영상이 있는 HTML로 내보낸다.

HTTP API·팀 UI·DB 계약은 변경하지 않는다. 관절 영상은 개발 검증 예제만
만들며, 표시 점수는 검증 전 초안임을 명시한다.
"""

import html
import json
import pathlib
import subprocess
import time

import cv2
import numpy as np

import audit
from resources import check_resources

CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
)


def overlay(row: dict, comparison: dict) -> pathlib.Path:
    """개발 검증 구간의 10Hz 관절 표시 MP4를 저장하고 경로를 반환한다.

    Args:
        row: 원본 영상·구간이 있는 감사 manifest 레코드.
        comparison: AI·규칙 점수와 캐시 경로가 있는 비교 레코드.

    Returns:
        pathlib.Path: H.264로 저장된 MP4 절대 경로.

    Raises:
        OSError: 캐시·영상 읽기 또는 출력 저장 실패.
        ValueError: 프레임 읽기 또는 임시 영상 기록 실패.
        subprocess.SubprocessError: ffmpeg 변환 실패 또는 60초 초과.
        RuntimeError: 온도·전원·메모리 보호 조건을 위반한 경우.
    """
    folder = pathlib.Path(comparison["cache_folder"])
    primary = (
        "side"
        if row["exercise"] in ("pushups", "squats", "plank")
        else "primary"
    )
    with np.load(folder / (primary + ".npz")) as data:
        sequence = data["image_xy"]
        indices = data["frame_numbers"]
    check_resources()
    source = cv2.VideoCapture(
        row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    temporary = folder / "overlay_raw.mp4"
    target = folder / "overlay.mp4"
    writer = None
    try:
        source.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        decoded_index = int(indices[0]) - 1
        next_check = time.monotonic() + 2
        for sample_index, (index, points) in enumerate(zip(indices, sequence)):
            if time.monotonic() >= next_check:
                check_resources()
                next_check = time.monotonic() + 2
            while decoded_index < int(index):
                if not source.grab():
                    raise ValueError("Sequential overlay decoding failed")
                decoded_index += 1
            ok, frame = source.retrieve()
            if not ok:
                raise ValueError("Overlay frame unavailable")
            height, width = frame.shape[:2]
            scale = min(1, 800 / max(height, width))
            frame = cv2.resize(
                frame, (round(width * scale), round(height * scale))
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise ValueError("Overlay writer unavailable")
            for start, end in CONNECTIONS:
                if np.isfinite(points[[start, end]]).all():
                    a, b = (points[[start, end]] * scale).round().astype(int)
                    cv2.line(frame, tuple(a), tuple(b), (80, 240, 100), 2)
            for point in points:
                if np.isfinite(point).all():
                    cv2.circle(
                        frame,
                        tuple((point * scale).round().astype(int)),
                        3,
                        (0, 200, 255),
                        -1,
                    )
            rule = comparison["rule_score"]
            label = (
                (
                    f"AI {comparison['ai_score']:.1f} | Rule "
                    if comparison["ai_score"] is not None
                    else "AI N/A | Rule "
                )
                + ("UNKNOWN" if rule is None else f"{rule:.1f}")
                + " | EXPERIMENTAL"
            )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (20, 20, 20), -1)
            cv2.putText(
                frame,
                label,
                (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if row["exercise"] == "ytw":
                phases = (
                    comparison["rule_details"]
                    .get("evidence", {})
                    .get("phase_detection", {})
                    .get("phases", {})
                )
                stage = next(
                    (
                        name
                        for name, window in phases.items()
                        if window["start_index"]
                        <= sample_index
                        <= window["end_index"]
                    ),
                    "transition / unknown",
                )
                morphology = (
                    comparison["rule_details"]
                    .get("evidence", {})
                    .get("w_shape", {})
                    .get("morphology", {})
                )
                w_grade = morphology.get("grade")
                stage += " | W item " + (
                    "UNKNOWN" if w_grade is None else str(w_grade) + "/2"
                )
                cv2.putText(
                    frame,
                    "Stage: " + stage + " | attempt, not form approval",
                    (7, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (60, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        source.release()
        if writer is not None:
            writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "fast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
        timeout=60,
    )
    temporary.unlink()
    return target


def render() -> pathlib.Path:
    """현재 감사·개발 비교 결과와 대표 관절 영상을 HTML로 저장한다.

    Returns:
        pathlib.Path: 실험 index.html 절대 경로.

    Raises:
        OSError: 입력 읽기 또는 HTML 저장 실패.
        ValueError: 검증 예제 관절 영상 생성 실패.
        subprocess.SubprocessError: 영상 인코딩 실패.
    """
    report = audit.read_json(audit.OUT / "comparison.json")
    checked = audit.read_json(audit.OUT / "audit.json")
    manifests = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "development_manifest.json")
    }
    parts = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>5종 운동 · AI / Rule Base 개발 비교</title>
<style>body{background:#101b24;color:#edf3f6;font:16px/1.65 sans-serif;
max-width:1180px;margin:auto;padding:28px}a{color:#8dd6ff}section{background:
#1b2b38;padding:24px;margin:22px 0;border-radius:14px}table{width:100%;min-width:640px;
border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:
1px solid #425464}video{max-width:100%;width:560px}small{color:#c4cdd5}
pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{line-height:1.3}
.notice{border-left:5px solid #ffd17d;padding:16px;background:#302c23}
.table-scroll{overflow-x:auto}td:first-child{overflow-wrap:anywhere}
@media(max-width:600px){body{padding:12px}section{padding:16px}}
</style><h1>5종 운동 · AI / Rule Base 개발 비교</h1>
<p class="notice">실험 진행 중 · 전문가 정답·독립 최종 시험·운영 성능 아님.<br>
MediaPipe Heavy는 관절 추출기이며, AI 점수는 별도 ExtraTrees 회귀 모델이
예측합니다. 규칙 출력은 AI 학습 정답으로 사용하지 않습니다.</p>
<p>두 방식은 같은 구간·관절 입력과 항목 배점을 사용합니다. 보이지 않는
항목은 보류합니다. 규칙의 각도·접지 대리값은 검증 전 초안입니다.</p>
<p><a href="unified_status.html">최신 통합 현황</a> · <a href="plank_support_enrichment.html">팔꿈치·지지 지속성 보강</a> · <a href="plank_partial_items.html">부분 영상 항목별 AI·규칙 검증</a> · <a href="plank_improvement_trial.html">플랭크 오류 개선 실험</a> · <a href="plank_side_progress.html">최신 플랭크 측면·AI 개발 비교</a> · <a href="plank_error_review.html">추가 플랭크 오류·W 시간 집계</a> · <a href="youtube_review.html">YouTube 구간 분리·관절 검토</a> · <a href="plank_static.html">플랭크 정지 정렬 항목 AI/규칙 비교</a> · <a href="aihub_review.html">AI-Hub 자료 검토</a> · <a href="w_analysis.html#item-errors">항목별 오차·상쇄</a> · <a href="w_inspection.html">W 추가 관절·구간 검토</a> · <a href="w_analysis.html">W 원인 분석·오류 자료 보충</a> · <a href="phases.html">Y–T–W 직접 표시·자동 단계 비교</a> · <a href="audit.json">감사 결과</a> · <a href="contracts.json">채점 기준</a>
 · <a href="comparison.json">비교 원본</a> ·
<a href="new_screening_decisions.json">추가 QEVD 구도 검토 16개</a></p>"""]
    names = {
        "pushups": "푸시업",
        "squats": "스쿼트",
        "plank": "팔꿈치 플랭크",
        "ytw": "전경 Y–T–W",
        "shoulder": "덤벨 숄더 프레스",
    }
    for exercise, result in report["results"].items():
        count = checked["exercises"][exercise]
        parts.append(f"<section><h2>{names[exercise]}</h2>")
        parts.append(
            f'<p>학습 분할의 관측 통과 자료 {result["train_records"]}개 / '
            f'{result["train_people"]}명 · 개발 검증 관측 통과 '
            f'{result["validation_records"]}개 · 독립 최종 시험 0개</p>'
            f'<p>전체 분할의 라벨·구도·파일 조건 통과 후보 {count["extraction_ready"]}개 '
            f"(학습·검증·시험 합계, 관절 관측 필터 적용 전). 모델 상태: "
            + (
                "개발 실험 모델 · 검증 미완료"
                if result["model_status"] == "unvalidated_development_model"
                else "학습 자료·다양성 부족"
            )
            + "</p>"
        )
        diagnostic = result.get("training_participant_diagnostic")
        if diagnostic:
            parts.append(
                "<p>학습 참여자별 제외 개발 진단 "
                f'{diagnostic["people"]}명 / {diagnostic["samples"]}개: '
                f'AI 평균 절대오차 {diagnostic["ai_MAE_points"]:.2f}점 · '
                f'학습 평균 기준선 {diagnostic["baseline_MAE_points"]:.2f}점. '
                "설정 자동 조정·독립 최종 시험에 사용하지 않았습니다.</p>"
            )
        metrics = result.get("metrics")
        if metrics:
            parts.append(
                "<p>동일 표본에서 함께 총점을 낸 비교 자료: "
                f'{metrics["paired_samples"]}개. '
                "이 작은 개발 자료로 우열을 확정하지 않습니다.</p>"
            )
            for key, label in (
                ("ai_MAE_points", "AI 평균 절대 오차"),
                ("rule_MAE_points", "규칙 평균 절대 오차"),
                ("train_mean_baseline_MAE_points", "학습 평균 점수 기준 오차"),
            ):
                value = metrics[key]
                text = "판정 자료 없음" if value is None else f"{value:.2f}점"
                parts.append(f"<p>{label}: {text}</p>")
            for key, label in (
                ("ai_median_ms", "AI 계산 중앙 시간"),
                ("rule_median_ms", "규칙 계산 중앙 시간"),
            ):
                value = metrics[key]
                text = "자료 없음" if value is None else f"{value:.2f}ms"
                parts.append(
                    f"<p>{label}: {text} (특징 계산 포함, 영상 읽기·Heavy 추출 제외; 11회 중앙값)</p>"
                )
            total = metrics["eligible_validation_labels"]
            parts.append(
                f"<p>개발 검증 라벨 {total}개 중 AI 평가 "
                f'{metrics["ai_assessed"]}개 / 규칙 평가 '
                f'{metrics["rule_assessed"]}개</p>'
            )
        if exercise == "ytw" and result.get("item_metrics"):
            parts.append(
                '<h3>관측 가능한 항목별 비교</h3><p><a href="ytw_w_failure_cases.json">현재 W 불일치</a> · <a href="snapshots/pre_w_contract_fix/ytw_w_failure_cases.json">이전 실패 보존본</a> · W는 화면 90도 조건 대신 굽힘·손 위치·팔꿈치 상승·좌우 모양 개발 규칙으로 항목 등급을 계산합니다. 전문가 검증 전이며 Y·전경 미관측을 채우지 않아 총점은 보류합니다. · 단계 식별 성공을 자세 등급 검증 성공으로 간주하지 않습니다.</p><p>항목 등급 오차 단위는 0~2등급입니다. 보류 항목을 제외한 값을 100점으로 환산하지 않습니다.</p><div class="table-scroll"><table><tr><th>항목</th><th>함께 평가한 수</th><th>AI 등급 오차</th><th>규칙 등급 오차</th></tr>'
            )
            for metric in result["item_metrics"]:
                fmt = lambda value: "보류" if value is None else f"{value:.2f}"
                parts.append(
                    f'<tr><td>{html.escape(metric["criterion_id"])}</td><td>{metric["paired_samples"]}</td><td>{fmt(metric["ai_MAE_grade"])}</td><td>{fmt(metric["rule_MAE_grade"])}</td></tr>'
                )
            parts.append("</table></div>")
        comparisons = result["comparisons"]
        if exercise == "plank" and result.get("rule_only_previews"):
            comparisons = result["rule_only_previews"]
            parts.append(
                "<p>아래는 학습 자료의 규칙 동작 확인 예시입니다. AI는 학습 자료·점수 다양성 부족으로 보류하며, 이 예시를 검증·시험 성능으로 계산하지 않습니다.</p>"
            )
        if comparisons:
            parts.append(
                '<div class="table-scroll"><table><tr><th>자료</th><th>검토 초안</th>'
                "<th>AI</th><th>규칙</th><th>AI − 규칙</th></tr>"
            )
            for comparison in comparisons:
                fmt = lambda v: "보류" if v is None else f"{v:.1f}"
                parts.append(
                    "<tr><td>"
                    + html.escape(comparison["id"])
                    + "</td>"
                    + "".join(
                        "<td>" + fmt(comparison[key]) + "</td>"
                        for key in (
                            "label_score",
                            "ai_score",
                            "rule_score",
                            "difference",
                        )
                    )
                    + "</tr>"
                )
            parts.append("</table></div>")
            for comparison in (
                comparisons[-2:] if exercise == "plank" else comparisons[:2]
            ):
                row = manifests[comparison["id"]]
                path = overlay(row, comparison)
                relative = path.relative_to(audit.OUT)
                parts.append(
                    f'<h3>{html.escape(row["id"])}</h3>'
                    f'<video controls preload="metadata" '
                    f'src="{relative}"></video><details>'
                    "<summary>규칙 측정값·경계·보류 근거</summary><pre>"
                    + html.escape(
                        json.dumps(
                            dict(
                                rule=comparison["rule_details"],
                                noise_sensitivity=comparison[
                                    "noise_sensitivity"
                                ],
                            ),
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    + "</pre></details>"
                )
        else:
            parts.append(
                "<p>동일 구간의 비교에 필요한 자료가 부족합니다. "
                "점수·오차를 임의로 채우지 않았습니다.</p>"
            )
        parts.append("</section>")
    parts.append("""<section><h2>추가 라벨과 남은 작업</h2>
<p>QEVD 새 플랭크 후보 16개는 다섯 시점 구도만 직접 검토했으며 측면
전신 관측 부족으로 총점을 만들지 않았습니다. 외부 플랭크 3명에서 각각 3초 구간의
24장 시간순 표본을 검토해 실험 라벨 3개를 추가했습니다. 정상 예시 중심이며
오류 사례·참여자·별도 검증·최종 시험은 계속 보충해야 합니다.</p>
<p>새 참여자 Y–T–W 개발 검증 4개를 추가 직접 표본 검토해 총점 초안
3개와 총점 보류·부분 항목 라벨 1개를 기록했습니다. 기존 시험 분할을
새 독립 시험으로 재사용하지 않았습니다.</p>
<p><a href="https://www.dvidshub.net/video/805472/question-day-proper-form-plank">
외부 영상 출처: Quentin Melson / Chief of Naval Personnel</a> ·
<a href="https://www.dvidshub.net/about/copyright">DVIDS 이용 조건</a></p>
<p><a href="https://www.dvidshub.net/video/840093/acft-prep-plank">추가 외부 영상: William China·Bertha Smith / U.S. Army Central</a></p>
<p><a href="https://www.dvidshub.net/video/640237/plank">추가 측면 영상: Matthew Holfinger / U.S. Marine Corps Training and Education Command</a> · 같은 수행자의 합성 패널은 하나의 시범으로 처리했습니다.</p>
<p>숄더 프레스는 개발 검증 10개·학습 후보 6개를 각각 36장 시간순 표본으로
검토했습니다. 앉은 변형·상단 잘림은 총점을 보류했습니다.
QEVD 플랭크 자동 구도 선별 137개는 직접 검토·점수 라벨 수에 포함하지 않습니다.</p>
<p>추가 육군 플랭크는 직접 검토 라벨은 있으나 Heavy 오른쪽 무릎 관측률이 77.4%로 공통 80% 기준에 못 미쳐 학습 입력을 보류했습니다.</p>
<p><a href="observation_failure_details.json">관절별 입력 보류 자동 진단</a> · <a href="cache_parity.json">추출 변경 회귀 검사</a></p>
<p><a href="external_source_screening.json">외부 후보·이용 조건·중복·보류 기록</a> ·
<a href="verification.json">계산·파일 연결 검사</a></p>
<small>The appearance of U.S. Department of War (DoW) visual information
does not imply or constitute DoW endorsement.</small></section></html>""")
    target = audit.OUT / "index.html"
    temporary = target.with_suffix(".tmp.html")
    temporary.write_text("".join(parts), encoding="utf-8")
    temporary.replace(target)
    return target


if __name__ == "__main__":
    print(render())
