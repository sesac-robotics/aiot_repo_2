"""보유 자료 보강의 수량·관측·동일 개발 비교를 로컬 HTML로 공개한다."""

import collections
import html
import json

import audit
from allfive_round1_report import NAMES, paired_summary, row_html
from held_round2 import OUT


def build() -> dict:
    """저장된 결과만 읽어 항목·총점·실패와 원본 근거를 연결한다.

    Returns:
        dict: 보고서 경로와 동일 검증 교집합 결과. 독립 시험이나 전문가
        정확도로 해석하지 않으며 이전 모델과 결과는 수정하지 않는다.

    Raises:
        ValueError: 재학습이 끝나지 않았거나 예측 ID가 중복된 경우.
        OSError: 비교 결과·근거 파일 접근 또는 보고서 저장 실패.
    """
    current = audit.read_json(OUT / "comparison.json")
    if not current["completed"]:
        raise ValueError("Training and comparison must finish first")
    baseline = audit.read_json(OUT.parent / "allfive_round1/comparison.json")
    prior = {r["exercise"]: r for r in baseline["exercises"]}
    inputs = audit.read_json(OUT / "candidate_input_manifest.json")
    qualification = current["qualification"]
    sections, paired, overview_rows = [], {}, []
    for result in current["exercises"]:
        exercise = result["exercise"]
        old = {r["id"]: r for r in prior[exercise]["fixed_predictions"]}
        connected = [
            dict(r, baseline=old[r["id"]])
            for r in result["fixed_predictions"]
            if r["id"] in old
        ]
        paired[exercise] = paired_summary(connected)
        if exercise == "ytw":
            previous_projection = audit.read_json(
                OUT.parent
                / "allfive_round1/ytw_projected_rule_trial/comparison.json"
            )
            by_id = {
                r["id"]: r for r in previous_projection["fixed_predictions"]
            }
            paired[exercise]["previous_projected_trial"] = paired_summary(
                [
                    dict(r, baseline=by_id[r["id"]])
                    for r in result["fixed_predictions"]
                    if r["id"] in by_id
                ]
            )
        fixed = result["fixed"]
        overview_rows.append(
            row_html(
                [
                    NAMES[exercise],
                    result["training_segments"],
                    qualification["full_reference_segments"][exercise],
                    f'{fixed["ai_total_available"]}/{fixed["submitted"]}',
                    f'{fixed["rule_total_available"]}/{fixed["submitted"]}',
                    fixed["paired_total_records"],
                    (
                        round(fixed["paired_total_MAE_points"]["ai_score"], 3)
                        if fixed["paired_total_records"]
                        else None
                    ),
                    (
                        round(fixed["paired_total_MAE_points"]["rule_score"], 3)
                        if fixed["paired_total_records"]
                        else None
                    ),
                ]
            )
        )
        train = [
            r
            for r in inputs
            if r["exercise"] == exercise
            and r["split"] == "train"
            and any(
                n["eligible"] and g is not None
                for n, g in zip(r["items"], r["levels"])
            )
        ]
        sources = {s for r in train for s in r["sources"]}
        distribution = []
        for index in range(4):
            grades = collections.Counter(
                r["levels"][index]
                for r in train
                if r["items"][index]["eligible"]
                and r["levels"][index] is not None
            )
            distribution.append(dict(grades))
        overview = [
            ["실제 ≥1항목 학습 구간", result["training_segments"]],
            [
                "4항목 관측·직접 총점 가능 학습 구간",
                qualification["full_reference_segments"][exercise],
            ],
            [
                "참여자 메타 그룹 / 원본 SHA 수",
                f'{qualification["metadata_groups"][exercise]} / {len(sources)}',
            ],
            [
                "학습 항목별 등급 분포 (0 오류 / 1 부분 / 2 정상)",
                json.dumps(distribution, ensure_ascii=False),
            ],
            ["고정 개발 검증", json.dumps(result["fixed"], ensure_ascii=False)],
            [
                "연결 그룹 3분할 교차검증",
                json.dumps(result["group_cv"], ensure_ascii=False),
            ],
            [
                "전후 공통 교집합",
                json.dumps(paired[exercise], ensure_ascii=False),
            ],
        ]
        if exercise == "squats":
            overview.append(
                [
                    "더 이전 배치 AI 총점 MAE /100",
                    "13.735 (현재 후보가 이 수치까지 개선된 것은 아님)",
                ]
            )
        predictions = "".join(
            row_html(
                [
                    r["id"],
                    r["target_score"],
                    r["ai_score"],
                    r["rule_score"],
                    r["levels"],
                    r["ai_levels"],
                    r["rule_levels"],
                ]
            )
            for r in result["fixed_predictions"]
        )
        sections.append(
            f"<h2>{NAMES[exercise]}</h2><table>"
            + "".join(row_html(v) for v in overview)
            + "</table><details><summary>항목별 학습 모델·등급·오류 그룹</summary><pre>"
            + html.escape(
                json.dumps(result["heads"], ensure_ascii=False, indent=2)
            )
            + "</pre></details><table><tr><th>검증 ID</th><th>초안 /100</th>"
            "<th>AI /100</th><th>Rule /100</th><th>초안 /2</th>"
            "<th>AI /2</th><th>Rule /2</th></tr>" + predictions + "</table>"
        )
    receipts = audit.read_json(OUT / "extraction.json")["records"]
    evidence = "".join(
        '<li><a href="joints/'
        + html.escape(r["joint_evidence"].split("/")[-1])
        + '">'
        + html.escape(r["id"])
        + "</a></li>"
        for r in receipts
    )
    body = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>보유 자료 2차 보강 비교</title><style>
body{font:16px sans-serif;max-width:1250px;margin:30px auto;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #aaa;padding:8px;overflow-wrap:anywhere}
pre{white-space:pre-wrap}p{max-width:1050px}a{color:#175bb5}
</style><h1>보유 자료 보강 · AI / Rule 개발 비교</h1>
<p>외부 탐색·다운로드 없이 보유 QEVD 미검토 자료를 먼저 검토했다.
추가 라벨은 AI가 시간순 영상 표본을 직접 검토한 비전문가 실험 초안이다.
정상·오류 제공자 문자열을 점수로 변환하지 않았고 Rule 출력을 정답으로 쓰지 않았다.</p>
<p>70개는 최소 한 항목이 실제 학습된 구간의 수다. 70명·70개 독립 원본·
70개 총점 학습 자료를 뜻하지 않는다. 플랭크 추가 유지 구간은 기존 4개 원본에 집중된다.
같은 사람일 가능성이 있는 해군 두 원본도 하나의 연결 그룹으로 묶었다.
고정 검증은 반복 사용한 개발 자료이며 새로운 독립 최종 시험이 아니다.</p>
<p>MediaPipe Heavy는 관절 추출기이며 별도 회귀 모델이 AI 점수를 예측한다.
미관측 항목은 비워 두고 총점을 환산하지 않는다.
YTW 규칙은 직접 확인한 구도·단계 관측 게이트를 사용하는 투영 기반 실험이며,
새 영상에서의 완전 자동 구도 판정과 공통 규칙 승격을 검증한 상태는 아니다.</p>
<p>보강 후보는 자동 승격하지 않았다. 플랭크 AI의 기존 공통 총점 오차는
6.636에서 10.256으로 증가했고, YTW AI도 이전 투영 실험보다 악화됐다.
구간 수 증가가 성능 향상을 보장하지 않으며 단일 등급만 학습한 항목은
오류 식별 성능을 입증하지 못한다.</p>
<p>플랭크 지지는 무릎 각도만 보던 특징에 팔꿈치–발끝 선 대비 골반·무릎 높이를 추가했다.
스쿼트 발 특징은 선 자세와 하강 자세를 구분하고 고정 발 길이를 분모로 사용한다.
이 구조 변경은 기존 개발 실패를 참고했으며 검증 자료를 임계값 학습에 넣지 않았다.</p>
<p><a href="score_annotations.json">점수 확장 JSON</a> ·
<a href="fine_grained_labels.json">QEVD 기본 4필드 JSON</a> ·
<a href="annotation_export_audit.json">분할·구간 감사</a> ·
<a href="subject_tracking_review.json">직접 관절 표본 검토</a> ·
<a href="comparison.json">전체 수치</a> ·
<a href="candidate_input_manifest.json">실제 학습 입력</a> ·
<a href="../squat_partial_top_trial/index.html">후속 스쿼트 관측 개선 비교</a> ·
<a href="plank_support_failure_diagnostic.json">플랭크 AI 오판 특징 분석</a> ·
<a href="../held_squat_errors/hypothesis_review.json">학습 전 철회한 발 오류 가설</a></p>"""
    previews = ""
    if (OUT / "previews.json").exists():
        for item in audit.read_json(OUT / "previews.json")["records"]:
            prediction = item["prediction"]
            video = str(item["video"]).split("/held_round2/")[-1]
            previews += (
                "<h3>" + NAMES[item["exercise"]] + "</h3>"
                "<p>연결 그룹 교차검증의 폴드 외 예측. 총점 누락은 미환산.</p>"
                '<video controls preload="metadata" width="640" src="'
                + html.escape(video)
                + '"></video><pre>'
                + html.escape(
                    json.dumps(
                        dict(
                            id=prediction["id"],
                            reference=prediction["levels"],
                            ai=prediction["ai_levels"],
                            rule=prediction["rule_levels"],
                            cached_joint_inference_ms=item[
                                "cached_joint_model_and_rule_median_ms"
                            ],
                            deterministic=item[
                                "identical_input_repeat_deterministic"
                            ],
                            separate_speed=item["separate_speed"],
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
        previews = (
            "<h2>새 보강 구간의 관절 영상과 항목 점수</h2>"
            "<p>표시 속도는 특징 캐시 이후 AI·규칙 합산 추론 시간이다. "
            "Heavy·특징 계산·파일 읽기는 제외한다. 동일 입력 3회 반복의 "
            "일치는 촬영 변화에 대한 일관성 검증을 뜻하지 않는다.</p>"
            + previews
        )
    summary = (
        "<table><tr><th>운동</th><th>≥1항목 학습</th>"
        "<th>총점 학습</th><th>AI 총점 가용/검증</th><th>Rule 총점 가용/검증</th>"
        "<th>공통 총점 비교 수</th><th>AI MAE /100</th><th>Rule MAE /100</th></tr>"
        + "".join(overview_rows)
        + "</table>"
    )
    (OUT / "index.html").write_text(
        body
        + summary
        + "".join(sections)
        + previews
        + "<details><summary>95구간의 자동 관절 표시와 직접 검토 근거</summary><ul>"
        + evidence
        + "</ul></details></html>",
        encoding="utf-8",
    )
    report = dict(
        path=str(OUT / "index.html"),
        paired=paired,
        browser_verification=False,
        independent_final_test=False,
    )
    audit.save_json(OUT / "report_audit.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
