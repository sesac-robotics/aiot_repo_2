"""두 번째 보유 스쿼트 배치의 직접 근거·개발 성능·관절 영상을 연결한다."""

import html
import json
import pathlib

import cv2
import joblib
import numpy as np

import audit
import render
from multiview_squat_batch2 import OUT
from multiview_squat_batch2_train import load_batch, mask_reviewed_frames
from recovery_finalize import LocalLinks
from recovery_preview import profile


def table_row(values: tuple) -> str:
    """값을 HTML 이스케이프한 한 행 문자열로 반환한다.

    Args:
        values: 표시할 병렬 열 값. None은 판정 불가로 표시한다.

    Returns:
        str: 표 행. 원본 값이나 점수 단위는 변환하지 않는다.
    """
    return (
        "<tr>"
        + "".join(
            "<td>" + html.escape("판정 불가" if v is None else str(v)) + "</td>"
            for v in values
        )
        + "</tr>"
    )


def run() -> dict:
    """지문·분할·구간을 검사하고 정상·오류 관절 영상과 비교 웹을 만든다.

    Returns:
        dict: 저장 지문, 영상 전체 디코딩, 로컬 링크, 같은 입력 반복 시간.
        브라우저 재생·전문가 검증·독립 최종 시험 인증은 아니다.

    Raises:
        ValueError: 지문·자료 분할·중복·영상 디코딩·파일 링크 불일치.
        OSError: 입력·영상·결과 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    report = audit.read_json(OUT / "comparison.json")
    checks = {
        name: sha == audit.digest(OUT / name)
        for name, sha in report["input_sha256"].items()
    }
    checks.update(
        {
            name: sha == audit.digest(pathlib.Path(__file__).with_name(name))
            for name, sha in report["code_sha256"].items()
        }
    )
    checks.update(
        {
            path: sha == audit.digest(pathlib.Path(path))
            for path, sha in report["preserved"].items()
        }
    )
    checks["model"] = report["model_sha256"] == audit.digest(
        OUT / "squats_candidate.joblib"
    )
    checks["fit_input_manifest"] = report[
        "fit_input_manifest_sha256"
    ] == audit.digest(OUT / "fit_input_manifest.json")
    if not all(checks.values()):
        raise ValueError("Saved result fingerprints changed")
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    rows = load_batch(OUT)
    receipts = {
        r["id"]: r for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    tracking = {
        (r["id"], r["view"]): r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    intervals = {}
    roles = {}
    for row in annotations.values():
        roles.setdefault(row["person"], set()).add(row["split"])
        for sha, segment in (
            (row["video_sha256"], row["segment"]),
            (row["front_sha256"], row["front_segment"]),
        ):
            first, last = segment["start_frame"], segment["end_frame"]
            if any(
                not (last < a or first > b) for a, b in intervals.get(sha, [])
            ):
                raise ValueError("Repeated source intervals overlap")
            intervals.setdefault(sha, []).append((first, last))
    if any(len(v) > 1 for v in roles.values()):
        raise ValueError("Metadata participant crosses roles")
    basic = audit.read_json(OUT / "fine_grained_labels.json")
    if len(basic) != 10 or any(
        set(r) != {"video_path", "labels", "labels_descriptive", "worker_id"}
        for r in basic
    ):
        raise ValueError("QEVD basic JSON contract changed")
    model = joblib.load(OUT / "squats_candidate.joblib")
    predictions = [p for p in report["predictions"] if p["subset"] == "batch2"]
    paired = [
        p
        for p in predictions
        if p["target_score"] is not None
        and p["ai_score"] is not None
        and p["rule_score"] is not None
    ]
    # 정상·오류를 각각 먼저 고른 뒤 가장 큰 오차 사례를 추가한다.
    selected = []
    for normal in (True, False):
        candidates = [p for p in paired if (p["target_score"] == 100) == normal]
        if candidates:
            selected.append(candidates[0])
    if paired:
        worst = max(
            paired,
            key=lambda p: max(
                abs(p["ai_score"] - p["target_score"]),
                abs(p["rule_score"] - p["target_score"]),
            ),
        )
        if worst["id"] not in {p["id"] for p in selected}:
            selected.append(worst)
    videos = []
    for prediction in selected:
        identifier = prediction["id"]
        ann = annotations[identifier]
        for view in ("side", "front"):
            folder = OUT / "previews" / identifier / view
            folder.mkdir(parents=True, exist_ok=True)
            # 표시에도 채점과 동일한 직접 제외 프레임을 적용한다. 원본 캐시는 보존한다.
            with np.load(receipts[identifier][view]["cache"]) as data:
                cache = {k: data[k].copy() for k in data.files}
            cache["image_xy"] = mask_reviewed_frames(
                cache["image_xy"],
                cache["frame_numbers"],
                tracking[identifier, view].get(
                    "unreliable_frame_intervals", []
                ),
            )
            np.savez_compressed(folder / "side.npz", **cache)
            shown = dict(ann)
            if view == "front":
                shown.update(
                    video_path=ann["front_path"], segment=ann["front_segment"]
                )
            video = render.overlay(
                shown,
                dict(prediction, cache_folder=str(folder), display_unit="/100"),
            )
            cap = cv2.VideoCapture(str(video))
            decoded = 0
            try:
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if decoded == expected // 2:
                        cv2.imwrite(str(folder / "preview.jpg"), frame)
                    decoded += 1
            finally:
                cap.release()
            if not decoded or decoded != expected:
                raise ValueError("Preview full decoding failed")
            videos.append(
                dict(
                    id=identifier,
                    view=view,
                    path=str(video),
                    sha256=audit.digest(video),
                    decoded_frames=decoded,
                )
            )
    timings = [profile(model, r) for r in rows if r["split"] == "validation"]
    page = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>보유 스쿼트 12묶음 검토·재학습 비교</title>',
        "<style>body{font:16px sans-serif;max-width:1260px;margin:28px auto;padding:16px;color:#213047;background:#f5f7fa}p{line-height:1.65}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #cbd4df;text-align:left;overflow-wrap:anywhere}video{width:46%;max-height:570px}img{max-width:100%}details{padding:12px;background:white;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
        '<h1>보유 MultiView 스쿼트: 12묶음 후속 검토</h1><p><a href="../../index.html">5종 기준 결과</a> · <a href="../index.html">첫 MultiView 실험</a></p>',
        "<p>12개 원본 묶음의 24개 시점 검토판과 선택 구간 확대 표본을 직접 대조했습니다. 10묶음에서 20개 반복 구간을 라벨 초안으로 기록했습니다(학습 12, 기존 개발 검증 8). 두 시점의 준비 동작·반복 대응을 확정하지 못한 017의 2묶음은 결합 총점에서 보류했습니다. 전체 연속 영상을 모두 검토했다는 뜻은 아닙니다.</p>",
        "<p>라벨은 AI의 시간순 표본 직접 검토 초안이며 전문가 정답이 아닙니다. 제공자 good/bad 문자열을 점수로 바꾸지 않았고, 같은 원본의 반복도 자세가 다르면 등급을 다르게 기록했습니다. 원본당 QEVD 기본 4필드 JSON을 하나씩 유지하고 구간별 점수·배점·근거는 확장 JSON에서 연결합니다.</p>",
        "<p>측면의 하강·복귀·발 지지와 정면의 몸통 균형을 결합합니다. 시차는 선택 반복의 사건 대조 결과이며 하드웨어 동기화 인증이 아닙니다. 관절 신뢰도가 높아도 발을 종아리에 잡거나 흐림 처리 안의 몸통을 추정한 입력은 항목·프레임별로 제외했습니다. 보이지 않는 점수를 채우거나 부분 배점을 100점으로 환산하지 않았습니다.</p>",
        "<p>학습과 개발 검증은 원본·기존 참여자 메타 그룹으로 분리했고 연결 그룹 교차검증을 적용했습니다. 참여자 메타 ID는 실제 사람의 신원을 인증하지 않습니다. 이번 개발 검증 8구간은 012·015의 2개 메타 그룹이며 독립 최종 시험이 아닙니다. 같은 반복의 두 시점·다른 반복은 새로운 참여자가 아닙니다.</p>",
        "<p>MediaPipe Heavy는 공통 관절 추출기입니다. 별도 AI 모델과 각도·정렬 Rule Base를 같은 관측 항목과 점수 초안으로 비교했습니다. 뒤꿈치 규칙 경계는 학습 역할만으로 보정했으며 검증 점수를 선택에 사용하지 않았습니다. 기존 5종 모델·플랭크·YTW·첫 실험 결과는 보존했고 이번 후보는 공통 모델로 자동 반영하지 않았습니다.</p>",
        "<h2>확보와 실제 사용 수</h2><pre>"
        + html.escape(
            json.dumps(
                dict(
                    added=report["added"],
                    training_effective_segments=report[
                        "training_effective_segments"
                    ],
                    training_four_item_observed_segments=report[
                        "training_four_item_observed_segments"
                    ],
                    training_metadata_groups=report["training_metadata_groups"],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>",
        "<h2>같은 개발 검증 입력 비교</h2><table><tr><th>자료</th><th>모델</th><th>전체 / 공통 총점 수</th><th>총점 AI MAE / Rule MAE (점)</th></tr>",
    ]
    for subset, variants in report["comparisons"].items():
        for stage, m in variants.items():
            errors = m["paired_total_MAE_points"]
            page.append(
                table_row(
                    (
                        subset,
                        stage,
                        f"{m['submitted']} / {m['paired_total_records']}",
                        f"{errors['ai_score']} / {errors['rule_score']}",
                    )
                )
            )
    page.append(
        "</table><p>총점이 없는 기존 개발 자료의 오차는 항목별로만 비교합니다. 새 검증의 총점 표본 수와 판정 가능 비율을 함께 봐야 하며, 작은 평균 오차만으로 완료나 일반화를 주장하지 않습니다.</p><h2>항목별 비교 (등급 0~2)</h2><table><tr><th>자료·후보</th><th>항목</th><th>공통 수</th><th>AI MAE / Rule MAE</th></tr>"
    )
    for subset, variants in report["comparisons"].items():
        for stage, m in variants.items():
            for item, v in m["items"].items():
                page.append(
                    table_row(
                        (
                            subset + " / " + stage,
                            item,
                            v["paired_records"],
                            f"{v['ai_MAE_grade_units']} / {v['rule_MAE_grade_units']}",
                        )
                    )
                )
    page.append(
        "</table><h2>구간별 초안·두 점수·차이·근거</h2><table><tr><th>구간</th><th>초안 등급 / 총점</th><th>AI 등급 / 총점</th><th>Rule 등급 / 총점</th><th>AI−Rule</th></tr>"
    )
    for p in predictions:
        difference = (
            p["ai_score"] - p["rule_score"]
            if p["ai_score"] is not None and p["rule_score"] is not None
            else None
        )
        page.append(
            table_row(
                (
                    p["id"],
                    str(p["levels"]) + " / " + str(p["target_score"]),
                    str(p["ai_levels"]) + " / " + str(p["ai_score"]),
                    str(p["rule_levels"]) + " / " + str(p["rule_score"]),
                    difference,
                )
            )
        )
        page.append(
            '<tr><td colspan="5">'
            + html.escape(annotations[p["id"]]["assessment"]["note_ko"])
            + "<br>"
            + html.escape(str(p["observation"]))
            + "</td></tr>"
        )
    page.append(
        "</table><h2>정상·오류·최대 오차 예시의 관절 영상</h2><p>표시 점수는 측면·정면 결합 반복의 점수이며 프레임별 점수가 아닙니다. 두 영상에 같은 결합 점수를 표시합니다.</p>"
    )
    for p in selected:
        page.append("<h3>" + html.escape(p["id"]) + "</h3>")
        for v in [v for v in videos if v["id"] == p["id"]]:
            page.append(
                '<video controls preload="metadata" src="'
                + pathlib.Path(v["path"]).relative_to(OUT).as_posix()
                + '"></video>'
            )
    page.append(
        "<h2>학습 내부 교차검증·규칙 보정 대조</h2><p>교차검증은 연결된 참여자·원본 그룹을 통째로 제외하고 매번 재학습합니다. 아래 수치는 학습 자료 내부 평가이며 독립 최종 시험이 아닙니다.</p><pre>"
        + html.escape(
            json.dumps(
                dict(
                    group_cross_validation=report["group_cross_validation"],
                    heads=report["heads"],
                    rule_calibration_ablation=report[
                        "rule_calibration_ablation"
                    ],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>"
    )
    page.append(
        "<h2>일관성·계산 시간</h2><p>동일 관측 특징 3회 재계산 결과 일치와 중앙값 밀리초입니다. CPU 25%·메모리 3GiB 제한 환경이며 공통 Heavy 관절 추출·특징 계산·디코딩 시간을 제외했습니다. 반복 간 강건성이나 서비스 전체 지연을 측정한 것은 아닙니다.</p><pre>"
        + html.escape(json.dumps(timings, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    failure_path = OUT / "failure_analysis.json"
    if failure_path.exists():
        failures = audit.read_json(failure_path)
        if failures["comparison_sha256"] != audit.digest(
            OUT / "comparison.json"
        ):
            raise ValueError("Failure analysis refers to an older candidate")
        page.append(
            "<h2>실패 원인·남은 부족 항목</h2><p>저장 특징·예측의 자동 집계이며 새 직접 라벨이 아닙니다. 후보 승격을 보류합니다.</p><pre>"
            + html.escape(json.dumps(failures, ensure_ascii=False, indent=2))
            + '</pre><p><a href="failure_analysis.json">실패 분석 JSON</a></p>'
        )
    page.append("<h2>20구간의 직접 검토 근거</h2>")
    for identifier, ann in annotations.items():
        ready = next(r for r in rows if r["id"] == identifier)
        page.append(
            "<details><summary>"
            + html.escape(identifier + " / " + ann["split"])
            + "</summary><p>"
            + html.escape(ann["assessment"]["note_ko"])
            + "</p><pre>"
            + html.escape(
                json.dumps(
                    dict(
                        levels=ready["levels"],
                        observed=[n["eligible"] for n in ready["notes"]],
                        total_reference_allowed=ready[
                            "reference_total_allowed"
                        ],
                        tracking=[
                            tracking[identifier, v] for v in ("side", "front")
                        ],
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre>"
        )
        for path in (
            ann["evidence_images"][0],
            str(OUT / "review" / f"{identifier}_joints_pair.jpg"),
        ):
            page.append(
                '<img loading="lazy" src="'
                + pathlib.Path(path).relative_to(OUT).as_posix()
                + '">'
            )
        page.append("</details>")
    for name in (
        "comparison.json",
        "score_annotations.json",
        "fine_grained_labels.json",
        "qevd_assessment_index.json",
        "manual_decisions.json",
        "subject_tracking_review.json",
        "fit_input_manifest.json",
        "preparation.json",
    ):
        page.append('<p><a href="' + name + '">' + name + "</a></p>")
    (OUT / "index.html").write_text("\n".join(page), encoding="utf-8")
    parser = LocalLinks()
    parser.feed((OUT / "index.html").read_text())
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Missing local links: " + str(missing))
    result = dict(
        comparison_sha256=audit.digest(OUT / "comparison.json"),
        fingerprints=checks,
        videos=videos,
        all_frames_decoded=True,
        score_calculation_profile=timings,
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        browser_rendering_verified=False,
        independent_final_test=False,
        interval_overlap=False,
        metadata_role_overlap=False,
        qevd_original_records=len(basic),
        score_interval_records=len(rows),
        code_sha256=audit.digest(pathlib.Path(__file__)),
    )
    audit.save_json(OUT / "report_verification.json", result)
    return dict(
        videos=len(videos),
        local_links_checked=len(parser.paths),
        all_frames_decoded=True,
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
