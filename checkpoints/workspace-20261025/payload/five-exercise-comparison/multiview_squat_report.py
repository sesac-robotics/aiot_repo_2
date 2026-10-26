"""스쿼트 별도 개발 실험의 근거·관절 영상·두 점수를 로컬 웹에 연결한다."""

import html
import json
import pathlib

import cv2
import joblib

import audit
import render
from multiview_squat_prepare import OUT
from multiview_squat_trial import load_new
from recovery_finalize import LocalLinks
from recovery_prepare import ROOT
from recovery_preview import profile


def run() -> dict:
    """저장 결과의 지문을 대조하고 오류 예시의 두 시점 영상을 만든다.

    Returns:
        dict: 로컬 링크·전체 영상 디코딩·모델 보존 검사. 브라우저 재생과
        독립 최종 성능 또는 전문가 정답 검증을 뜻하지 않는다.

    Raises:
        ValueError: 입력·모델·코드 지문 불일치 또는 영상·링크 누락.
        RuntimeError: 영상 생성 중 자원 보호 기준 위반.
        OSError: 파일 접근 또는 HTML 저장 실패.
    """
    report = audit.read_json(OUT / "comparison.json")
    checks = {
        "baseline": report["source_comparison_sha256"]
        == audit.digest(ROOT / "comparison.json"),
        "labels": report["labels_sha256"]
        == audit.digest(OUT / "score_annotations.json"),
        "manual_decisions": report["manual_decisions_sha256"]
        == audit.digest(OUT / "manual_decisions.json"),
        "subject_tracking_review": report["subject_tracking_review_sha256"]
        == audit.digest(OUT / "subject_tracking_review.json"),
        "model": report["model_sha256"]
        == audit.digest(OUT / "squats_candidate.joblib"),
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
    if not all(checks.values()):
        raise ValueError("Trial fingerprints do not match saved inputs")
    receipts = audit.read_json(OUT / "extraction.json")["records"]
    rows = load_new(receipts)
    model = joblib.load(OUT / "squats_candidate.joblib")
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    videos = []
    identifier = "subject_002_squat_bad_paired_direct_rep0"
    selected = annotations[identifier]
    prediction = next(p for p in report["predictions"] if p["id"] == identifier)
    receipt = next(r for r in receipts if r["id"] == identifier)
    for view in ("side", "front"):
        folder = OUT / "previews" / identifier / view
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / "side.npz"
        cache = pathlib.Path(receipt[view]["cache"])
        if link.is_symlink() and link.resolve() != cache:
            raise ValueError("Unexpected preview cache link")
        if not link.exists():
            link.symlink_to(cache)
        shown = dict(selected)
        if view == "front":
            shown.update(
                video_path=selected["front_path"],
                segment=selected["front_segment"],
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
        if decoded != expected or not decoded:
            raise ValueError("Preview full decoding failed")
        videos.append(
            dict(
                view=view,
                path=str(video),
                sha256=audit.digest(video),
                decoded_frames=decoded,
            )
        )
    timings = [profile(model, r) for r in rows if r["split"] == "validation"]
    page = [
        '<!doctype html><meta charset="utf-8"><title>보유 MultiView 스쿼트 개발 실험</title>',
        "<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;padding:16px;background:#f5f7fa;color:#213047}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd4df;padding:10px;text-align:left}p{line-height:1.6}video{max-width:46%;max-height:560px}img{max-width:100%}details{background:white;margin:16px 0;padding:12px}</style>",
        "<h1>보유 MultiView 스쿼트 — 별도 개발 후보</h1>",
        '<p><a href="../index.html">기존 5종 결과로 돌아가기</a></p>',
        "<p>8개 원본 묶음·16개 시점 검토판을 직접 표본 검토했습니다. 선택 반복 5개 중 학습 2개, 기존 개발 검증 역할 3개입니다. 의상·장면과 움직임이 맞지 않는 2묶음은 격리했고, 단일 시차가 불충분한 1묶음은 추가 대조 대상으로 남겼습니다.</p>",
        "<p>라벨은 AI가 시간순 표본을 직접 검토한 비전문가 초안입니다. 연속 전체 영상·전문가 정답·독립 최종 시험이 아닙니다. 제공자 good/bad를 수치로 변환하지 않았습니다. 각 원본의 기존 참여자 메타데이터와 개발 분할을 유지했습니다.</p>",
        "<p>두 시점은 선택 반복의 동작 사건을 대조한 관측 정렬입니다. 하드웨어 동기화 인증이 아닙니다. 직접 대조 허용차 2원본 프레임, 추출 10Hz 반올림 허용차 최대 1프레임을 기록하며 관절 누락은 채우지 않습니다.</p>",
        "<p>MediaPipe Heavy는 공통 관절 추출기이며, AI 점수 모델과 각도·정렬 Rule Base는 별도로 계산합니다. 기존 5종 모델과 W 실험은 보존했고 이 후보를 공통 모델로 승격하지 않았습니다.</p>",
        "<p>직접 라벨 5개가 곧 총점 학습 입력 5개라는 뜻은 아닙니다. 측면 3시도는 관절 누락·대상 추적 불안정으로 측면 의존 항목을 차단했습니다. 오류 예시의 원본 추출은 배경 인물 관절을 섞었고, 영역 자르기 재추출도 하체 관절을 충분히 복구하지 못했습니다. 따라서 이 예시는 전체 점수 성능을 입증하지 않습니다.</p>",
        "<p>이번 추가 자료의 실제 관측·라벨 교집합: 학습 항목 "
        + str(report["added_training_eligible_item_labels"])
        + "개, 검증 항목 "
        + str(report["added_validation_eligible_item_labels"])
        + "개, 네 항목 모두 관측 가능한 시도 "
        + str(report["added_four_item_observed_segments"])
        + "개입니다.</p>",
        "<h2>같은 개발 검증 입력의 비교</h2><table><tr><th>검증 묶음</th><th>모델</th><th>공통 총점 수</th><th>AI MAE / 규칙 MAE (점)</th><th>제출 수</th></tr>",
    ]
    for subset, variants in report["comparisons"].items():
        for stage, metrics in variants.items():
            mae = metrics["paired_total_MAE_points"]
            page.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(v)) + "</td>"
                    for v in (
                        subset,
                        stage,
                        metrics["paired_total_records"],
                        str(mae["ai_score"]) + " / " + str(mae["rule_score"]),
                        metrics["submitted"],
                    )
                )
                + "</tr>"
            )
    page.append(
        "</table><p>기존 검증과 새로 검토한 개발 검증을 분리 표시합니다. 후자는 미공개 최종 시험으로 재명명하지 않습니다. 3개 시도는 2개 참여자 메타 그룹이므로 결과의 일반화 근거가 제한적입니다.</p>"
    )
    page.append(
        "<h2>동일 관측 항목 비교</h2><table><tr><th>자료</th><th>후보</th><th>항목</th><th>같이 평가한 수</th><th>AI MAE / Rule MAE (0~2 등급)</th></tr>"
    )
    for subset, variants in report["comparisons"].items():
        for stage, metrics in variants.items():
            for item, values in metrics["items"].items():
                cells = (
                    subset,
                    stage,
                    item,
                    values["paired_records"],
                    str(values["ai_MAE_grade_units"])
                    + " / "
                    + str(values["rule_MAE_grade_units"]),
                )
                page.append(
                    "<tr>"
                    + "".join(
                        "<td>" + html.escape(str(c)) + "</td>" for c in cells
                    )
                    + "</tr>"
                )
    page.append(
        "</table><h2>추가 개발 검증: 항목·총점·차이</h2><table><tr><th>시도</th><th>초안 등급 / 점수</th><th>AI 등급 / 점수</th><th>Rule 등급 / 점수</th><th>AI−Rule</th></tr>"
    )
    for p in report["predictions"]:
        if p["subset"] != "added_multiview":
            continue
        difference = (
            p["ai_score"] - p["rule_score"]
            if p["ai_score"] is not None and p["rule_score"] is not None
            else None
        )
        values = (
            p["id"],
            str(p["levels"]) + " / " + str(p["target_score"]),
            str(p["ai_levels"]) + " / " + str(p["ai_score"]),
            str(p["rule_levels"]) + " / " + str(p["rule_score"]),
            difference,
        )
        page.append(
            "<tr>"
            + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values)
            + "</tr>"
        )
        page.append(
            '<tr><td colspan="5">'
            + html.escape(annotations[p["id"]]["assessment"]["note_ko"])
            + "<br>"
            + html.escape(str(p["observation"]))
            + "</td></tr>"
        )
    page.append(
        "</table><h2>뒤꿈치 들림 오류 예시: 같은 반복의 두 입력 시점</h2>"
    )
    for video in videos:
        relative = pathlib.Path(video["path"]).relative_to(OUT).as_posix()
        page.append(
            '<video controls preload="metadata" src="' + relative + '"></video>'
        )
    page.append(
        "<p>총점은 두 시점 결합 입력의 시도 점수이며 각 프레임의 독립 점수가 아닙니다. 아래 처리 시간은 CPU 할당 25%의 저부하 보호 환경에서 같은 입력 3회로 측정한 점수 계산 시간입니다. 공통 Heavy 관절 추출·특징·영상 디코딩 비용은 제외하며 서비스 실시간 지연이나 일반 장치 속도로 해석하지 않습니다.</p><pre>"
        + html.escape(json.dumps(timings, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    page.append("<h2>초안과 직접 검토 근거</h2>")
    for row in annotations.values():
        evidence = (
            pathlib.Path(row["evidence_images"][0]).relative_to(OUT).as_posix()
        )
        page.append(
            "<details><summary>"
            + html.escape(row["id"])
            + " · "
            + row["split"]
            + "</summary><p>"
            + html.escape(str(row["alignment"]))
            + '</p><img loading="lazy" src="'
            + evidence
            + '"></details>'
        )
    for file, title in (
        ("comparison.json", "비교 결과"),
        ("score_annotations.json", "점수 확장 JSON"),
        ("fine_grained_labels.json", "QEVD 기본 4필드 JSON"),
        ("manual_decisions.json", "직접 검토·격리 결정"),
        ("motion_alignment.json", "자동 광류 시차 후보"),
        ("subject_tracking_review.json", "관절 표시 직접 표본 검토·차단 항목"),
    ):
        page.append('<p><a href="' + file + '">' + title + "</a></p>")
    (OUT / "index.html").write_text("\n".join(page), encoding="utf-8")
    parser = LocalLinks()
    parser.feed((OUT / "index.html").read_text(encoding="utf-8"))
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Missing local report links")
    result = dict(
        comparison_sha256=audit.digest(OUT / "comparison.json"),
        fingerprints=checks,
        all_frames_decoded=True,
        videos=videos,
        score_calculation_profile=timings,
        benchmark_environment=dict(
            cpu_quota_percent=25,
            protected_memory_max_gib=3,
            same_feature_repeats=3,
            end_to_end_latency=False,
        ),
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        browser_rendering_verified=False,
        independent_final_test=False,
    )
    audit.save_json(OUT / "report_verification.json", result)
    return dict(
        videos=len(videos),
        all_frames_decoded=True,
        local_links_checked=len(parser.paths),
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
