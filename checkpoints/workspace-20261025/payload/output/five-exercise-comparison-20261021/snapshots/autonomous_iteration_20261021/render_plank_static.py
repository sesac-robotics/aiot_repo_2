"""정지 플랭크 항목 개발 비교와 관측 실패를 로컬 웹에 표시한다."""

import html
import pathlib

import cv2
import numpy as np

import audit
from resources import check_resources


def run() -> pathlib.Path:
    """56장 관절 표시 이미지와 참여자 분리 비교 페이지를 저장한다.

    Returns:
        pathlib.Path: 생성한 정지 항목 HTML의 절대 경로. 원본 이미지는
        변경하지 않으며 픽셀 좌표를 출력 이미지 크기에 맞춰 축소한다.

    Raises:
        OSError: 입력·출력 파일 접근 실패 또는 이미지 디코딩 실패.
        RuntimeError: 자원 보호 기준 초과.
        ValueError: 원본 해시가 추출 시점과 다를 때.
    """
    folder = audit.OUT / "plank_static_review"
    report = audit.read_json(folder / "comparison.json")
    cache = np.load(folder / "heavy_static.npz", allow_pickle=False)
    indices = {str(value): i for i, value in enumerate(cache["ids"])}
    overlays = folder / "overlays"
    overlays.mkdir(exist_ok=True)
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>플랭크 정지 정렬 항목 개발 비교</title><style>body{font:16px/1.6 system-ui;background:#14212c;color:#edf4fa;max-width:1150px;margin:auto;padding:24px}a{color:#80dcff}section{background:#213544;padding:18px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #567}summary{cursor:pointer}.muted{color:#b6c7d5}</style><h1>플랭크: 정지 몸통 정렬 항목 개발 비교</h1><p><a href="index.html">5종 비교</a> · <a href="aihub_review.html">AI-Hub 자료 검토</a> · <a href="plank_static_review/comparison.json">전체 수치</a> · <a href="plank_static_review/direct_labels.json">동결 직접 검토 초안</a></p>'
    ]
    parts.append(
        f'<section><h2>비전문가 초안에 대한 개발 실험</h2><p>이미지 {report["source_images"]}장, 직접 정렬 초안 {report["directly_drafted_trunk_labels"]}장, 관절 관측 {report["observable_images"]}장, 공통 비교 {report["common_evaluated_images"]}장. 평가 가능률은 전체 이미지 대비 {report["common_evaluation_fraction"]:.1%}, 라벨 보유 이미지 대비 {report["common_fraction_among_labeled"]:.1%}입니다. 참여자 토큰 {len(report["folds"])}개를 하나씩 제외해 예측했습니다. 모든 이미 같은 개발 자료이며 독립 최종 시험은 0개입니다.</p><p>동일 Heavy IMAGE 관절에서 AI는 몸통 길이로 정규화한 발목 위치 2개 성분을 학습했고, 규칙은 골반 내각 168도/155도의 사전 고정 경계를 적용했습니다. 제공자 정상·오류 문자열은 점수로 변환하지 않았습니다. 새 추출·규칙 결과를 보기 전에 라벨을 저장했습니다.</p><p>골반의 밴드·기구 지지, 사선 투영, 실제 FPS·연속 지지 미확인 때문에 기본 플랭크 총점은 전부 보류합니다. 아래는 0~2 등급과 원래 배점 40점의 항목 기여분이며 100점 총점이 아닙니다.</p><table><tr><th>방식</th><th>등급 MAE (0~2)</th><th>40점 항목 MAE</th></tr>'
    )
    for key, name in [
        ("ai_oof_grade", "AI 참여자 제외 예측"),
        ("rule_grade", "고정 Rule Base"),
        ("train_mean_grade", "학습 평균값 기준선"),
    ]:
        metric = report["metrics"][key]
        parts.append(
            f'<tr><td>{name}</td><td>{metric["grade_mae"]:.3f}</td><td>{metric["contribution_40_mae"]:.2f}</td></tr>'
        )
    parts.append(
        "</table><p>AI가 단순 평균값 기준선을 넘지 못했습니다. 공통 플랭크 총점 모델에 채택하지 않았습니다. 실제 인물·외부 구도 일반화와 재촬영·시간 일관성도 검증되지 않았습니다.</p>"
    )
    timing = report["timing_seconds"]
    parts.append(
        f'<p>이 실행의 중앙 처리 시간: 디코딩·축소·Heavy IMAGE {timing["heavy_decode_resize_inference_median"]*1000:.1f}ms, 공통 특징 {timing["feature_median"]*1000:.3f}ms, 규칙 {timing["rule_compute_median"]*1000:.3f}ms, AI 단일 예측 {timing["ai_single_prediction_median"]*1000:.3f}ms. Heavy 초기화 {timing["heavy_initialization"]:.2f}초는 별도입니다. 정지 이미지 실행 시간이며 실시간 영상 FPS가 아닙니다.</p></section>'
    )
    parts.append(
        "<section><h2>참여자별 제외 검증</h2><table><tr><th>제외 참여자 토큰</th><th>학습/검증 이미지</th><th>AI 등급 MAE</th></tr>"
    )
    for fold in report["folds"]:
        parts.append(
            f'<tr><td>{esc(fold["held_person"])}</td><td>{fold["train_images"]}/{fold["test_images"]}</td><td>{fold["grade_mae"]:.3f}</td></tr>'
        )
    parts.append("</table></section>")
    diagnostic_path = folder / "quality_audit.json"
    if diagnostic_path.exists():
        diagnostic = audit.read_json(diagnostic_path)
        parts.append(
            '<section><h2>평가 실패와 누락 편향</h2><p><a href="plank_static_review/quality_audit.json">감사·직접 확인 실패 근거</a></p><table><tr><th>직접 초안 등급</th><th>관측/전체</th><th>관측률</th></tr>'
        )
        for grade, values in diagnostic["coverage_by_draft_grade"].items():
            parts.append(
                f'<tr><td>{esc(grade)}</td><td>{values["observable"]}/{values["images"]}</td><td>{values["coverage"]:.1%}</td></tr>'
            )
        parts.append(
            "</table><p>큰 정렬 오류인 0등급은 절반 이상 관절 관측에 실패했습니다. Z56의 8장은 모두 미관측입니다. 관측 가능한 이미지의 오차만으로 전체 성능을 낙관하지 않습니다.</p><p>원본·관절 표시 7장을 직접 재검토해 기구 가림과 잘못 추정된 다리 관절을 확인했습니다. 높은 visibility만으로 실제 위치의 정확성을 보장할 수 없습니다. 직접 초안은 결과에 맞춰 변경하지 않았습니다.</p>"
        )
        equal = diagnostic["equal_grade_weight_mae"]
        parts.append(
            f'<p>등급별 동일 가중 MAE도 AI {equal["ai_oof_grade"]:.3f}, 규칙 {equal["rule_grade"]:.3f}, 평균값 기준선 {equal["train_mean_grade"]:.3f}로 AI의 우위를 확인하지 못했습니다.</p><p>별도 프로세스로 56장 재추출: 관측 마스크와 유한 좌표가 일치하고 최대 좌표 차이 0픽셀이었습니다. 동일 이미지 재실행 검사이며 시간 안정성·재촬영 검증은 아닙니다.</p></section>'
        )
    for row in report["rows"]:
        check_resources()
        source = pathlib.Path(row["image_path"])
        if audit.digest(source) != row["image_sha256"]:
            raise ValueError("Changed source image")
        frame = cv2.imread(str(source))
        if frame is None:
            raise OSError("Failed image decode")
        scale = 960 / frame.shape[1]
        frame = cv2.resize(frame, (960, round(frame.shape[0] * scale)))
        xy = cache["image_xy"][indices[row["id"]]]
        for a, b in [
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ]:
            if np.isfinite(xy[[a, b]]).all():
                cv2.line(
                    frame,
                    tuple(np.rint(xy[a] * scale).astype(int)),
                    tuple(np.rint(xy[b] * scale).astype(int)),
                    (220, 180, 40),
                    2,
                )
        for point in xy:
            if np.isfinite(point).all():
                cv2.circle(
                    frame,
                    tuple(np.rint(point * scale).astype(int)),
                    3,
                    (40, 230, 250),
                    -1,
                )
        target = overlays / (row["id"] + ".jpg")
        if not cv2.imwrite(str(target), frame):
            raise OSError("Overlay write failed")
        values = [
            (
                "보류"
                if row[key] is None
                else f"{row[key]:.3f} ({20*row[key]:.2f}/40)"
            )
            for key in ["draft_grade", "ai_oof_grade", "rule_grade"]
        ]
        angle = (
            "관절 미관측"
            if row["pose"] is None
            else f'{row["pose"]["side"]} · 골반 내각 {row["pose"]["angle_deg"]:.1f}도'
        )
        difference = (
            "보류"
            if row["ai_oof_grade"] is None
            else f'{20*(row["ai_oof_grade"]-row["rule_grade"]):+.2f}/40'
        )
        parts.append(
            f'<section id="{esc(row["id"])}"><h2>{esc(row["id"])}</h2><p>초안 등급·기여분 {values[0]} · AI {values[1]} · 규칙 {values[2]} · AI−규칙 {difference}<br>{angle} · 총점 보류</p><details><summary>관절 표시와 원본</summary><img loading="lazy" alt="관측 관절 표시 {esc(row["id"])}" src="{target.relative_to(audit.OUT).as_posix()}"><p><a href="{source.relative_to(audit.OUT).as_posix()}">변경 없는 원본 이미지</a></p></details></section>'
        )
    parts.append("</html>")
    target = audit.OUT / "plank_static.html"
    target.write_text("".join(parts), encoding="utf-8")
    index = audit.OUT / "index.html"
    page = index.read_text(encoding="utf-8")
    if 'href="plank_static.html"' not in page:
        page = page.replace(
            "</h1>",
            '</h1><p><a href="plank_static.html">플랭크 정지 정렬 항목 · AI/규칙 비교</a></p>',
            1,
        )
        index.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
