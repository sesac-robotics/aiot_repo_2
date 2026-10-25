"""추가 플랭크 오류 초안·관절·W 집계 실패를 기존 로컬 웹에 연결한다."""

import html
import pathlib

import cv2
import numpy as np

import audit
from resources import check_resources


def run() -> pathlib.Path:
    """5개 항목 검토와 새 W 집계안의 개발 결과 HTML을 저장한다.

    Returns:
        pathlib.Path: HTML 절대 경로. 총점과 AI는 보류이며 정상·오류가
        같은 원본에 있더라도 train 그룹을 유지한다. 썸네일은 중앙 표본이다.

    Raises:
        OSError: 파일 접근 또는 관절 이미지 저장 실패.
        ValueError: 영상 디코딩 실패.
        RuntimeError: 자원 보호 중단.
    """
    direct = audit.read_json(audit.OUT / "plank_error_direct_labels.json")
    results = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "plank_error_results.json")[
            "results"
        ]
    }
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>플랭크 오류·W 시간 집계 개발 검토</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px;background:#15232e;color:#eef5fc}a{color:#8fddff}section{background:#243b4c;margin:18px 0;padding:18px;border-radius:12px}img,video{width:100%;max-width:800px;height:auto}td,th{padding:10px;border-bottom:1px solid #678;text-align:left}</style><h1>플랭크 오류 구간·W 집계 개발 검토</h1><p><a href="index.html">5종 비교</a> · <a href="youtube_review.html">이전 YouTube 검토</a> · <a href="plank_error_direct_labels.json">동결 직접 초안</a> · <a href="plank_error_results.json">Heavy·항목 진단</a></p><section><h2>확보와 학습 가능 수를 구분</h2><p>재사용 표기를 확인한 원본 2개에서 5개 구간을 직접 검토했습니다. 서로 다른 수행자의 골반 들림 초안 2개를 확보했지만 짧은 유지·팔꿈치 잘림·위쪽 사선으로 전체 3초 총점 학습 자료에는 추가하지 않았습니다. 기존 플랭크 총점 학습 가능 수는 7개·6명 그대로입니다.</p><p>초안은 AI의 비전문가 직접 시각 검토이며 Heavy·규칙 결과를 보기 전에 저장했습니다. 오류 자막만 있고 자세 등급이 불확실한 구간은 null입니다. 규칙 출력으로 정답을 채우거나 보이는 항목만 100점으로 환산하지 않습니다. 정상·오류·전환 모두 동일 원본·참여자 train 그룹입니다.</p></section>'
    ]
    esc = html.escape
    for row in direct["reviews"]:
        check_resources()
        result = results[row["id"]]
        with np.load(result["extraction"]["cache"]) as cache:
            index = len(cache["frame_numbers"]) // 2
            frame = int(cache["frame_numbers"][index])
            xy = cache["image_xy"][index].copy()
            xy[~cache["observed"][index].astype(bool)] = np.nan
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Source overlay frame unavailable")
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
                    pixels,
                    tuple(np.rint(xy[a]).astype(int)),
                    tuple(np.rint(xy[b]).astype(int)),
                    (20, 220, 250),
                    2,
                )
        target = audit.OUT / "review" / (row["id"] + "_joints.jpg")
        if not cv2.imwrite(str(target), pixels):
            raise OSError("Joint preview save failed")
        segment = row["segment"]
        start = segment["start_frame"] / segment["fps"]
        end = (segment["end_frame"] + 1) / segment["fps"]
        source = (
            pathlib.Path(row["video_path"]).relative_to(audit.OUT).as_posix()
        )
        parts.append(
            f'<section><h2>{esc(row["id"])}</h2><p>{start:.3f}~{end:.3f}초 · 초안 총점 보류 · AI 총점 보류 · 규칙 총점 보류</p><p>{esc(row["note_ko"])}</p><table><tr><th>항목</th><th>직접 초안 등급</th><th>규칙 진단 등급</th></tr>'
        )
        for item in result["items"]:
            parts.append(
                f'<tr><td>{esc(item["criterion_id"])}</td><td>{item["draft_grade"]}</td><td>{item["rule_grade"]}</td></tr>'
            )
        parts.append(
            f'</table><img loading="lazy" src="{target.relative_to(audit.OUT).as_posix()}" alt="중앙 표본에서 관측된 Heavy 관절"><details><summary>선택 원본 구간</summary><video preload="none" controls src="{source}#t={start:.3f},{end:.3f}"></video></details><p>제작자 {esc(row["creator"])} · <a href="{esc(row["source_url"])}">원본 YouTube</a> · <a href="https://www.youtube.com/t/creative_commons">CC Attribution 재사용 표기</a></p></section>'
        )
    parts.append(
        "<section><h2>새로 드러난 불일치</h2><p>Pannell 정상 정렬 후보의 몸통 직접 초안은 2등급인데 관절 규칙은 0등급입니다. 고정 규칙의 어깨–골반–발목 각도 하위 10%가 약 152.2도로 측정됐습니다. 초안·관절 위치·투영 및 수치 경계를 함께 재검토해야 하며, 규칙에 맞춰 초안을 바꾸지 않았습니다.</p><p>Colossus 골반 들림은 직접 보이지만 Heavy 관절의 무릎·발목 등이 부족하여 해당 규칙 항목을 산출하지 못했습니다. 눈으로 보이는 오류와 현재 추출기로 채점 가능한 오류를 구분합니다.</p></section>"
    )
    trial = audit.read_json(audit.OUT / "w_temporal_trial_results.json")
    parts.append(
        "<section><h2>W 시간 집계안: 채택하지 않음</h2><p>동일 프레임에서 양팔 모양을 먼저 채점한 후 시간 비율로 집계하는 안을 시험했습니다. 자동 구간을 고정하고 앞뒤 0.1·0.2초 변화도 대조했습니다. 직접 등급과 같은 시도로 확인된 관측 가능 사례만 오차에 포함합니다. 모두 기존 개발 자료입니다.</p><table><tr><th>방식</th><th>비교 수</th><th>등급 MAE</th><th>경계 변화 시 등급 변경</th></tr>"
    )
    for name, r in trial["summary"].items():
        parts.append(
            f'<tr><td>{name}</td><td>{r["paired"]}</td><td>{r["grade_MAE"]:.3f}</td><td>{r["boundary_changed"]}/{r["boundary_observable"]}</td></tr>'
        )
    parts.append(
        '</table><p>경계 민감도 감소만으로 오차 증가를 상쇄했다고 판단하지 않았습니다. 공통 규칙·모델·라벨은 변경하지 않았습니다. 실제 재촬영 일관성 또는 독립 최종 정확도 결과가 아닙니다.</p><a href="w_temporal_trial_results.json">원본 사례별 결과</a></section></html>'
    )
    path = audit.OUT / "plank_error_review.html"
    path.write_text("".join(parts), encoding="utf-8")
    for name in ["index.html", "youtube_review.html"]:
        target = audit.OUT / name
        page = target.read_text(encoding="utf-8")
        if 'href="plank_error_review.html"' not in page:
            page = page.replace(
                "</h1>",
                '</h1><p><a href="plank_error_review.html">추가 플랭크 오류·W 시간 집계 검토</a></p>',
                1,
            )
            target.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(run())
