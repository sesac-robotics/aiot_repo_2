"""YouTube 출처·구간 분리·직접 초안·Heavy 관측을 기존 웹에 연결한다."""

import html
import pathlib

import cv2
import numpy as np

import audit
from resources import check_resources


def run() -> pathlib.Path:
    """9개 개발 구간의 출처·관절 표시·점수 보류 근거를 HTML로 저장한다.

    Returns:
        pathlib.Path: 출력 HTML 절대 경로. 원본 영상은 표시 초 범위를
        지정해 연결하고 무손실 추출본을 학습 정답으로 취급하지 않는다.

    Raises:
        OSError: 입력·출력 접근 또는 이미지 저장 실패.
        ValueError: 구간·관절 디코딩 오류.
        RuntimeError: 자원 보호 중단.
    """
    exports = audit.read_json(audit.OUT / "youtube_segment_exports.json")
    joints = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "youtube_joint_results.json")[
            "results"
        ]
    }
    drafts = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "youtube_direct_labels.json")[
            "reviews"
        ]
    }
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YouTube 운동 구간·관절 검토</title><style>body{font:16px/1.65 system-ui;background:#14212c;color:#edf4fa;max-width:1120px;margin:auto;padding:24px}a{color:#80dcff}section{background:#213544;margin:20px 0;padding:18px;border-radius:12px}img,video{width:100%;max-width:800px;height:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #567}.notice{border-left:4px solid #ffd17d;padding:12px}</style><h1>YouTube 운동 구간·관절 개발 검토</h1><p><a href="index.html">5종 점수 비교</a> · <a href="plank_static.html">정지 정렬 항목 실험</a> · <a href="youtube_segment_exports.json">프레임 연결·픽셀 검증</a> · <a href="youtube_direct_labels.json">직접 검토 초안</a> · <a href="youtube_joint_results.json">관측·규칙 미리보기</a></p><section><h2>이번에 실제 확보한 범위</h2>'
    ]
    parts.append(
        f'<p>CC Attribution 재사용 표기를 확인한 원본 3개에서 구간 {len(exports["records"])}개를 분리했습니다. 파생 영상은 {exports["bytes"]/2**20:.1f}MiB입니다. 각 구간의 처음·중간·마지막 3장씩 원본 크롭과 픽셀이 일치했습니다. 전체 프레임 픽셀 검사로 표현하지 않습니다.</p><p>같은 영상의 정상 시범·변형·촬영 방향은 같은 원본·참여자 그룹의 train 후보입니다. YTW 측면과 정면은 동기화된 동일 시도가 아니므로 점수를 합쳐 총점을 만들지 않습니다. 영상의 정상·오류 설명과 제목을 숫자 점수로 바꾸지 않았습니다.</p><p class="notice">AI의 비전문가 직접 검토 초안입니다. Chris 플랭크 한 구간은 4항목 초안과 공통 관절 관측을 통과했지만 정상 예시 1개 추가이며 오류 부족은 해결되지 않았습니다. 플랭크 총점 AI는 여전히 학습 자료 부족으로 산출하지 않습니다. 기존 AI-Hub 정지 항목 모델을 이 총점으로 표시하지 않습니다. 독립 최종 시험은 0개입니다.</p></section>'
    )
    for source in exports["records"]:
        check_resources()
        row, draft = joints[source["id"]], drafts[source["id"]]
        cache = np.load(row["extraction"]["cache"], allow_pickle=False)
        index = len(cache["frame_numbers"]) // 2
        frame_index = int(cache["frame_numbers"][index])
        cap = cv2.VideoCapture(
            source["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame_index)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Overlay frame unavailable")
        xy = cache["image_xy"][index].copy()
        xy[~cache["observed"][index].astype(bool)] = np.nan
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
                    (230, 170, 30),
                    2,
                )
        for point in xy:
            if np.isfinite(point).all():
                cv2.circle(
                    pixels,
                    tuple(np.rint(point).astype(int)),
                    3,
                    (40, 235, 250),
                    -1,
                )
        thumbnail = (
            audit.OUT / "youtube_segments" / (source["id"] + "_joints.jpg")
        )
        if not cv2.imwrite(str(thumbnail), pixels):
            raise OSError("Joint thumbnail write failed")
        start = source["start_frame"] / source["fps"]
        end = source["end_frame_exclusive"] / source["fps"]
        original = (
            pathlib.Path(source["source_path"])
            .relative_to(audit.OUT)
            .as_posix()
        )
        preview = row["rule_preview"]
        score = (
            "보류"
            if preview is None or preview.get("score") is None
            else str(preview["score"])
        )
        holds = (
            "; ".join(row["hold_reasons"])
            or "공통 관절 관측 통과; 규칙 미리보기이며 학습 정답이 아님"
        )
        for key, description in {
            "different_exercise_variant_not_an_error_label": "무릎 지지 변형이므로 기본 플랭크 오류 라벨에서 제외",
            "oblique_elbow_sagittal_position_not_score_approved": "사선 구도로 팔꿈치의 앞뒤 위치 점수 확정 불가",
            "single_phase_view_not_synchronized_complete_YTW": "단일 단계·시점이며 동기화된 전체 YTW가 아님",
        }.items():
            holds = holds.replace(key, description)
        parts.append(
            f'<section><h2>{esc(source["id"])}</h2><p>{esc(source["kind"])} · 원본 {start:.3f}~{end:.3f}초<br>직접 초안 {draft["draft_levels"]} · 초안 총점 {draft["total_score"]} · AI 총점 보류 · 규칙 미리보기 {score}</p><p>{esc(draft["note_ko"])}</p><p>{esc(holds)}</p><img loading="lazy" src="{thumbnail.relative_to(audit.OUT).as_posix()}" alt="Heavy 관측 관절"><details><summary>선택 원본 구간 보기</summary><video preload="none" controls src="{esc(original)}#t={start:.3f},{end:.3f}"></video></details><p>제작자: {esc(source["creator"])} · <a href="{esc(source["source_url"])}">원본 YouTube</a> · <a href="https://www.youtube.com/t/creative_commons">Creative Commons Attribution 재사용 표기</a><br>원본 그룹: {esc(source["source_group"])}</p></section>'
        )
    w = audit.read_json(audit.OUT / "next_w_results.json")
    parts.append(
        "<section><h2>새 W 개발 사례 3개</h2><p>각 36개 시간순 표본과 원본 장면을 직접 검토했습니다. 두 개는 등급 불확실, 한 개는 W 대신 팔을 거의 편 채 들어 0등급 초안입니다. 고정 규칙·단계 코드를 변경하지 않았습니다.</p><table><tr><th>후보</th><th>직접 W 초안</th><th>자동 구간 규칙</th><th>직접 구간 진단</th></tr>"
    )
    for row in w["results"]:
        parts.append(
            f'<tr><td>{esc(row["id"])}</td><td>{row["draft_grade"]}</td><td>{row["automatic"]["grade"]}</td><td>{row["direct_window_diagnostic"]["grade"]}</td></tr>'
        )
    parts.append(
        '</table><p>새 0등급 사례의 자동 구간은 0등급이지만 직접 지정 범위는 1등급입니다. 구간 경계 민감도가 남아 있으며 한 사례의 일치를 일반 정확도로 표현하지 않습니다. 두 불확실 초안의 규칙 출력을 정답으로 채우지 않았습니다.</p><p><a href="next_w_direct_review.json">직접 초안</a> · <a href="next_w_results.json">고정 규칙 결과</a> · <a href="next_view_direct_decisions.json">새 QEVD 15개·외부 후보 선별</a></p></section></html>'
    )
    target = audit.OUT / "youtube_review.html"
    target.write_text("".join(parts), encoding="utf-8")
    index = audit.OUT / "index.html"
    page = index.read_text(encoding="utf-8")
    if 'href="youtube_review.html"' not in page:
        page = page.replace(
            "</h1>",
            '</h1><p><a href="youtube_review.html">YouTube 구간 분리·관절 검토</a></p>',
            1,
        )
        index.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
