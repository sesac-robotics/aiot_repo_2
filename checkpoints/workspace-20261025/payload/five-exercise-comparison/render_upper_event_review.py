"""새 QEVD 상지 사건 초안과 참여자 제외 예측을 관절 영상으로 연결한다.

학습 참여자에 대한 최종 모델 예측을 검증값으로 표시하지 않는다.
상지 사건은 총점이나 전신 정상 판정으로 변환하지 않는다.
"""

import html
import shutil

import cv2

import audit
import render


def run() -> dict:
    """기존 Heavy 캐시를 표시하고 전체 영상 디코딩 여부를 저장한다.

    Returns:
        dict: 영상별 디코딩 표본 수와 검토용 중간 이미지 경로.
        AI 확률은 해당 참여자를 제외한 바깥 폴드 결과이다.

    Raises:
        OSError: 초안·관절 캐시·결과 접근 실패.
        ValueError: 예측 누락, 영상 디코딩 또는 이미지 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    drafts = audit.read_json(audit.OUT / "qevd_upper_event_direct.json")
    result = audit.read_json(audit.OUT / "upper_event_nested_results.json")
    predictions = {r["id"]: r for r in result["predictions"]}
    checks = {}
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        "<title>상지 사건 관절 검토</title><style>body{font:16px/1.6 "
        "system-ui;max-width:1000px;margin:auto;padding:24px}"
        "video{width:100%;max-width:600px}section{margin:24px 0}</style>"
        '<h1>새 QEVD 상지 사건 검토</h1><p><a href="unified_status.html">'
        '통합 현황</a> · <a href="upper_event_nested.html">중첩 진단</a>'
        "</p><p>비전문가 직접 검토 사건 초안입니다. 영상의 총점은 미판정입니다. "
        "AI는 해당 참여자·원본을 제외한 바깥 폴드 예측이며, 사건 경계는 "
        "남은 학습 자료 안에서만 선택했습니다. QEVD 원본 이용 조건을 "
        "따르며 외부 재배포용 자료가 아닙니다.</p>"
    ]
    for draft in drafts["reviews"]:
        record_id = draft["id"]
        prediction = predictions[record_id]
        source = audit.OUT / "upper_support_cache" / record_id
        folder = audit.OUT / "upper_event_overlays" / record_id
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "primary.npz", folder / "side.npz")
        video = render.overlay(
            dict(draft, exercise="plank"),
            dict(cache_folder=str(folder), ai_score=None, rule_score=None),
        )
        cap = cv2.VideoCapture(str(video))
        count = 0
        preview = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1
                if count == 15:
                    preview = frame
        finally:
            cap.release()
        if count < 15 or preview is None:
            raise ValueError("Incomplete upper-event overlay")
        image = folder / "preview.jpg"
        if not cv2.imwrite(str(image), preview):
            raise ValueError("Upper-event preview write failed")
        checks[record_id] = dict(decoded_frames=count, preview=str(image))
        relative = video.relative_to(audit.OUT)
        parts.append(
            f"<section><h2>{html.escape(record_id)}</h2><p>"
            f'{html.escape(draft["note_ko"])}</p><p>직접 사건 초안 '
            f'{draft["visible_upper_support_change"]} / AI 사건 확률 '
            f'{prediction["probability"]:.3f} / 학습 내부 경계 '
            f'{prediction["inner_selected_threshold"]:.2f} / 예측 사건 '
            f'{int(prediction["event"])}</p><video controls preload="metadata" '
            f'src="{relative}" poster="{image.relative_to(audit.OUT)}">'
            "</video></section>"
        )
    parts.append("</html>")
    (audit.OUT / "upper_event_review.html").write_text(
        "".join(parts),
        encoding="utf-8",
    )
    audit.save_json(audit.OUT / "upper_event_overlay_checks.json", checks)
    return checks


if __name__ == "__main__":
    print(run())
