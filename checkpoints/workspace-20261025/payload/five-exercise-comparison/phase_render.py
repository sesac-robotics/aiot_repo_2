"""Y·T·W 직접 검토 구간과 자동 시도 구간을 로컬 시간 막대로 표시한다.

표시는 점수 정답·정상 자세 확인이 아니며 골반 전경 깊이는 계속 보류한다.
"""

import html
import pathlib
import shutil

import audit

COLORS = {
    "Y_attempt": "#e9b44c",
    "T_attempt": "#63d6ab",
    "W_attempt": "#a5a5ff",
}


def timeline(windows: dict, start: int, end: int) -> str:
    """프레임 범위를 같은 축의 SVG 막대로 반환한다.

    Args:
        windows: 단계 이름별 start_frame·end_frame 사전.
        start: 축 시작 원본 프레임. 단위는 프레임이다.
        end: 축 종료 원본 프레임. start보다 커야 한다.

    Returns:
        str: 단계 이름·원본 프레임을 표시하는 SVG 문자열.

    Raises:
        ValueError: 축 범위가 유효하지 않은 경우.
    """
    if end <= start:
        raise ValueError("Invalid frame axis")
    parts = [
        '<svg viewBox="0 0 900 120" role="img" aria-label="원본 프레임 기준 단계 구간">'
    ]
    for j, (name, color) in enumerate(COLORS.items()):
        y = j * 36 + 4
        parts.append(
            f'<text x="0" y="{y+18}" fill="white">{name.replace("_attempt", " 시도")}</text>'
        )
        parts.append(
            f'<rect x="130" y="{y}" width="760" height="28" fill="#253644"/>'
        )
        if name in windows:
            a, b = windows[name]["start_frame"], windows[name]["end_frame"]
            x = 130 + 760 * (a - start) / (end - start)
            w = max(2, 760 * (b - a) / (end - start))
            parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="28" fill="{color}"><title>f{a}–f{b}</title></rect>'
            )
            parts.append(
                f'<text x="{x+3}" y="{y+18}" fill="#111">{a}–{b}</text>'
            )
    return "".join(parts) + "</svg>"


def run() -> str:
    """직접 표본·단계 비교·불확실성을 HTML로 저장한다.

    Returns:
        str: phases.html 절대 경로. 프레임 구간은 원본 좌표계다.

    Raises:
        OSError: 검토 JSON·근거 이미지를 읽거나 결과를 저장하지 못한 경우.
        ValueError: 직접·자동 구간의 축 계약이 잘못된 경우.
    """
    annotations = audit.read_json(audit.OUT / "ytw_phase_annotations.json")
    report = audit.read_json(audit.OUT / "ytw_phase_comparison.json")
    by_id = {r["id"]: r for r in annotations["records"]}
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y–T–W 단계 검토</title><style>body{background:#12212c;color:#edf4fa;font:16px/1.65 system-ui;max-width:1100px;margin:24px auto;padding:16px}section{background:#1b303e;padding:20px;margin:20px 0;border-radius:12px}a{color:#77d9ff}svg,img,video{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #536673;text-align:left}.scroll{overflow:auto}summary{cursor:pointer}</style><h1>Y–T–W 시도 구간 직접 표시·자동 인식</h1><p><a href="index.html">점수 비교로 돌아가기</a> · <a href="ytw_phase_annotations.json">직접 표시 원본</a> · <a href="ytw_phase_comparison.json">자동 대조 원본</a></p><p>이미 노출된 개발 자료입니다. Y 시도는 순서상의 첫 팔 들기 시도이며 정상 Y나 전방 깊이를 확정한 표시가 아닙니다. 직접 구간은 시간순 표본에서 확인한 핵심 구간으로 경계에 불확실성이 있습니다. 점수 라벨은 바꾸지 않았습니다.</p>"""
    ]
    for role, summary in report["summary_by_role"].items():
        title = (
            "설계용 학습 자료"
            if role == "design_training"
            else "기존 개발 검증 자료"
        )
        parts.append(
            f'<section><h2>{title}</h2><p>세 시도 구간 식별: 이전 {summary["previous_complete"]}/{summary["records"]} → 현재 {summary["new_complete"]}/{summary["records"]}. 대표점이 직접 핵심 구간 안에 든 수: 이전 {summary["previous_anchors_inside_core"]}/{summary["phase_anchors"]} → 현재 {summary["anchors_inside_direct_core"]}/{summary["phase_anchors"]}. 표본 기반 구간 중첩 평균: {summary["mean_sample_interval_IoU"]:.2f}. 자세 점수 정확도가 아닙니다.</p></section>'
        )
    for row in report["rows"]:
        direct = by_id[row["id"]]
        parts.append("<section><h2>" + html.escape(row["id"]) + "</h2>")
        if "detected" not in row:
            parts.append(
                "<p>점수 적격 캐시 없음. 단계를 강제로 채우지 않았습니다.</p></section>"
            )
            continue
        windows = direct["phase_windows"]
        all_frames = [v for w in windows.values() for v in w.values()]
        start = direct.get("segment", {}).get("start_frame", 0)
        end = direct.get("segment", {}).get("end_frame", max(all_frames))
        parts.append(
            "<h3>직접 검토 핵심 구간</h3>" + timeline(windows, start, end)
        )
        automatic = {
            c["phase"]: dict(
                start_frame=c["detected_frames"][0],
                end_frame=c["detected_frames"][2],
            )
            for c in row["phase_comparisons"]
        }
        parts.append(
            "<h3>자동 인식 유지 구간</h3>" + timeline(automatic, start, end)
        )
        parts.append(
            "<p>이전 대표 프레임: "
            + html.escape(str(row.get("previous_anchor_frames", [])))
            + " · 현재 보류 사유: "
            + html.escape(
                "없음"
                if row["detected"]["hold_reason"] is None
                else "순서에 맞는 세 유지 구간을 확인하지 못함"
            )
            + "</p>"
        )
        if row["phase_comparisons"]:
            parts.append(
                '<div class="scroll"><table><tr><th>시도</th><th>직접 핵심 구간</th><th>자동 시작·대표·끝</th><th>구간 중첩</th></tr>'
            )
            for c in row["phase_comparisons"]:
                parts.append(
                    f'<tr><td>{c["phase"]}</td><td>{c["direct_core_frames"]}</td><td>{c["detected_frames"]}</td><td>{c["sample_based_interval_IoU"]:.2f}</td></tr>'
                )
            parts.append("</table></div>")
        overlay = audit.OUT / "heavy" / row["id"] / "overlay.mp4"
        if overlay.exists():
            parts.append(
                f'<video controls preload="metadata" src="heavy/{row["id"]}/overlay.mp4"></video>'
            )
        src = pathlib.Path(direct["evidence_images"][0])
        dest = audit.OUT / "review" / ("phase_" + row["id"] + ".jpg")
        shutil.copy2(src, dest)
        parts.append(
            f'<details><summary>직접 검토한 시간순 표본</summary><img loading="lazy" alt="시간순 표본 검토판" src="review/{dest.name}"></details></section>'
        )
    target = audit.OUT / "phases.html"
    temporary = target.with_suffix(".tmp.html")
    temporary.write_text("".join(parts) + "</html>", encoding="utf-8")
    temporary.replace(target)
    return str(target)


if __name__ == "__main__":
    print(run())
