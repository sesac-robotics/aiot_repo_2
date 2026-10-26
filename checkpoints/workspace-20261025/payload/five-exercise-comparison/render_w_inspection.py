"""추가 W 후보의 시간축·관절 영상·구성 요소를 연결한 검토 화면을 만든다.

공통 단계·채점 규칙과 학습 라벨은 수정하지 않는다. 부분 T→W 인식은
채택 전 진단으로 따로 표시하며 Y·전경·총점으로 확장하지 않는다.
"""

import html
import pathlib
import cv2
import numpy as np

import audit
import render
from phase_render import timeline
from resources import check_resources


def frame_windows(phases: dict, frames: np.ndarray) -> dict:
    """표본 인덱스 구간을 원본 프레임 구간으로 변환한다.

    Args:
        phases: 단계별 양 끝 포함 표본 인덱스 사전.
        frames: 원본 프레임 번호 N개 배열.

    Returns:
        dict: 단계별 start_frame·end_frame. 단위는 원본 프레임이다.

    Raises:
        IndexError: 단계 인덱스가 캐시 범위를 벗어난 경우.
    """
    return {
        name: dict(
            start_frame=int(frames[w["start_index"]]),
            end_frame=int(frames[w["end_index"]]),
        )
        for name, w in phases.items()
    }


def run() -> dict:
    """12개 W 후보의 H.264 관절 영상과 근거 HTML을 생성·검사한다.

    Returns:
        dict: 원본·캐시 해시, 영상 디코딩 프레임 수, 웹 경로. 검사는
        파일 무결성이며 전문가 정확도·직접 영상 전체 검토를 뜻하지 않는다.

    Raises:
        ValueError: 원본 해시·캐시 지문·출력 디코딩 프레임 불일치.
        OSError: 입력·캐시·출력 접근 실패.
        RuntimeError: 온도·전원·메모리 보호 중단.
        subprocess.SubprocessError: H.264 변환 실패.
    """
    direct = audit.read_json(audit.OUT / "w_targeted_direct_review.json")
    results = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "w_targeted_results.json")[
            "results"
        ]
    }
    trials = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "phase_partial_trial.json")[
            "results"
        ]
        if r["batch"] == "w_targeted"
    }
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W 추가 후보 관절·구간 검토</title><style>body{background:#14212c;color:#edf4fa;font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px}section{background:#213544;padding:20px;margin:20px 0;border-radius:12px}a{color:#80dcff}video,img,svg{max-width:100%;height:auto}video{width:min(100%,800px);aspect-ratio:16/9;object-fit:contain}button{background:#80dcff;color:#14212c;border:0;border-radius:6px;padding:10px 14px;cursor:pointer}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid #567}details{margin:12px 0}.scroll{overflow:auto}</style><h1>W 추가 12개 관절·구간 검토</h1><p><a href="index.html">점수 비교</a> · <a href="w_analysis.html">W 원인 분석</a> · <a href="phase_partial_trial.json">단계 인식 실험 원본</a></p><p>모든 자료는 노출된 개발 자료입니다. 직접 초안은 전문가 정답이 아닙니다. 녹색 선은 Heavy 관절 추정이며 보이지 않는 점은 연결하지 않습니다. 관절 영상은 원본의 10Hz 표본을 표시하며 전체 원본 동작을 빠짐없이 재생한 영상이 아닙니다. AI 점수 모델을 학습하거나 새 총점을 생성하지 않았습니다.</p><p>기존 규칙은 유지합니다. 별도 부분 구간 실험은 Y가 불명확할 때 T 뒤 좁아진 팔 시도를 찾습니다. 휴식·다른 반복 오인과 어두운 장면의 오판이 남아 자동 채점 기준에 채택하지 않았습니다. 총점·Y 정답으로 쓰지 않습니다.</p>"""
    ]
    checks = []
    for row in direct["reviews"]:
        check_resources()
        r = results[row["id"]]
        folder = audit.OUT / "w_targeted_cache" / row["id"]
        metadata = audit.read_json(folder / "primary.json")
        if (
            audit.digest(pathlib.Path(row["video_path"])) != row["video_sha256"]
            or metadata["config"]["source_sha256"] != row["video_sha256"]
        ):
            raise ValueError("Targeted W source changed")
        with np.load(folder / "primary.npz") as data:
            frames = data["frame_numbers"]
            if str(data["fingerprint"]) != metadata["fingerprint"]:
                raise ValueError("Targeted W cache mismatch")
        comp = dict(
            cache_folder=str(folder),
            ai_score=None,
            rule_score=None,
            rule_details=dict(
                evidence=dict(
                    phase_detection=r["phase_detection"],
                    w_shape=dict(morphology=r["automatic"]),
                )
            ),
        )
        video = render.overlay(dict(row, exercise="ytw"), comp)
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        if decoded != len(frames):
            raise ValueError("Targeted W video frame count differs")
        checks.append(
            dict(
                id=row["id"],
                frames=decoded,
                source_sha256=row["video_sha256"],
                cache_fingerprint=metadata["fingerprint"],
                video_sha256=audit.digest(video),
                video=str(video),
                direct_full_video_review=False,
            )
        )
        ident = html.escape(row["id"])
        core = row["direct_W_core_frames"]
        manual = r["manual_window_diagnostic"]
        partial = trials[row["id"]]["variants"]["partial_trial"]
        parts.append(
            f'<section id="{ident}"><h2>{ident}</h2><p>직접 W 초안: {row["W_grade_draft"]}/2 · 자동 W: {r["automatic"]["grade"]}/2 · 직접 구간 규칙 진단: {manual["grade"] if manual else None}/2</p><p>{html.escape(row["note_ko"])}</p><video id="video-{ident}" controls preload="none" src="{video.relative_to(audit.OUT)}"></video>'
        )
        selected_index = int(np.argmin(np.abs(frames - sum(core) / 2)))
        selected_frame = int(frames[selected_index])
        parts.append(
            f'<p><button data-seek="{selected_index / 10}" data-video="video-{ident}">직접 W 핵심 표본 f{selected_frame}로 이동</button></p>'
        )
        first, last = int(frames[0]), int(frames[-1])
        parts.append(
            "<h3>직접 W 핵심 구간</h3>"
            + timeline(
                dict(W_attempt=dict(start_frame=core[0], end_frame=core[1])),
                first,
                last,
            )
        )
        parts.append(
            "<h3>현재 자동 구간</h3>"
            + timeline(
                frame_windows(r["phase_detection"]["phases"], frames),
                first,
                last,
            )
        )
        parts.append(
            "<details><summary>채택 전 부분 T→W 실험</summary>"
            + timeline(
                frame_windows(partial["phases"]["phases"], frames), first, last
            )
            + f'<p>W 후보 등급 {partial["grade"]}/2. 기존 점수 비교에는 반영하지 않았습니다.</p></details>'
        )
        if manual:
            parts.append(
                '<h3>직접 구간에서 계산한 규칙 구성 요소</h3><div class="scroll"><table><tr><th>팔</th><th>굽힘 °</th><th>손 높이/몸통</th><th>팔꿈치 상승/상완</th><th>굽힘·손·팔꿈치 등급</th></tr>'
            )
            for side, arm in manual["arms"].items():
                if "projected_bend_deg" in arm:
                    parts.append(
                        f'<tr><td>{side}</td><td>{arm["projected_bend_deg"]:.1f}</td><td>{arm["hand_above_elbow_torso_ratio"]:.3f}</td><td>{arm["elbow_up_upper_arm_ratio"]:.3f}</td><td>{arm["bend_grade"]} · {arm["hand_grade"]} · {arm["elbow_position_grade"]}</td></tr>'
                    )
                else:
                    parts.append(
                        f'<tr><td>{side}</td><td colspan="4">관측 부족: {arm["observed_fraction"]:.0%}</td></tr>'
                    )
            parts.append("</table></div>")
        parts.append(
            f'<details><summary>직접 검토한 시간순 표본·원본 크기 장면</summary><img src="{pathlib.Path(row["evidence_image"]).relative_to(audit.OUT)}" alt="시간순 36장 표본"><img src="{pathlib.Path(row["raw_evidence_image"]).relative_to(audit.OUT)}" alt="직접 W 핵심 원본 크기 표본"></details></section>'
        )
        audit.save_json(
            audit.OUT / "w_inspection_partial.json",
            dict(checks=checks, complete=False),
        )
    parts.append(
        '<script>document.querySelectorAll("button[data-seek]").forEach(b=>b.addEventListener("click",()=>{const v=document.getElementById(b.dataset.video);const seek=()=>{v.currentTime=Number(b.dataset.seek);v.pause();};if(v.readyState>=1){seek();}else{v.addEventListener("loadedmetadata",seek,{once:true});v.preload="metadata";v.load();}}));</script></html>'
    )
    (audit.OUT / "w_inspection.html").write_text(
        "".join(parts).replace("None/2", "판정 불가"), encoding="utf-8"
    )
    report = dict(
        checks=checks,
        complete=True,
        new_total_score_labels=0,
        models_retrained=False,
        common_phase_rule_changed=False,
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "w_inspection_verification.json", report)
    return report


if __name__ == "__main__":
    print(run()["complete"])
