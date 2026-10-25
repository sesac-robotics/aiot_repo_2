"""추가 자료·별도 모델·실패·관절 정지화면을 한 로컬 화면으로 연결한다.

새 영상 디코딩·GPU 추출·재학습은 하지 않는다. 이미 저장된 원본 PNG와
동일 프레임의 관절을 SVG로 표시하고 검증된 관절 표시 영상을 연결한다.
원본 영상 플레이어는 별도로 구분한다.
"""

import html
import pathlib

import joblib
import numpy as np
from PIL import Image

import audit
from cpu_analysis_guard import check_resources
from render import CONNECTIONS
from w_item_learning import features
import w_component_trial
import ytw_first_cycle_trial


def layer(cache: pathlib.Path, png: pathlib.Path, frame: int) -> str:
    """원본 PNG와 정확히 같은 프레임의 관절 SVG를 반환한다.

    Args:
        cache: image_xy·frame_numbers가 있는 NPZ 경로.
        png: 원본 프레임 PNG 경로. 재인코딩 영상에서 추출하지 않는다.
        frame: 원본 프레임 인덱스, 0부터 시작.

    Returns:
        str: 픽셀 좌표계 SVG. 관측 NaN 관절·선은 그리지 않는다.

    Raises:
        OSError: 캐시·PNG 접근 실패.
        ValueError: 일치하는 표본 프레임이 없거나 중복된 경우.
    """
    with np.load(cache) as data:
        indices = np.flatnonzero(data["frame_numbers"] == frame)
        if len(indices) != 1:
            raise ValueError("Exact still/landmark frame required")
        points = data["image_xy"][indices[0]]
    with Image.open(png) as image:
        width, height = image.size
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="원본 프레임 {frame} 관절 표시"><image href="{png.relative_to(audit.OUT)}" width="{width}" height="{height}"/><g class="landmarks" fill="#ffc400" stroke="#50ed75" stroke-width="3">'
    ]
    for a, b in CONNECTIONS:
        if np.isfinite(points[[a, b]]).all():
            parts.append(
                f'<line x1="{points[a,0]}" y1="{points[a,1]}" x2="{points[b,0]}" y2="{points[b,1]}"/>'
            )
    for point in points:
        if np.isfinite(point).all():
            parts.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="4"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def run() -> pathlib.Path:
    """기존 결과를 구분해 표시하고 자동 W 결과는 같은 시행 겹침을 감사한다.

    Returns:
        pathlib.Path: followup_results.html 절대 경로. 화면은 연구 초안이며
        전문가 정확도·5종 완료로 표현하지 않는다. 이미 생성한 관절 영상만 연결한다.

    Raises:
        OSError: 보고서·모델·이미지 접근 실패.
        ValueError: 직접 초안·관절 캐시 대응 오류.
        RuntimeError: CPU 분석 보호 조건 위반.
    """
    check_resources()
    plank = audit.read_json(audit.OUT / "followup_plank_learning_results.json")
    events = audit.read_json(audit.OUT / "followup_event_learning_results.json")
    w = audit.read_json(audit.OUT / "followup_W_learning_results.json")
    phase = audit.read_json(audit.OUT / "first_cycle_trial_results.json")
    mono = audit.read_json(audit.OUT / "plank_monotone_trial_results.json")
    world = audit.read_json(audit.OUT / "plank_world_trial_results.json")
    masked = audit.read_json(
        audit.OUT / "plank_masked_multitask_trial_results.json"
    )
    draftW = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_W_direct.json")[
            "reviews"
        ]
    }
    caches = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_extraction.json")[
            "W_records"
        ]
    }
    model = joblib.load(audit.OUT / "followup_W_model/model.joblib")["model"]
    automatic = []
    for row in w["new_development_validation"]:
        draft = draftW[row["id"]]
        with np.load(caches[row["id"]]["cache"]) as data:
            seq = data["image_xy"].copy()
            frames = data["frame_numbers"]
        detected = ytw_first_cycle_trial.detect(seq)
        win = detected["phases"].get("W_attempt")
        item = dict(
            id=row["id"],
            target=row["target"],
            ai=None,
            rule=None,
            same_attempt=False,
            window=None,
        )
        if win:
            a, b = win["start_index"], win["end_index"]
            x = features(seq, a, b)
            item["window"] = [int(frames[a]), int(frames[b])]
            ca, cb = draft["direct_W_core_frames"]
            item["same_attempt"] = bool(frames[a] <= cb and frames[b] >= ca)
            if x is not None:
                item.update(
                    ai=float(model.predict([x])[0]),
                    rule=w_component_trial.evaluate(seq, a, b)["grade"],
                )
        automatic.append(item)
    auto_report = dict(
        records=automatic,
        known_direct_cases=len(automatic),
        automatic_ai_assessed=sum(r["ai"] is not None for r in automatic),
        same_attempt_pairable=sum(
            r["ai"] is not None and r["same_attempt"] for r in automatic
        ),
        posthoc_development_diagnostic=True,
        independent_final_test=False,
    )
    audit.save_json(
        audit.OUT / "followup_W_automatic_analysis.json", auto_report
    )
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>추가 자료·재학습 검증</title><style>body{font:16px/1.65 system-ui;background:#13212b;color:#edf6ff;max-width:1180px;margin:auto;padding:24px}a{color:#8bdbff}section{background:#203747;padding:20px;margin:22px 0;border-radius:12px}td,th{padding:9px;text-align:left;border-bottom:1px solid #587080}table{width:100%;border-collapse:collapse}svg,video,img{max-width:100%;width:700px}.scroll{overflow-x:auto}#pose-toggle:not(:checked)~main .landmarks{display:none}.note{color:#ffd092}</style></head><body><h1>추가 자료·재학습·검증</h1><p><a href="unified_status.html">통합 현황</a> · <a href="followup_iteration_verification.json">검사·자원 기록</a> · <a href="followup_source_dispositions.json">선별·제외·출처</a></p><p>Heavy는 관절 추출기입니다. 점수 AI는 별도 학습 모델이며 직접 등급은 비전문가 실험 초안입니다. 규칙 등급·제공자 문자열을 AI 정답으로 쓰지 않았습니다.</p><input type="checkbox" id="pose-toggle" checked><label for="pose-toggle">관절 정지화면 표시</label><main>'
    ]
    parts.append(
        f'<section><h2>플랭크: 오류 원본 추가 후 재학습</h2><p>기존 {plank["old_training_records"]}구간에 새 참여자 원본의 정상·높은 골반 오류 {plank["new_training_records"]}구간을 추가했습니다. 총 {plank["total_training_records"]}구간 / {plank["training_person_groups"]}참여자 그룹입니다. 같은 사람·원본을 통째로 제외한 동일 12구간에서 비교했습니다.</p><div class="scroll"><table><tr><th>방식</th><th>평균 절대 점수 오차</th></tr>'
    )
    for name, value in [
        (
            "보강 전 ExtraTrees",
            plank["LOPO_metrics"]["before_ai"]["MAE_points"],
        ),
        ("보강 후 ExtraTrees", plank["LOPO_metrics"]["after_ai"]["MAE_points"]),
        (
            "기하 통계로 학습한 2D 단조 회귀",
            mono["metrics"]["after_ai"]["MAE_points"],
        ),
        (
            "관측 관절의 추정 3D 단조 회귀",
            world["metrics"]["after_ai"]["MAE_points"],
        ),
        (
            "직접 부분 라벨의 항목별 AI",
            masked["metrics"]["after_ai"]["MAE_points"],
        ),
        ("동결 Rule Base", plank["LOPO_metrics"]["rule_score"]["MAE_points"]),
    ]:
        parts.append(f"<tr><td>{name}</td><td>{value:.3f}점</td></tr>")
    parts.append(
        '</table></div><p class="note">보강 후 총점 AI가 악화되어 공통 모델을 교체하지 않았습니다. 단조 회귀·3D·부분 항목 학습 실험도 개선되지 않았습니다. 부분 항목 모델은 항목별로 13~15구간을 사용했으나 오류 참여자가 항목당 1~4그룹뿐입니다. 3D는 Heavy 추정값이며 실측 정답이 아닙니다. 팔꿈치·지속성 머리는 현재 전신 학습 초안의 등급이 하나뿐인 상수 기준선입니다.</p><p><a href="followup_plank_learning_results.json">전체 예측</a> · <a href="plank_monotone_trial_results.json">단조 실험</a> · <a href="plank_world_trial_results.json">투영·3D 진단</a> · <a href="plank_masked_multitask_trial_results.json">부분 라벨 항목 학습</a></p></section>'
    )
    for id, title, frame, png in [
        (
            "followup_fitness_basic",
            "새 학습 원본: 정상 초안",
            1034,
            "followup_fitness_basic_1034.png",
        ),
        (
            "followup_fitness_highhip",
            "같은 원본: 높은 골반 오류 초안",
            1244,
            "followup_fitness_highhip_1244.png",
        ),
        (
            "followup_amanda_normal",
            "별도 참여자: 정상 개발 검증",
            84,
            "followup_amanda_normal_84.png",
        ),
    ]:
        draft = next(
            r
            for r in audit.read_json(audit.OUT / "followup_plank_direct.json")[
                "reviews"
            ]
            if r["id"] == id
        )
        parts.append(
            f'<section><h3>{title}</h3><p>{esc(draft["note_ko"])}</p><p>직접 등급 초안 {draft["draft_levels"]} / 총점 {draft["total_score"]}. <a href="{esc(draft["source_url"])}">{esc(draft["creator"])}</a> · CC BY 메타데이터 보존.</p>'
        )
        parts.append(
            layer(
                audit.OUT / "heavy" / id / "side.npz",
                audit.OUT / "review" / png,
                frame,
            )
        )
        start = draft["segment"]["start_frame"] / draft["segment"]["fps"]
        end = (draft["segment"]["end_frame"] + 1) / draft["segment"]["fps"]
        source = pathlib.Path(draft["video_path"]).relative_to(audit.OUT)
        parts.append(
            f'<p>표시 AI는 보강 후보 모델의 제외 참여자 예측이며 공통 모델 교체 결과가 아닙니다.</p><video controls preload="metadata" src="followup_overlays/{id}/overlay.mp4" poster="followup_overlays/{id}/preview.jpg"></video><details><summary>원본 영상 해당 구간</summary><video controls preload="none" src="{source}#t={start:.3f},{end:.3f}"></video></details></section>'
        )
    parts.append(
        "<section><h2>상지 지지 변경 사건</h2><p>새 QEVD 3명의 변경 사건과 머리 움직임·골반 높이 변화의 음성 대조 2개를 추가했습니다. 자세 총점과 분리한 사건 확률 모델입니다.</p><table><tr><th>방식</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr>"
    )
    for name, key in [
        ("보강 전 AI", "before_event"),
        ("보강 후 AI", "after_event"),
        ("고정 상완 변위 규칙", "rule_event"),
    ]:
        m = events["metrics"][key]
        parts.append(
            f'<tr><td>{name}</td><td>{m["TP"]}</td><td>{m["FN"]}</td><td>{m["FP"]}</td><td>{m["TN"]}</td></tr>'
        )
    parts.append(
        f'</table><p>{events["training_records"]}구간 / {events["training_people"]}참여자 그룹, 양성 {events["positive_person_groups"]}참여자 그룹. 같은 참여자·원본 제외 진단이며 경계 0.25는 이번 결과 후 조정하지 않았습니다. 머리·골반 움직임 대조에서 오탐이 남았습니다.</p><a href="followup_event_learning_results.json">전체 사건 결과</a></section>'
    )
    parts.append(
        f'<section><h2>W: 실제 항목 모델과 규칙</h2><p>기존 직접 W 초안 {w["training_records"]}구간 / {w["training_people"]}참여자 그룹으로 별도 항목 AI를 학습했습니다. 새 직접 등급 5개 중 필수 관절을 관측한 4개만 조건부 비교가 가능했습니다. 직접 표시한 핵심 구간을 사용한 결과입니다.</p><table><tr><th>사례</th><th>직접 초안</th><th>구간 조건부 AI</th><th>부분 성분 규칙</th><th>첫 자동 구간 AI / 같은 시도</th></tr>'
    )
    for row in w["new_development_validation"]:
        a = next(r for r in automatic if r["id"] == row["id"])
        parts.append(
            f'<tr><td>{esc(row["id"])}</td><td>{row["target"]}</td><td>{row["ai"]}</td><td>{row["component_rule"]}</td><td>{a["ai"]} / {a["same_attempt"]}</td></tr>'
        )
    m = w["new_conditional_metrics"]
    parts.append(
        f'</table><p>같은 직접 구간에서 평균 절대 등급 오차: AI {m["ai"]["MAE_grade"]:.3f}, 규칙 {m["component_rule"]["MAE_grade"]:.3f}. 판정 가능 4개에는 0등급 사례가 포함되지 않았습니다. 자동 입력에서는 평가 {auto_report["automatic_ai_assessed"]}/5, 같은 시도와 겹쳐 공통 비교 가능한 것은 {auto_report["same_attempt_pairable"]}/5입니다. 이 차이를 숨기고 자동 W 정확도라고 표현하지 않습니다.</p>'
    )
    parts.append(
        f'<p>기존 22구간의 부분 성분 규칙 오차는 {w["development_metrics"]["old_rule"]["MAE_grade"]:.3f}→{w["development_metrics"]["component_rule"]["MAE_grade"]:.3f}등급이지만 새 4구간에서는 개선되지 않았습니다. 최초 시행 선택은 전체 {phase["summary"]["first"]["reference_cases"]}개 참조에서 핵심 구간 겹침 {phase["summary"]["longest"]["core_overlapping"]}→{phase["summary"]["first"]["core_overlapping"]}개로 소폭 개선됐습니다. 공통 규칙은 교체하지 않았습니다.</p><p><a href="followup_W_learning_results.json">W 학습·조건부 비교</a> · <a href="followup_W_automatic_analysis.json">자동 입력 결과</a> · <a href="first_cycle_trial_results.json">반복 시행 선택</a></p></section>'
    )
    parts.append(
        "<section><h2>W 관절 영상 — 직접 핵심 구간, 등급 0~2</h2><p>직접 표시한 구간의 조건부 결과입니다. 자동 단계 인식 완료 영상이 아닙니다.</p>"
    )
    for row in w["new_development_validation"]:
        if row["ai"] is not None:
            name = row["id"]
            parts.append(
                f'<details><summary>{esc(name)} / 초안 {row["target"]}</summary><video controls preload="metadata" src="followup_overlays/{name}/overlay.mp4" poster="followup_overlays/{name}/preview.jpg"></video></details>'
            )
    parts.append(
        '</section><section><h2>새 측면 Y·전경 관측 자료</h2><p><a href="followup_y_side_results.json">CPU Heavy·성분 비교</a> · <a href="followup_y_side_direct.json">직접 초안</a></p><p>Josh Sabraw Training의 CC 시범에서 무릎 아래가 잘려도 필요한 어깨·팔꿈치·손목·골반을 관측했습니다. 같은 사람의 두 구간입니다. 전환 구간은 수행 중 자세 오류로 세지 않으며 Y 등급도 만들지 않습니다.</p>'
    )
    y_report = audit.read_json(audit.OUT / "followup_y_side_results.json")
    for row in y_report["records"]:
        name = row["id"]
        r = row["rule"]
        parts.append(
            f'<h3>{esc(name)}</h3><p>전경 초안 / 규칙 {row["direct_hinge_grade"]} / {r["hinge_grade"]} · Y 측면 성분 초안 / 규칙 {row["direct_Y_sagittal_grade"]} / {r["y_sagittal_component_grade"]}. 전체 Y·YTW 총점은 미판정입니다.</p><video controls preload="metadata" src="followup_overlays/{name}/overlay.mp4" poster="followup_overlays/{name}/preview.jpg"></video>'
        )
    parts.append(
        "</section><section><h2>관측 한계와 전원 상태</h2><p>추가 YTW 18개 구도·8개 상세 표본과 새 측면 Y 원본을 검토했습니다. 새 CC 엎드린 Y–W는 다른 변형으로 남겨 전체 YTW 학습에서 제외했습니다. 같은 시행에서 양팔 Y·전경·T·W를 모두 관측할 다양한 참여자·오류 원본과 별도 Y 성분 AI의 오류 학습 자료는 아직 부족합니다.</p><p>기존 GPU 관절 추출의 AC 필수 보호는 유지합니다. 전원 분리 후 캐시 통계는 CPU25%·배터리75% 중단, 새 짧은 CPU Heavy·표시 작업은 CPU25%·시작 잔량85%·중단80%·CPU70℃/GPU60℃ 조건으로 별도 실행했습니다. CPU 추출 결과는 이전 GPU 처리 시간 표에 합산하지 않았습니다. 관절 영상 9개는 전체 프레임 디코딩을 검사했습니다.</p><p>참여자·영상 그룹은 출처 메타데이터의 잠정 식별자이며 독립적인 실제 인물 재식별 보증이 아닙니다. 기존 반복 확인 자료를 새 독립 최종 시험으로 세지 않았습니다.</p></section></main></body></html>"
    )
    target = audit.OUT / "followup_results.html"
    target.write_text("".join(parts), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
