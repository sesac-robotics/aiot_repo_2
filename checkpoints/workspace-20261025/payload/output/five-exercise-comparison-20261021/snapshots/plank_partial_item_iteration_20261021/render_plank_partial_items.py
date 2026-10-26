"""항목별 부분 입력 AI·규칙·신규 출처 검증을 로컬 웹에 연결한다."""

import html
import pathlib
import shutil

import cv2

import audit
import render
from plank_item_observation import ITEMS

NAMES = ["몸통·골반 정렬", "팔꿈치 전후 위치", "다리·몸 지지", "지지 지속성"]


def run() -> pathlib.Path:
    """관측 항목·초안·두 방식·관절 영상을 HTML로 저장한다.

    Returns:
        pathlib.Path: 결과 HTML 절대 경로. 부분 항목을 100점으로
        재배점하지 않으며 단일 등급 학습은 상수 기준선으로 표시한다.

    Raises:
        OSError: 입력·캐시·영상·HTML 접근 실패.
        ValueError: 영상 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    trial = audit.read_json(audit.OUT / "plank_partial_item_trial.json")
    prospective = audit.read_json(audit.OUT / "plank_prospective_results.json")
    direct = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "plank_prospective_direct.json")[
            "reviews"
        ]
    }
    exports = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_prospective_plan_exports.json"
        )["records"]
    }
    esc = html.escape
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>부분 영상 항목 평가</title><style>body{font:16px/1.65 system-ui;max-width:1100px;padding:24px;margin:auto;background:#14212c;color:#eef5ff}section{background:#243a4b;margin:18px 0;padding:18px;border-radius:12px}a{color:#88dcff}td,th{padding:9px;text-align:left;border-bottom:1px solid #678}table{width:100%;border-collapse:collapse}video,img{max-width:100%;width:800px}.notice{border-left:4px solid #ffd17d;padding:12px}small{color:#cdd}</style><h1>부분 영상도 관측 가능한 항목은 활용</h1><p><a href="index.html">5종 비교</a> · <a href="plank_improvement_trial.html">앞선 플랭크 실험</a> · <a href="plank_partial_item_trial.json">항목별 결과</a> · <a href="plank_partial_label_revisions.json">직접 재검토 개정 이력</a> · <a href="plank_prospective_results.json">고정 모델의 신규 출처 결과</a></p><section><h2>전신 대신 항목에 필요한 정보 확인</h2><p>현재 플랭크 네 항목은 얼굴·손목의 완전한 노출을 요구하지 않습니다. 팔꿈치 위치는 어깨·팔꿈치·골반이 보이면 평가할 수 있습니다. 몸통 정렬은 어깨·골반·발목, 다리 지지와 지속성의 관절 대리값은 골반·무릎·발목이 필요합니다. 해당 관절의 동시 관측률 80%와 구도 적합성을 항목별로 확인합니다.</p><p>정렬은 짧은 구간도 5개 이상 표본이 있으면 활용합니다. 3초 지지 지속성과 총점은 별도로 판단합니다. 일부 항목만 확인되면 그 항목 등급만 표시하며 100점으로 환산하지 않습니다. 관절만으로 실제 하중·바닥 접촉을 확정하지 않습니다.</p><p class="notice">직접 라벨은 AI가 시간순 장면을 검토한 비전문가 실험 초안입니다. 전문가 정답·독립 최종 시험이 아닙니다. 공통 총점 모델과 규칙은 교체하지 않았습니다.</p></section><section><h2>부분 초안을 활용한 항목별 학습</h2><p>기존 총점 학습 10개 구간과 부분 초안의 유효한 항목을 사용했습니다. Pannell의 가까운 팔꿈치 잘림 때문에 보류했던 항목은, 화면 안의 반대쪽 어깨·팔꿈치를 직접 재검토해 2등급 초안으로 별도 개정했습니다. 원래 라벨 파일은 보존하며 이미 모델 출력을 본 뒤의 재검토임을 기록했습니다. 항목마다 필요한 관절만 입력받는 별도 ExtraTrees 모델이며, 학습 폴드 안에서 등급·참여자 비중을 맞췄습니다. 규칙 출력은 학습 라벨이나 특징에 넣지 않았습니다.</p><table><tr><th>항목</th><th>학습 구간 / 그룹</th><th>부분 자료 추가 항목 수</th><th>참여자 제외 등급 MAE AI / 규칙</th><th>모델 상태</th></tr>"""
    ]
    for index, row in enumerate(trial["items"]):
        parts.append(
            f'<tr><td>{NAMES[index]}</td><td>{row["training_records"]} / {row["training_people"]}</td><td>{len(row["added_partial_records"])}</td><td>{row["person_heldout_ai_MAE_grade"]:.3f} / {row["person_heldout_rule_MAE_grade"]:.3f}</td><td>{"한 등급뿐인 상수 기준선" if row.get("model_status")=="single_grade_constant_baseline" else "미검증 항목 회귀"}</td></tr>'
        )
    parts.append(
        "</table><p>다리 지지 항목은 AI의 참여자 제외 오차가 규칙보다 큽니다. 팔꿈치와 지지 지속성은 학습 등급이 하나뿐이므로 작은 오차를 오류 인식 능력으로 해석하면 안 됩니다. 부분 자료를 활용할 수 있게 된 것과 모델 정확도가 좋아진 것은 구분합니다.</p></section><section><h2>기존 부분 초안의 실제 관측 상태</h2><table><tr><th>구간</th><th>사용 가능한 항목</th><th>판정 불가 이유</th></tr>"
    )
    for row in trial["records"]:
        if row["split"] != "train" or not row["partial"]:
            continue
        valid = [
            NAMES[i]
            for i, item in enumerate(row["observation"]["items"])
            if item["eligible"]
        ]
        parts.append(
            f'<tr><td>{esc(row["id"])}</td><td>{" · ".join(valid) or "현재 관절 입력으로 사용 불가"}</td><td>{"필요 관절 미관측 / 구도·시간·직접 초안 조건을 항목별로 확인; 상세 JSON 참조"}</td></tr>'
        )
    parts.append(
        "</table><p>Colossus 장면의 직접 초안은 남겨 두었습니다. 해당 구간에서는 Heavy가 필요한 하체 관절을 충분히 추출하지 못해, 현재 관절 모델에 넣지 않았습니다. 자료 전체를 삭제하거나 사람의 직접 관측까지 불가능하다고 판단한 것은 아닙니다.</p></section><section><h2>학습에서 제외한 신규 출처</h2><p>아래 두 출처의 모델·규칙은 영상 직접 라벨링 전에 고정한 스냅샷으로 계산했습니다. 새 영상은 기존·항목별 모델 어느 쪽에도 학습하지 않았습니다. 기존에 확인한 Amanda·Ottobre 개발 확인 자료도 항목별 결과 JSON에 함께 보존했습니다.</p></section>"
    )
    verification = {}
    for row in prospective["results"]:
        identifier = row["id"]
        draft = direct[identifier]
        parts.append(
            f'<section><h2>{esc(identifier)}</h2><p>{esc(draft["note_ko"])}</p><p>직접 초안 등급 {esc(str(draft["draft_levels"]))} · 총점 {draft["total_score"] if draft["total_score"] is not None else "판정 불가"}</p><table><tr><th>항목</th><th>직접 초안</th><th>항목 AI</th><th>항목 규칙</th></tr>'
        )
        for index, item in enumerate(trial["items"]):
            prediction = next(
                (v for v in item["validation"] if v["id"] == identifier), None
            )
            parts.append(
                f'<tr><td>{NAMES[index]}</td><td>{draft["draft_levels"][index] if draft["draft_levels"][index] is not None else "판정 불가"}</td><td>{format(prediction["ai_grade"],".3f") if prediction else "판정 불가"}</td><td>{prediction["rule_grade"] if prediction else "판정 불가"}</td></tr>'
            )
        parts.append(
            "</table><p>다음 총점은 앞서 고정한 전신 입력 모델 결과이며 위 항목별 모델과 다릅니다.</p><table><tr><th>방식</th><th>등급</th><th>총점</th></tr>"
        )
        for method, value in row["methods"].items():
            parts.append(
                f'<tr><td>{esc(method)}</td><td>{esc(str(value["levels"]))}</td><td>{value["score"] if value["score"] is not None else "판정 불가"}</td></tr>'
            )
        folder = audit.OUT / "plank_partial_overlays" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            pathlib.Path(row["extraction"]["cache"]), folder / "side.npz"
        )
        video = render.overlay(
            dict(
                id=identifier,
                exercise="plank",
                video_path=exports[identifier]["video_path"],
            ),
            dict(
                cache_folder=str(folder),
                ai_score=row["methods"]["balanced_leaf2"]["score"],
                rule_score=row["methods"]["trial_rule"]["score"],
            ),
        )
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        frames = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if frames == 15:
                cv2.imwrite(str(folder / "preview.jpg"), pixels)
            frames += 1
        cap.release()
        if frames < 5:
            raise ValueError("Overlay cannot be decoded")
        verification[identifier] = frames
        parts.append(
            f'<video controls preload="metadata" poster="{(folder/"preview.jpg").relative_to(audit.OUT)}" src="{video.relative_to(audit.OUT)}"></video><p><a href="{esc(draft["source_url"])}">출처: {esc(draft["creator"])}</a> · CC BY 표기 확인 · 무음 구간 분리와 크롭, 관절 표시를 적용함.</p></section>'
        )
    parts.append(
        "<section><h2>확인된 한계와 다음 보충</h2><p>새 정상 구간에서 기존 규칙은 초안 100점과 일치했지만 골반 높이 보정 규칙은 90점으로 감점했습니다. 따라서 그 보정안을 공통 규칙으로 채택하지 않습니다. 새 사선 구간의 경미한 골반 이탈은 비전문가 1등급 초안으로 보존했고, 미세한 팔꿈치 등급과 총점은 채우지 않았습니다.</p><p>단일 등급인 팔꿈치·지지 지속성에는 각 항목에 맞는 오류·중간 등급의 관측 가능한 자료가 더 필요합니다. 새로운 자료는 원본·참여자 단위로 분리하고 검증 자료를 반복 학습에 넣지 않습니다.</p></section></html>"
    )
    target = audit.OUT / "plank_partial_items.html"
    target.write_text("".join(parts), encoding="utf-8")
    audit.save_json(
        audit.OUT / "plank_partial_render_verification.json",
        dict(
            decoded_overlay_frames=verification,
            local_browser_visual_check=False,
        ),
    )
    return target


if __name__ == "__main__":
    print(run())
