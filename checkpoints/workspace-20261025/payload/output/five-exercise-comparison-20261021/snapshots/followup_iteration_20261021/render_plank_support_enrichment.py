"""팔꿈치·지지 지속성 보강과 부분 지지 변경 규칙의 개발 결과를 표시한다."""

import html
import pathlib
import shutil

import cv2
import numpy as np

import audit
import render
from plank_partial_item_trial import load_rows
from plank_support_motion import detect


def run() -> pathlib.Path:
    """이전 모델과 같은 개발 검증에서 비교하고 신규 학습 구간을 표시한다.

    Returns:
        pathlib.Path: 결과 HTML 절대 경로. 종료 사건과 오류 자세를 구별하고
        학습 구간의 참여자 제외 예측만 표시한다. 독립 최종 시험이 아니다.

    Raises:
        OSError: 입력·영상·캐시·출력 접근 실패.
        ValueError: 비교 ID·영상 디코딩·관측 계약 불일치.
        RuntimeError: 자원 보호 실패.
    """
    old = audit.read_json(audit.OUT / "plank_partial_item_trial.json")
    new = audit.read_json(audit.OUT / "plank_support_item_trial.json")
    direct = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_support_enrichment_direct.json"
        )["reviews"]
    }
    exports = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_support_enrichment_plan_exports.json"
        )["records"]
    }
    extracted = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "plank_support_extraction.json")[
            "records"
        ]
    }
    motions = []
    for row in load_rows(True):
        result = detect(row["seq"])
        draft = row["grades"][3]
        motions.append(
            dict(
                id=row["id"],
                split=row["split"],
                direct_continuity_grade=draft,
                diagnostic=result,
                approved_continuity_grade_upper_bound=(
                    result["continuity_grade_upper_bound"]
                    if draft is not None
                    else None
                ),
            )
        )
    audit.save_json(
        audit.OUT / "plank_support_motion_results.json",
        dict(
            records=motions,
            independent_final_test=False,
            expert_validated=False,
            normal_grade_never_inferred_from_no_event=True,
            common_rule_modified=False,
        ),
    )
    esc = html.escape
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>팔꿈치·지지 지속성 보강</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px;background:#14212c;color:#edf5fc}a{color:#8cdbff}section{background:#243a4b;padding:18px;margin:18px 0;border-radius:12px}td,th{padding:9px;text-align:left;border-bottom:1px solid #678}table{width:100%;border-collapse:collapse}video,img{width:100%;max-width:900px}.notice{border-left:4px solid #ffd17d;padding:12px}</style><h1>팔꿈치·지지 지속성 보강</h1><p><a href="index.html">5종 비교</a> · <a href="plank_partial_items.html">직전 부분 항목 모델</a> · <a href="plank_support_item_trial.json">학습·참여자 제외·개발 검증 결과</a> · <a href="plank_support_motion_results.json">팔 지지 변경 검출</a> · <a href="plank_support_input_adjustment.json">입력 구간 수정 계보</a></p><section><h2>한 등급만 있던 두 항목 보강</h2><p>기존 학습 출처에서 팔꿈치 기본 위치·전방 이동·운동 종료의 3개 구간을 직접 검토했습니다. 종료 장면은 지지가 끊기는 사건이며 잘못된 자세나 체력 부족으로 분류하지 않습니다. 점수가 없는 항목과 총점은 채우지 않습니다.</p><p class="notice">라벨은 비전문가 AI 직접 검토 초안입니다. 기존에 반복 확인한 두 출처를 학습에 보충한 개발 실험이며 독립 최종 시험이 아닙니다. 기존 공통 모델·규칙과 직전 항목 모델 파일은 유지했습니다.</p><table><tr><th>항목</th><th>직전 등급 수</th><th>보강 후 등급 수</th><th>학습 구간 / 그룹</th><th>2 미만 등급의 그룹 수</th></tr>"""
    ]
    for index in [1, 3]:
        a, b = old["items"][index], new["items"][index]
        parts.append(
            f'<tr><td>{esc(b["criterion_id"])}</td><td>{esc(str(a["training_grade_counts"]))}</td><td>{esc(str(b["training_grade_counts"]))}</td><td>{b["training_records"]} / {b["training_people"]}</td><td>{b["training_nonfull_grade_people"]}</td></tr>'
        )
    parts.append(
        "</table><p>두 항목 모두 정상과 다른 등급은 아직 한 참여자 그룹에만 있습니다. 이 사람을 통째로 제외하면 해당 등급 학습 사례가 없어지므로 일반화를 확인할 수 없습니다. 상수 예측만 하던 학습 상태를 벗어났지만 완성된 오류 판별 모델은 아닙니다.</p></section><section><h2>동일한 개발 검증 영상에서 전후 비교</h2><p>이전부터 학습에서 제외한 Amanda·Ottobre·Climber·Gotti 네 출처를 계속 제외했습니다. 아래는 두 모델 모두 해당 항목을 평가한 같은 구간의 등급 오차입니다. 총점 정확도로 환산하지 않습니다.</p><table><tr><th>항목 / 구간</th><th>직접 초안</th><th>직전 AI</th><th>보강 AI</th><th>규칙</th></tr>"
    )
    paired = []
    for index in [1, 3]:
        before = {r["id"]: r for r in old["items"][index]["validation"]}
        after = {r["id"]: r for r in new["items"][index]["validation"]}
        if set(before) != set(after):
            raise ValueError("Validation cohort changed")
        for identifier, b in after.items():
            a = before[identifier]
            paired.append(
                dict(
                    criterion_id=new["items"][index]["criterion_id"],
                    id=identifier,
                    draft_grade=b["draft_grade"],
                    before_ai=a["ai_grade"],
                    after_ai=b["ai_grade"],
                    rule_grade=b["rule_grade"],
                )
            )
            parts.append(
                f'<tr><td>{esc(new["items"][index]["criterion_id"])} / {esc(identifier)}</td><td>{b["draft_grade"]}</td><td>{a["ai_grade"]:.3f}</td><td>{b["ai_grade"]:.3f}</td><td>{b["rule_grade"]}</td></tr>'
            )
    parts.append(
        "</table></section><section><h2>기존 규칙의 사선 구도 실패와 새 변경 근거</h2><p>화면 가로 거리만 쓰는 기존 팔꿈치 규칙은 Colossus의 기본 위치를 감점하고, 앞쪽 이동 후 위치를 정상으로 판단했습니다. 원근·카메라 방향 때문에 화면 x축이 실제 전후 방향을 뜻하지 않는 실패입니다. 이 사례를 근거로 공통 규칙을 임의 보정하지 않았습니다.</p><p>별도 실험은 어깨 대비 팔꿈치 위치의 처음·끝 변위를 몸통 길이로 정규화합니다. 충분히 보이는 팔이 크게 움직이면 지지 지속성 2등급을 배제하는 양성 근거로 사용합니다. 중단 0등급인지 변경 1등급인지는 별도 판단해야 합니다. 변화가 검출되지 않으면 판정 불가이며 정상 2등급을 주지 않습니다. 카메라 회전·몸통 회전·추적 오류로 오검출할 수 있어 공통 규칙에 채택하지 않았습니다.</p><table><tr><th>구간</th><th>직접 지속성 초안</th><th>변위 / 몸통 길이</th><th>지속성 등급 상한</th></tr>"
    )
    for row in motions:
        if row["id"] in [
            "error_review_colossus_forward_core",
            "ottobre_challenge",
            "support_pannell_stop",
            "climber_prospective",
            "amanda_challenge",
        ]:
            d = row["diagnostic"]
            value = (
                "판정 불가"
                if d["relative_displacement"] is None
                else f'{d["relative_displacement"]:.3f}'
            )
            parts.append(
                f'<tr><td>{esc(row["id"])}</td><td>{row["direct_continuity_grade"]}</td><td>{value}</td><td>{d["continuity_grade_upper_bound"] if d["continuity_grade_upper_bound"] is not None else "근거 없음 / 정상 판정 아님"}</td></tr>'
            )
    parts.append("</table></section>")
    frames = {}
    for identifier, draft in direct.items():
        entry = extracted[identifier]
        folder = audit.OUT / "plank_support_overlays" / identifier
        folder.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(entry["extraction"]["cache"], folder / "side.npz")
        video = render.overlay(
            dict(
                id=identifier,
                exercise="plank",
                video_path=exports[identifier]["video_path"],
            ),
            dict(cache_folder=str(folder), ai_score=None, rule_score=None),
        )
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        count = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if count == 15:
                cv2.imwrite(str(folder / "preview.jpg"), pixels)
            count += 1
        cap.release()
        if count < 5:
            raise ValueError("Overlay decoding failed")
        frames[identifier] = count
        parts.append(
            f'<section><h2>{esc(identifier)}</h2><p>{esc(draft["note_ko"])}</p><p>직접 초안 {esc(str(draft["draft_levels"]))} · 총점 판정 불가</p><table><tr><th>관측 항목</th><th>직접 초안</th><th>참여자 제외 AI</th><th>규칙</th></tr>'
        )
        for index, grade in enumerate(draft["draft_levels"]):
            if grade is None:
                continue
            pred = next(
                (
                    r
                    for r in new["items"][index]["person_heldout"]
                    if r["id"] == identifier
                ),
                None,
            )
            parts.append(
                f'<tr><td>{esc(new["items"][index]["criterion_id"])}</td><td>{grade}</td><td>{format(pred["ai_grade"],".3f") if pred else "판정 불가"}</td><td>{entry["observation"]["items"][index]["rule_grade"]}</td></tr>'
            )
        parts.append(
            f'</table><video controls preload="metadata" poster="{(folder/"preview.jpg").relative_to(audit.OUT)}" src="{video.relative_to(audit.OUT)}"></video><p><a href="{esc(draft["source_url"])}">{esc(draft["creator"])}</a> · CC BY 표기 확인 · 구간 분리·무음·관절 표시 적용</p></section>'
        )
    parts.append(
        "<section><h2>남은 보충</h2><p>다른 참여자의 측면 팔꿈치 위치 이탈과 지지 전환 자료가 더 필요합니다. 오류 등급이 있는 참여자가 학습과 검증 양쪽에 독립적으로 있어야 하며, 같은 원본의 반복 장면을 새 참여자나 독립 시험으로 세지 않습니다.</p></section></html>"
    )
    target = audit.OUT / "plank_support_enrichment.html"
    target.write_text("".join(parts).replace("直接", "직접"), encoding="utf-8")
    audit.save_json(
        audit.OUT / "plank_support_enrichment_comparison.json",
        dict(
            paired_validation=paired,
            decoded_overlay_frames=frames,
            independent_final_test=False,
            local_browser_visual_check=False,
            all_five_completed=False,
        ),
    )
    return target


if __name__ == "__main__":
    print(run())
