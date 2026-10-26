"""측면 오류 보충·첫 플랭크 점수 모델·참여자 제외 개발 비교를 표시한다."""

import html
import pathlib
import shutil

import cv2
import numpy as np

import audit
import render


def run() -> pathlib.Path:
    """동일 10개 개발 구간의 참여자 제외 AI와 고정 규칙을 비교한다.

    Returns:
        pathlib.Path: 최신 HTML 절대 경로. 새 3개 관절 MP4에 표시하는 AI는
        해당 참여자 전체를 학습에서 제외한 fold 예측이다. 독립 최종 시험이나
        전문가 정확도가 아니며 학습된 최종 모델의 재대입 점수도 아니다.

    Raises:
        OSError: 캐시·보고서·영상 접근 실패.
        ValueError: 참여자 제외 예측 누락 또는 관절 영상 디코딩 실패.
        RuntimeError: 영상 생성 자원 보호 중단.
    """
    result = audit.read_json(audit.OUT / "comparison.json")["results"]["plank"]
    diagnostic = result["training_participant_diagnostic"]
    predictions = {
        p["id"]: dict(p, held_person=f["held_person"])
        for f in diagnostic["folds"]
        for p in f["predictions"]
    }
    rows = []
    for r in result["rule_only_previews"]:
        pred = predictions[r["id"]]
        rows.append(
            dict(
                id=r["id"],
                label_score=r["label_score"],
                ai_oof_score=pred["predicted_score"],
                rule_score=r["rule_score"],
                ai_absolute_error=abs(
                    pred["predicted_score"] - r["label_score"]
                ),
                rule_absolute_error=(
                    abs(r["rule_score"] - r["label_score"])
                    if r["rule_score"] is not None
                    else None
                ),
                held_person=pred["held_person"],
            )
        )
    summary = {}
    for name, subset in [
        ("all", rows),
        ("normal_draft", [r for r in rows if r["label_score"] == 100]),
        ("error_draft", [r for r in rows if r["label_score"] < 100]),
    ]:
        paired = [r for r in subset if r["rule_absolute_error"] is not None]
        summary[name] = dict(
            samples=len(paired),
            ai_MAE=float(np.mean([r["ai_absolute_error"] for r in paired])),
            rule_MAE=float(np.mean([r["rule_absolute_error"] for r in paired])),
        )
    summary["normal_error_equal_weight"] = dict(
        ai_MAE=float(
            np.mean(
                [summary[k]["ai_MAE"] for k in ["normal_draft", "error_draft"]]
            )
        ),
        rule_MAE=float(
            np.mean(
                [
                    summary[k]["rule_MAE"]
                    for k in ["normal_draft", "error_draft"]
                ]
            )
        ),
    )
    report = dict(
        records=rows,
        summary=summary,
        method="leave_one_training_participant_out",
        independent_final_test=False,
        expert_validated=False,
        source="comparison.json",
        all_five_validated=False,
    )
    audit.save_json(audit.OUT / "plank_person_oof_comparison.json", report)
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>플랭크 측면 보충·AI 개발 비교</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px;background:#14212c;color:#edf5fc}a{color:#8addff}section{padding:18px;margin:20px 0;background:#243a4b;border-radius:12px}td,th{padding:8px;border-bottom:1px solid #678;text-align:left}video,img{width:100%;max-width:800px}.notice{border-left:4px solid #ffd17d;padding:12px}</style><h1>플랭크 측면 보충·AI 개발 비교</h1><p><a href="index.html">5종 비교</a> · <a href="plank_error_review.html">이전 오류 검토</a> · <a href="plank_side_direct_labels.json">규칙 확인 전 동결 초안</a> · <a href="plank_person_oof_comparison.json">참여자 제외 비교</a> · <a href="plank_side_joint_results.json">공통 관측 검사</a></p>'
    ]
    parts.append(
        f'<section><h2>현재 상태</h2><p>새 3초 측면 구간 3개가 공통 관절 관측 조건을 통과했습니다. 원본 2개·참여자 그룹 2개이며 정상 2개와 몸통·골반 처짐/추가 지지 오류 1개입니다. 같은 원본의 정상·오류는 전부 같은 train 그룹입니다.</p><p>학습 자료 {result["train_records"]}개·{result["train_people"]}명으로 별도 ExtraTrees 플랭크 점수 모델의 첫 개발 학습을 실행했습니다. 100점 초안 8개·40점 초안 2개뿐이며 오류 참여자도 2명입니다. 검증용 분할 0개·독립 최종 시험 0개·전문가 정답 0개입니다.</p><p class="notice">아래 AI는 해당 사람의 모든 자료를 제외하고 다시 학습한 개발 예측입니다. 기존에 본 자료로 진행한 진단이며 최종 정확도가 아닙니다. 전체 평균과 오류 사례 성능을 분리해 확인해야 합니다.</p><table><tr><th>범위</th><th>수</th><th>AI MAE(점)</th><th>규칙 MAE(점)</th></tr>'
    )
    for name, r in summary.items():
        parts.append(
            f'<tr><td>{esc(name)}</td><td>{r.get("samples","정상·오류 동일 비중")}</td><td>{r["ai_MAE"]:.2f}</td><td>{r["rule_MAE"]:.2f}</td></tr>'
        )
    parts.append(
        "</table><p>오류 두 사례에서 AI 오차가 크고, 정상·오류에 같은 비중을 주면 규칙보다 불리합니다. 단순히 전체 MAE가 작다는 이유로 AI 우위를 결론 내리지 않습니다.</p></section>"
    )
    exports = audit.read_json(
        audit.OUT / "plank_side_export_plan_exports.json"
    )["records"]
    manifests = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "development_manifest.json")
    }
    drafts = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "plank_side_direct_labels.json")[
            "reviews"
        ]
    }
    for export in exports:
        identifier = export["id"]
        row = next(r for r in rows if r["id"] == identifier)
        original = next(
            r for r in result["rule_only_previews"] if r["id"] == identifier
        )
        folder = audit.OUT / "plank_side_oof_overlays" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            audit.OUT / "heavy" / identifier / "side.npz", folder / "side.npz"
        )
        comparison = dict(
            original, cache_folder=str(folder), ai_score=row["ai_oof_score"]
        )
        video = render.overlay(manifests[identifier], comparison)
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, int(cap.get(7)) // 2)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Overlay unavailable")
        thumbnail = folder / "preview.jpg"
        if not cv2.imwrite(str(thumbnail), pixels):
            raise OSError("Preview save failed")
        parts.append(
            f'<section><h2>{esc(identifier)}</h2><p>직접 초안 {row["label_score"]:.1f} · AI 참여자 제외 {row["ai_oof_score"]:.2f} · 규칙 {row["rule_score"]:.1f}</p><p>{esc(drafts[identifier]["note_ko"])}</p><video preload="none" controls poster="{thumbnail.relative_to(audit.OUT).as_posix()}" src="{video.relative_to(audit.OUT).as_posix()}"></video><p>제작자 {esc(export["creator"])} · <a href="{esc(export["source_url"])}">원본 YouTube</a> · <a href="https://www.youtube.com/t/creative_commons">CC Attribution 재사용 표기</a><br>원본 {export["start_frame"]/export["fps"]:.3f}~{export["end_frame_exclusive"]/export["fps"]:.3f}초 · 고정 크롭 {export["crop_xyxy"]} · 무음·관절 표시 파생본</p></section>'
        )
    parts.append(
        '<section><h2>불일치 원인 분석</h2><p>Pannell 후보의 몸통 각도 하위 10%는 약 152.2도, 중앙값은 161.8도입니다. 4픽셀 좌표 변동이나 앞선 1·3초 문맥 추가로 기존 0등급이 바뀌지 않았습니다. 초기 추적만으로 불일치를 설명하기 어렵습니다. 원본 크기 세 장 재검토에서는 골반 높이 변화와 의복·관절 중심 구별의 불확실성을 확인했습니다. 원래 초안은 보존하고 학습 정답을 규칙 0등급으로 바꾸지 않았습니다.</p><p>새 Jonple 오류의 골반 추가 지지는 원본에서 보이지만 무릎 각도만 쓰는 현재 규칙은 이를 놓쳤습니다. 새 정상 시범도 투영 각도 규칙과 다릅니다. 접촉·체형·촬영 투영을 구분할 자료와 항목 규칙 검증이 더 필요합니다.</p><a href="plank_alignment_diagnostic.json">양쪽 각도·4픽셀 진단</a> · <a href="plank_context_trial.json">동일 프레임 문맥 실험</a> · <a href="plank_pannell_reinspection.json">규칙 노출 후 재검토 기록</a></section></html>'
    )
    target = audit.OUT / "plank_side_progress.html"
    target.write_text("".join(parts), encoding="utf-8")
    for name in [
        "index.html",
        "youtube_review.html",
        "plank_error_review.html",
    ]:
        path = audit.OUT / name
        page = path.read_text(encoding="utf-8")
        if 'href="plank_side_progress.html"' not in page:
            page = page.replace(
                "</h1>",
                '</h1><p class="notice">이전 실험 수치는 당시 기록입니다. <a href="plank_side_progress.html">최신 플랭크 자료·첫 AI 개발 비교</a>를 확인하세요.</p>',
                1,
            )
            path.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
