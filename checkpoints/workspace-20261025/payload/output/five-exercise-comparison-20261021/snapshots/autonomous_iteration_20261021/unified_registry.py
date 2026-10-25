"""기본 총점 비교와 후속 항목 실험을 실제 사용 자료 기준으로 통합한다.

옛 progress 집계 숫자를 재사용하지 않는다. 학습 모델의 존재와 독립
검증 완료를 구별하고 부족한 자료·관측 항목·미실시 실험을 보존한다.
"""

import datetime
import html
import pathlib

import audit


def run() -> dict:
    """원본 보고서·모델 해시와 검증 범위를 연결한 통합 현황을 저장한다.

    Returns:
        dict: 운동별 기본 비교·후속 실험·현재 라벨 수·완료 조건.
        partial 실험을 총점 비교로 합산하지 않는다. 전문가·독립 시험
        건수는 근거 파일에서 확인되는 완료 건만 반영한다.

    Raises:
        OSError: 필수 보고서·모델·출력 접근 실패.
        ValueError: 등록한 모델 파일이 없거나 비교 운동이 누락된 경우.
    """
    comparison = audit.read_json(audit.OUT / "comparison.json")
    items = []
    for exercise in audit.VIEWS:
        row = comparison["results"][exercise]
        model = audit.OUT / (exercise + "_experimental.joblib")
        if not model.exists():
            raise ValueError("Missing baseline model: " + exercise)
        items.append(
            dict(
                exercise=exercise,
                baseline_training_records=row["train_records"],
                baseline_training_person_groups=row["train_people"],
                baseline_validation_records=row["validation_records"],
                baseline_metrics=row.get("metrics"),
                baseline_model_path=str(model),
                baseline_model_sha256=audit.digest(model),
                scope="development_nonexpert_drafts_not_final_accuracy",
                latest_artifacts=[],
            )
        )
    by_name = {r["exercise"]: r for r in items}
    for exercise, names in [
        (
            "plank",
            [
                "plank_balance_trial.json",
                "plank_balance_challenge_results.json",
                "plank_prospective_results.json",
                "plank_partial_item_trial.json",
                "plank_support_item_trial.json",
                "upper_support_event_results.json",
                "upper_event_nested_results.json",
                "autonomous_prospective_results.json",
                "autonomous_failure_analysis.json",
            ],
        ),
        (
            "ytw",
            [
                "w_temporal_trial_results.json",
                "w_challenge_v2_results.json",
                "ytw_phase_comparison.json",
                "ytw_side_component_results.json",
            ],
        ),
    ]:
        for name in names:
            path = audit.OUT / name
            if path.exists():
                by_name[exercise]["latest_artifacts"].append(
                    dict(path=str(path), sha256=audit.digest(path))
                )
    gates = dict(
        all_five_full_paired_development_comparisons=all(
            (r["baseline_metrics"] or {}).get("paired_samples", 0) > 0
            for r in items
        ),
        expert_reference_validated=False,
        independent_all_five_final_test_completed=False,
        real_recapture_consistency_measured=False,
        controlled_end_to_end_runtime_measured=(
            audit.read_json(audit.OUT / "autonomous_runtime_results.json").get(
                "completed", False
            )
            if (audit.OUT / "autonomous_runtime_results.json").exists()
            else False
        ),
    )
    report = dict(
        updated_at=datetime.datetime.now().astimezone().isoformat(),
        records=len(audit.read_json(audit.OUT / "labels.json")),
        supplemental_records=len(
            audit.read_json(audit.OUT / "supplemental_labels.json")
        ),
        items=items,
        completion_gates=gates,
        all_five_completed=all(gates.values()),
        base_comparison_sha256=audit.digest(audit.OUT / "comparison.json"),
        raw_source_contract_hashes={
            name: audit.digest(audit.SOURCE / name)
            for name in [
                "labels.json",
                "summary.json",
                "rubrics.json",
                "verification.json",
                "qevd_inventory.json",
                "usage_and_input_contract.json",
            ]
        },
        note_ko="기본 comparison은 이전 총점 실험이다. 후속 플랭크 항목 비교는 별도 링크로 구별하며 기본 검증 0을 후속 검증이 전혀 없다는 뜻으로 사용하지 않는다.",
    )
    for name in [
        "autonomous_source_reservation.json",
        "autonomous_prospective_results.json",
        "autonomous_runtime_results.json",
        "ytw_side_component_results.json",
    ]:
        if (audit.OUT / name).exists():
            report[name] = dict(
                path=str(audit.OUT / name),
                sha256=audit.digest(audit.OUT / name),
            )
    audit.save_json(audit.OUT / "unified_registry.json", report)
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5종 최신 통합 현황</title><style>body{font:16px/1.65 system-ui;max-width:1150px;margin:auto;padding:24px;background:#14212c;color:#eef5ff}a{color:#8cdbff}section{padding:18px;margin:20px 0;background:#243a4b;border-radius:12px}td,th{padding:9px;border-bottom:1px solid #678;text-align:left}</style><h1>5종 최신 통합 현황</h1><p><a href="index.html">기본 비교</a> · <a href="plank_partial_items.html">플랭크 부분 항목</a> · <a href="plank_support_enrichment.html">팔꿈치·지지 보강</a> · <a href="unified_registry.json">모델·보고서 해시와 완료 조건</a></p><p>MediaPipe Heavy는 관절 추출기이며, AI 점수 모델은 별도 ExtraTrees입니다. 아래 수치는 비전문가 직접 검토 초안과의 개발 비교입니다. 모델 파일의 존재를 정확도 검증 완료로 표현하지 않습니다.</p><section><h2>기본 총점 비교</h2><table><tr><th>운동</th><th>학습 구간 / 참여자 그룹</th><th>두 방식 공통 개발 비교 수</th><th>AI / 규칙 평균 점수 오차</th></tr>'
    ]
    for row in items:
        m = row["baseline_metrics"]
        a = m.get("ai_MAE_points")
        b = m.get("rule_MAE_points")
        parts.append(
            f'<tr><td>{esc(row["exercise"])}</td><td>{row["baseline_training_records"]} / {row["baseline_training_person_groups"]}</td><td>{m.get("paired_samples",0)}</td><td>{f"{a:.2f} / {b:.2f}" if a is not None and b is not None else "총점 공통 비교 미완료"}</td></tr>'
        )
    parts.append(
        "</table><p>플랭크의 후속 개발 검증과 항목 모델은 위 기본 표에 합산하지 않았습니다. Y–T–W는 Y·전경 관측 계약과 전체 총점 규칙을 더 검증해야 합니다. 라벨 전체 건수를 학습 가능 건수로 세지 않습니다.</p></section><section><h2>후속 실험 근거</h2>"
    )
    prospective = audit.OUT / "autonomous_prospective_results.json"
    if prospective.exists():
        trial = audit.read_json(prospective)
        parts.append(
            f"<p>사전 동결 플랭크 비교: 새 출처 "
            f'{trial["new_source_groups"]}그룹 중 총점 공통 평가 '
            f'{trial["paired_full_records"]}구간 / '
            f'{trial["prospective_full_source_groups"]}원본·참여자 그룹. '
            f'AI 평균 오차 {trial["full_ai_MAE_points"]:.3f}점, '
            f'규칙 {trial["full_rule_MAE_points"]:.3f}점. '
            "동일 인물의 두 구간이며 전문가 정답·5종 최종 정확도가 아닙니다. "
            "이 출처는 재학습에 사용하지 않았습니다.</p>"
        )
    event = audit.OUT / "upper_event_nested_results.json"
    if event.exists():
        trial = audit.read_json(event)
        m = trial["metrics"]
        parts.append(
            f'<p>상지 변경 사건 별도 모델: {trial["training_records"]}구간 / '
            f'{trial["person_groups"]}참여자 그룹. 중첩 참여자 제외 개발 진단 '
            f'TP {m["TP"]}, FN {m["FN"]}, FP {m["FP"]}, TN {m["TN"]}. '
            "새 QEVD 참여자 3그룹을 포함합니다. 사건 확률은 자세 점수가 "
            "아니며 오탐 때문에 공통 규칙·모델에 채택하지 않았습니다.</p>"
        )
    for row in items:
        if row["latest_artifacts"]:
            parts.append("<h3>" + esc(row["exercise"]) + "</h3><ul>")
            for a in row["latest_artifacts"]:
                p = pathlib.Path(a["path"])
                parts.append(
                    f'<li><a href="{p.relative_to(audit.OUT)}">{esc(p.name)}</a></li>'
                )
            parts.append("</ul>")
    for name in [
        "autonomous_prospective.html",
        "upper_support_event.html",
        "upper_event_nested.html",
        "upper_event_review.html",
        "autonomous_runtime.html",
        "ytw_side_components.html",
    ]:
        if (audit.OUT / name).exists():
            parts.append(f'<p><a href="{name}">{esc(name)}</a></p>')
    parts.append("</section><section><h2>미충족 완료 조건</h2><ul>")
    for name, passed in gates.items():
        parts.append(f'<li>{esc(name)}: {"충족" if passed else "미충족"}</li>')
    parts.append(
        "</ul><p>독립 최종 시험은 새 출처·참여자 그룹을 사전에 예약하고, 해당 자료를 모델 선택·재학습에 사용하지 않은 경우에만 별도로 표시합니다. 반복 처리 일관성과 실제 재촬영 일관성은 구별합니다.</p></section></html>"
    )
    (audit.OUT / "unified_status.html").write_text(
        "".join(parts), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(run()["completion_gates"])
