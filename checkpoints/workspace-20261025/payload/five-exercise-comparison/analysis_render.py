"""W 원인 진단과 새 자료 선별 근거를 기존 로컬 웹에 연결한다."""

import html

import audit


def run() -> None:
    """원본·수동 위치·Heavy 진단을 나란히 표시하는 HTML을 저장한다.

    Returns:
        None: w_analysis.html 파일을 출력 폴더에 저장한다.

    Raises:
        OSError: 진단 JSON 읽기 또는 HTML 저장 실패.
        KeyError: 필수 진단 항목이 없는 경우.
    """
    report = audit.read_json(audit.OUT / "ytw_w_root_cause.json")
    escape = html.escape
    parts = [
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W 원인 분석·자료 보충</title><style>body{background:#14212c;color:#edf4fa;font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px}section{background:#213544;padding:20px;margin:20px 0;border-radius:12px}a{color:#80dcff}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #567}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:650px){.pair{display:block}}</style><h1>W 규칙 불일치 원인 분석</h1><p><a href="index.html">점수 비교</a> · <a href="w_inspection.html">W 추가 12개 관절·구간 검토</a> · <a href="phases.html">단계 구간</a> · <a href="ytw_w_root_cause.json">수치·수동 점 원본</a> · <a href="snapshots/pre_w_contract_fix/comparison.json">변경 전 비교 보존본</a></p><section><h2>확인된 계약 불일치</h2><p>라벨의 W 기준은 양 팔꿈치 굽힘과 손을 위에 두는 W 모양입니다. 화면 각도 90±20도 조건은 기준에 없었습니다. 90도 대리값을 폐기하고 양팔 굽힘·손의 위쪽 위치·팔꿈치의 과도한 상승·좌우 차이로 0·1·2등급을 계산하는 개발 규칙을 구현했습니다. 이전 실패 3개와 수치는 보존합니다.</p><p>규칙 경계는 전문가 검증 전 공학 초안입니다. 미관측은 보류하지만 관측된 W 미충족은 0등급으로 채점합니다. Y·전경 관측 문제는 별개이며 총점은 여전히 산출하지 않습니다. 라벨 초안은 규칙에 맞춰 고치지 않았습니다.</p><h2>같은 투영, 다른 실제 각도</h2><p>직교 투영에서 같은 화면 각도 150도를 만드는 두 3D 벡터쌍이 실제로는 각각 150도와 90도일 수 있음을 수학적으로 검사했습니다. 이는 개별 영상의 실제 각도를 복원했다는 뜻이 아닙니다. Heavy의 단일 영상 추정 3D도 독립 깊이 정답으로 간주하지 않습니다.</p><h2>원본 장면 재검토의 한계</h2><p>아래 6개는 이미 등급과 규칙 결과에 노출된 개발 자료입니다. 수동 점은 원본 PNG만 띄운 뒤 기록한 비전문가의 대략적인 위치입니다. 노란색은 수동 점, 파란색은 Heavy입니다. 수동 점의 ±8픽셀 상자 꼭짓점 민감도도 JSON에 남겼습니다. 해부학적 정답·독립 눈가림 검토가 아닙니다.</p></section>"""
    ]
    challenge = audit.read_json(audit.OUT / "w_challenge_results.json")
    summary = challenge["summary"]
    negative = audit.read_json(audit.OUT / "w_prior_negative_results.json")
    parts.append(
        "<section><h2>실제 W 규칙 개발 검증</h2>"
        '<p><a href="w_challenge_results.json">12개 개발 후보 결과</a> · '
        '<a href="w_challenge_direct_review.json">시간순 직접 검토 초안</a> · '
        '<a href="w_prior_negative_results.json">기존 0등급 실패 재검사</a> · '
        '<a href="snapshots/w_morphology_v1_1/w_challenge_results.json">v1.1 보존본</a></p>'
        "<p>12개 영상을 각 36개 시간순 장면으로 직접 검토하여 W 항목 9개 초안과 판정 불가 3개를 기록했습니다. "
        "총점 라벨을 추가한 것은 아닙니다. 이 후보들의 W 초안에는 0등급이 없습니다. "
        "v1.1은 새 초안을 보기 전에 고정했고, v2는 별도의 기존 음성 사례 실패를 본 뒤 개선했습니다. "
        "현재 수치는 재사용한 개발 자료의 진단이며 독립 시험 정확도가 아닙니다.</p>"
        f'<p>동일 시도 핵심 구간과 자동 구간이 겹친 비교 {summary["same_attempt_pairs"]}개, '
        f'W 등급 평균 절대 오차 {summary["same_attempt_MAE_grade"]:.2f}/2, '
        f'완전 일치 {summary["same_attempt_exact_matches"]}개. '
        f'전체 클립 기준 평가 가능 {summary["automatic_assessed_known"]}/{summary["directly_graded"]}개는 '
        "다른 반복 시도를 선택한 경우도 포함하므로 동일 시도 정확도로 쓰지 않습니다.</p>"
        f'<p>기존 0등급 자료의 v1.1 오판 2 → v2 {negative["automatic"]["grade"]}. '
        "팔꿈치를 머리 쪽으로 올린 굽힌 Y를 W로 오인한 문제를 보강했습니다. "
        "수정에 사용한 사례이므로 독립 오류 검출 성능으로 세지 않습니다.</p>"
        '<img src="w_review/prior_negative_same_frame.png" alt="같은 프레임의 원본과 관절 진단">'
        "<table><tr><th>자료</th><th>직접 초안</th><th>자동 규칙</th><th>동일 시도 겹침</th></tr>"
    )
    for item in challenge["results"]:
        parts.append(
            f'<tr><td>{escape(item["id"])}</td><td>{item["W_grade_draft"]}</td>'
            f'<td>{item["automatic"]["grade"]}</td><td>{item["same_attempt_core_overlaps"]}</td></tr>'
        )
    parts.append(
        "</table><p>None은 관측·구간 불충분입니다. 0점으로 채우지 않습니다.</p></section>"
    )
    second = audit.read_json(audit.OUT / "w_challenge_v2_results.json")
    stats = second["summary"]
    parts.append(
        "<section><h2>추가 12개·단계 인식 보강</h2>"
        '<p><a href="w_challenge_v2_results.json">현재 추가 개발 결과</a> · '
        '<a href="snapshots/phase_v2_W_v2/w_challenge_v2_results.json">단계 보강 전 보존본</a> · '
        '<a href="candidate_identity_review.json">같은 참여자 의심 그룹</a></p>'
        "<p>T 진입은 팔 폭 상위 기준의 85%, T 종료는 70%로 구분해 T 내부의 작은 폭 변화를 W로 오인하는 문제를 보강했습니다. "
        "관절 누락은 연결하지 않습니다. 첫 12개에서는 같은 시도 비교가 5→4개로 줄었고, 추가 12개에서는 6→8개로 늘었습니다. "
        "한 묶음의 개선을 전체 구간 인식 개선으로 일반화하지 않습니다. 두 묶음 모두 재사용한 개발 진단입니다.</p>"
        f'<p>추가 12개: W 직접 초안 9개, 판정 불가 3개. 같은 시도 비교 {stats["same_attempt_pairs"]}개, '
        f'평균 오차 {stats["same_attempt_MAE_grade"]:.2f}/2, 완전 일치 {stats["same_attempt_exact_matches"]}개. '
        f'직접 판정 불가 3개에도 자동 등급 {stats["automatic_grade_on_human_unknown"]}개가 나왔습니다. '
        "어두움·흐림에서 높은 관절 신뢰도가 관측 가능성을 보장하지 못한 실패이며 성공률에 넣지 않습니다.</p>"
        "<p>신규 초안 18개에는 0등급이 없습니다. 기존 0등급 한 사례는 수정에 사용했으므로 독립 오류 검증은 부족합니다. "
        "참여자 ID가 달라도 같은 사람으로 의심되는 2개 묶음은 같은 분할에 묶어야 하며 새 사람 수는 미확정입니다. "
        "아직 공통 점수 학습·최종 시험으로 편입하지 않았습니다.</p>"
        '<p><a href="w_policy_sensitivity.json">수치 경계 민감도</a>: 자동 구간 18개를 고정한 8가지 단일 경계 변화 중 6개 설정은 등급 변화가 없고, 2개 설정에서 각각 한 사례가 변했습니다. 라벨에 맞춰 경계를 최적화하지 않았으며 실제 재촬영 일관성을 뜻하지 않습니다.</p><h3>현재 규칙의 수치 경계</h3>'
        "<p>화면 굽힘은 180도에서 투영 팔꿈치 각도를 뺀 값입니다. 20도 이상은 굽힘 충족, 8도 이상은 부분 수행입니다. "
        "손 높이는 몸통 방향으로 팔꿈치보다 몸통 길이 5% 이상 위이면 충족, -5%까지 부분입니다. "
        "팔꿈치의 위쪽 성분이 상완 길이의 25% 이하면 충족, 50%까지 부분, 그 이상은 굽힌 Y 혼동을 막기 위한 미충족입니다. "
        "좌우 굽힘 차이 35도 초과는 부분 수행이며 한쪽만 충족해도 부분 등급입니다. "
        "양팔 관측 80% 이상·3표본 이상에서만 계산합니다. 이 수치는 전문가가 보정한 해부학적 경계가 아닙니다.</p></section>"
    )
    targeted = audit.OUT / "w_targeted_results.json"
    if targeted.exists():
        diagnostic = audit.read_json(targeted)
        parts.append(
            '<section id="targeted-errors"><h2>오류 유형 추가 12개·라벨 구간 재감사</h2>'
            "<p>36장 시간순 표본과 원본 크기 장면을 직접 검토했습니다. 최초 0등급 4개 중 2개는 W 뒤 휴식 구간 오표시였고 실제 W는 부분·충족 등급으로 수정했습니다. "
            "수정 전 기록과 최초 비교를 보존했습니다. 수정본은 규칙 결과 노출 뒤 재검토한 개발 자료로서 독립 시험이 아닙니다.</p>"
            '<p><a href="w_targeted_direct_review.json">현재 직접 초안·수정 이유</a> · '
            '<a href="snapshots/w_targeted_initial_annotation/w_targeted_direct_review.json">최초 초안 보존본</a> · '
            '<a href="w_targeted_results.json">고정 규칙 비교</a> · '
            '<a href="phase_low_arm_trial.json">낮은 팔 단계 확장 실험</a></p>'
            "<p>팔 높이 제한을 몸통 길이 0.6에서 1.1로 확장한 별도 실험은 기존 두 묶음의 같은 시도 비교를 4→5, 8→9개로 늘렸지만 등급 오차도 0.25→0.40, 0.25→0.44로 커졌습니다. "
            "공통 단계 코드에 채택하지 않았으며 W 보류를 0점으로 바꾸지 않았습니다.</p>"
            "<table><tr><th>후보</th><th>직접 초안</th><th>자동 구간 W</th><th>수동 구간 진단 W</th></tr>"
        )
        for row in diagnostic["results"]:
            manual = row["manual_window_diagnostic"]
            parts.append(
                f'<tr><td>{escape(row["id"])}</td><td>{row["W_grade_draft"]}</td><td>{row["automatic"]["grade"]}</td><td>{manual["grade"] if manual else None}</td></tr>'
            )
        parts.append(
            "</table><p>수동 구간 진단은 자동 평가 성공률에 포함하지 않습니다. 36개 플랭크 오류 유형 후보도 5장씩 직접 구도 선별했지만 정면 상체 중심이어서 새 측면 총점은 0개입니다. "
            '<a href="targeted_plank_direct_decisions.json">구도 제외 기록</a>. 신규 W 후보에도 기존과 같은 참여자로 의심되는 영상이 있으며 새 참여자 수는 확정하지 않았습니다.</p></section>'
        )
    item_audit = audit.OUT / "item_error_audit.json"
    if item_audit.exists():
        audit_rows = audit.read_json(item_audit)["records"]
        example = next(
            (r for r in audit_rows if r["id"] == "dvids_navy773312_2997_3087"),
            None,
        )
        if example:
            rule = example["Rule"]
            parts.append(
                '<section id="item-errors"><h2>총점 오차의 항목 간 상쇄</h2>'
                '<p><a href="item_error_audit.json">AI·규칙 항목별 오차 감사 원본</a> · '
                '<a href="w_inspection.html">W 추가 12개 관절·시간축 검토</a></p>'
                f'<p>새 플랭크 실패 상태는 직접 초안 40점·규칙 37.5점으로 총점 오차 {rule["absolute_total_error_points"]:.1f}점입니다. '
                f'그러나 항목별 배점 절대 오차의 합은 {rule["paired_item_absolute_error_sum_points"]:.1f}점이며, 반대 방향 오차가 {rule["cancellation_points"]:.1f}점 상쇄됐습니다. '
                "총점이 가깝다는 이유로 각 판정 근거까지 맞았다고 해석하지 않습니다. 초안은 전문가 정답이 아닙니다.</p>"
                "<table><tr><th>항목</th><th>직접 초안 등급</th><th>규칙 등급</th><th>규칙−초안 배점 차이</th></tr>"
            )
            for name, draft, predicted, difference in zip(
                example["criterion_ids"],
                example["draft_levels"],
                rule["predicted_levels"],
                rule["signed_item_error_points"],
                strict=True,
            ):
                parts.append(
                    f"<tr><td>{escape(name)}</td><td>{draft}</td><td>{predicted}</td><td>{difference:+.1f}점</td></tr>"
                )
            parts.append(
                "</table><p>이 사례는 학습 후보의 규칙 미리보기이며 검증 정확도로 합산하지 않습니다. 미관측 항목이 있으면 총점 오차·상쇄량은 계산하지 않고, 비교 가능한 항목과 배점만 기록합니다.</p></section>"
            )
    trial_path = audit.OUT / "w_trial_audit.json"
    if trial_path.exists():
        trial = audit.read_json(trial_path)
        parts.append(
            '<section id="input-trials"><h2>W 입력 크롭·굽힘 유지 실험</h2>'
            '<p><a href="w_trial_audit.json">같은 사례 대응 비교·강화된 구간 진단</a> · '
            '<a href="phase_bend_dwell_trial.json">굽힘·유지 구간 실험</a></p>'
            "<p>전체 클립의 관절 범위로 손실 없는 크롭을 만들고 같은 Heavy VIDEO 설정으로 재추출했습니다. "
            "점수와 직접 핵심 구간은 크롭 선택에 사용하지 않았습니다. 이미 노출된 개발 자료이며 독립 시험이 아닙니다.</p>"
            "<table><tr><th>입력</th><th>초안 9개 중 평가 가능</th><th>양쪽 모두 평가한 사례</th><th>대응 등급 MAE 원본→크롭</th></tr>"
        )
        for name, label in (
            ("upper_body", "상체"),
            ("head_to_knee", "머리~무릎"),
        ):
            r = trial["roi"][name]
            parts.append(
                f'<tr><td>{label}</td><td>{r["baseline"]["assessed"]}→{r["roi"]["assessed"]}/9</td>'
                f'<td>{r["paired_count"]}</td><td>{r["baseline"]["paired_MAE_grade"]:.2f}→{r["roi"]["paired_MAE_grade"]:.2f}/2</td></tr>'
            )
        parts.append(
            "</table><p>크롭은 한 사례의 초안과 일치도가 나아졌지만 관측 탈락도 생겼습니다. 자동 W 구간 검출은 개선되지 않았으며 공통 입력에 채택하지 않았습니다. "
            "아래는 동일 프레임·원본 좌표로 되돌린 관절입니다. 두 번째 영상에서는 크롭 후 몸통·팔 위치 추정이 불안정해졌습니다. "
            "이 장면만으로 관절 오차의 수치 정답을 정하지 않습니다.</p>"
            '<img src="w_roi_context/qevd_00239387_same_frame.png" alt="같은 원본 프레임의 원본·상체·머리 무릎 크롭 관절">'
            '<img src="w_roi_context/qevd_00101824_same_frame.png" alt="크롭 후 관측이 탈락한 같은 프레임 비교">'
            "<p>단순 겹침 외에 직접 핵심 구간의 중앙점과 50% 이상을 포함하는지 검사했습니다. "
            "굽힘·유지 실험은 오류 후보 9개에서 이 조건과 등급 계산을 충족한 사례가 1→3개지만 등급 MAE는 1.0/2로 큽니다. "
            "기존 추가 묶음에서는 비교 8→9개와 함께 오차가 0.25→0.33으로 증가했습니다. 공통 단계 규칙은 유지합니다.</p></section>"
        )
    if (audit.OUT / "aihub_documents/raw_decode_and_sheets.json").exists():
        import aihub_review_render

        parts.append(aihub_review_render.section())
    for row in report["cases"]:
        identifier = row["id"]
        raw = next((audit.OUT / "w_review").glob(identifier + "*_raw.png"))
        overlay = audit.OUT / "w_review" / (identifier + "_manual_heavy.png")
        parts.append(
            f'<section><h2>{escape(identifier)}</h2><p>원본 f{row["direct_raw_frame"]}, Heavy f{row["nearest_heavy_frame"]}. 프레임이 다르면 동작 변화도 차이의 원인일 수 있습니다.</p><div class="pair"><img src="{raw.relative_to(audit.OUT)}" alt="원본 W 표본"><img src="{overlay.relative_to(audit.OUT)}" alt="수동과 Heavy 대략 위치"></div><table><tr><th>팔</th><th>수동 화면 각도</th><th>Heavy 화면 각도</th></tr>'
        )
        for side in row["sides"]:
            heavy = side["heavy_projected_deg"]
            parts.append(
                f'<tr><td>{escape(side["side"])}</td><td>{side["manual_projected_deg"]:.1f}°</td><td>{heavy:.1f}°</td></tr>'
                if heavy is not None
                else "<tr><td>관측 불가</td></tr>"
            )
        parts.append(
            "</table><p>수동 위치와 Heavy 차이만으로 관절 추출 오류의 크기나 원인을 확정하지 않습니다. 전완이 짧게 투영되거나 가려진 경우 수동 위치 자체가 불확실합니다.</p></section>"
        )
    procurement = audit.OUT / "targeted_acquisition_review.json"
    if procurement.exists():
        record = audit.read_json(procurement)
        parts.append(
            '<section><h2>오류 사례·촬영 구도 보충</h2><p><a href="targeted_acquisition_review.json">직접 선별 결정과 출처</a></p>'
        )
        for row in record["records"]:
            parts.append(
                f'<p><b>{escape(row["id"])}</b> · {escape(row["decision_ko"])}</p>'
            )
            if row.get("url"):
                parts.append(
                    f'<p><a href="{escape(row["url"])}">원 출처·이용 조건 확인</a></p>'
                )
            if row.get("evidence"):
                parts.append(
                    f'<details><summary>직접 확인한 표본</summary><img src="{escape(row["evidence"])}" alt="시간순 구도 선별판"></details>'
                )
        parts.append("</section>")
    parts.append("</html>")
    (audit.OUT / "w_analysis.html").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    run()
