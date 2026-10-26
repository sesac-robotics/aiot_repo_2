"""플랭크 균형 학습·규칙 개선안·신규 출처 개발 확인을 웹에 표시한다."""

import html
import pathlib
import shutil

import cv2

import audit
import render


def run() -> pathlib.Path:
    """세 AI·두 규칙과 신규 2개 구간의 항목·관절 영상을 저장한다.

    Returns:
        pathlib.Path: 새 HTML 절대 경로. 기존 10개 자료의 참여자 제외
        진단과 새 출처의 개발 확인을 구분하며 공통 모델을 교체하지 않는다.

    Raises:
        OSError: 파일·캐시·영상 접근 실패.
        ValueError: 영상 디코딩 또는 입력 구간 불일치.
        RuntimeError: 자원 보호 중단.
    """
    trial = audit.read_json(audit.OUT / "plank_balance_trial.json")
    challenge = audit.read_json(
        audit.OUT / "plank_balance_challenge_results.json"
    )
    direct = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_balance_challenge_direct.json"
        )["reviews"]
    }
    exports = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "plank_balance_challenge_plan_exports.json"
        )["records"]
    }
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>플랭크 오류 개선 실험</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px;background:#14212c;color:#edf5fc}a{color:#8addff}section{padding:18px;margin:20px 0;background:#243a4b;border-radius:12px}td,th{padding:8px;border-bottom:1px solid #678;text-align:left}video{width:100%;max-width:800px}.notice{border-left:4px solid #ffd17d;padding:12px}</style><h1>플랭크 오류 개선 실험</h1><p><a href="index.html">5종 비교</a> · <a href="plank_side_progress.html">직전 플랭크 개발 결과</a> · <a href="plank_balance_trial.json">모든 예측·가중치 실험 결과</a> · <a href="plank_balance_challenge_results.json">신규 출처 확인</a></p><section><h2>학습 폴드 안에서 비중 조정</h2><p>공통 입력·초안·기존 개발 자료 10개·8명을 고정했습니다. 정상 8개·오류 2개의 불균형과 같은 사람의 반복 시범을 학습 가중치로 보정했습니다. 제외 참여자의 라벨은 가중치·결측 대체값 계산에 쓰지 않습니다. Heavy는 그대로이고 별도 ExtraTrees 모델을 다시 학습했습니다.</p><p class="notice">아래 수치는 이미 확인한 개발 자료의 참여자 제외 진단입니다. 전문가 정답·독립 최종 정확도가 아니며 새 오류 참여자는 아직 추가하지 못했습니다. 균형 학습 후보와 새 규칙안은 공통 모델·규칙을 교체하지 않은 별도 실험입니다.</p><table><tr><th>AI 설정</th><th>전체 MAE</th><th>정상 MAE(8개)</th><th>오류 MAE(2개)</th></tr>'
    ]
    for variant in trial["variants"]:
        parts.append(
            "<tr><td>"
            + esc(variant["variant"])
            + "</td>"
            + "".join(
                f'<td>{variant["metrics"][k]["ai_score"]["MAE"]:.2f}</td>'
                for k in ["all", "normal", "error"]
            )
            + "</tr>"
        )
    parts.append(
        "</table><p>balanced_leaf2는 오류 오차를 줄이는 대신 정상 오차가 증가했습니다. balanced_leaf1은 전체 평균이 더 작지만 오류 두 사례의 오차가 다시 커집니다. 전체 평균만으로 설정을 고르지 않습니다.</p></section><section><h2>Rule Base 별도 개선안</h2><p>무릎 각도만으로 놓친 골반 추가 지지 실패를 계기로, 몸통선 대비 골반 이탈과 팔꿈치–발끝 선 대비 골반 높이 대리값을 추가했습니다. 알려진 실패를 보고 설계했으므로 아래 개선은 독립 검증 성과가 아닙니다. 실제 접촉·하중을 확정하지 않습니다.</p><table><tr><th>방식</th><th>전체 MAE</th><th>정상 MAE</th><th>오류 MAE</th></tr>"
    )
    metrics = trial["variants"][0]["metrics"]
    for method in ["original_rule_score", "trial_rule_score"]:
        parts.append(
            "<tr><td>"
            + esc(method)
            + "</td>"
            + "".join(
                f'<td>{metrics[k][method]["MAE"]:.2f}</td>'
                for k in ["all", "normal", "error"]
            )
            + "</tr>"
        )
    parts.append(
        "</table><p>경계값·접촉 대리값은 미검증입니다. 새 출처의 정상 1개만으로 오류 일반화를 확인할 수 없어 공통 규칙에 채택하지 않았습니다.</p></section><section><h2>새 출처 개발 확인</h2><p>새 원본 2개·참여자 그룹 2개는 학습에 넣지 않았습니다. 한 구간은 측면 정상 초안, 다른 구간은 사선에서 발 위치를 바꾸는 전환입니다. 전환을 큰 골반 오류 0점으로 변환하지 않았고, 보이지 않는 항목·총점은 보류했습니다. 이번에 새 총점 오류 라벨은 추가하지 못했으며 학습 자료는 10개·8명 그대로입니다.</p></section>"
    )
    for row in challenge["results"]:
        identifier = row["id"]
        export = exports[identifier]
        draft = direct[identifier]
        parts.append(
            f'<section><h2>{esc(identifier)}</h2><p>{esc(draft["note_ko"])}</p><p>직접 항목 초안 {row["direct_levels"]} · 직접 총점 {row["direct_total"]}</p><table><tr><th>방식</th><th>관측 가능한 항목</th><th>총점</th></tr>'
        )
        for method, values in row["methods"].items():
            parts.append(
                f'<tr><td>{esc(method)}</td><td>{values["levels"]}</td><td>{values["score"]}</td></tr>'
            )
        folder = audit.OUT / "plank_improvement_overlays" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            pathlib.Path(row["extraction"]["cache"]), folder / "side.npz"
        )
        manifest = dict(
            id=identifier, exercise="plank", video_path=export["video_path"]
        )
        comparison = dict(
            cache_folder=str(folder),
            ai_score=row["methods"]["balanced_leaf2"]["score"],
            rule_score=row["methods"]["trial_rule"]["score"],
        )
        video = render.overlay(manifest, comparison)
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, int(cap.get(7)) // 2)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Preview frame unavailable")
        thumbnail = folder / "preview.jpg"
        if not cv2.imwrite(str(thumbnail), pixels):
            raise OSError("Preview save failed")
        parts.append(
            f'</table><p>영상 표시는 balanced_leaf2 AI와 별도 trial_rule이며 기존 공통 규칙이 아닙니다.</p><video preload="none" controls poster="{thumbnail.relative_to(audit.OUT).as_posix()}" src="{video.relative_to(audit.OUT).as_posix()}"></video><p>제작자 {esc(export["creator"])} · <a href="{esc(export["source_url"])}">원본 YouTube</a> · <a href="https://www.youtube.com/t/creative_commons">CC Attribution 재사용 표기</a><br>고정 크롭 {export["crop_xyxy"]} · 무음·관절 표시 파생 영상 · <a href="plank_balance_challenge_direct.json">예측 확인 전 동결 직접 초안</a></p></section>'
        )
    parts.append(
        '<section><h2>라벨·자료 보충의 남은 한계</h2><p>Pannell 2등급 초안과 0등급 규칙의 불일치는 원본 재검토에서도 의복·관절 중심과 구간 내 골반 높이 변화가 섞인 불확실 사례로 남았습니다. 규칙 점수로 초안을 덮어쓰지 않았고 총점 학습에 추가하지 않았습니다. Jonple의 매트 추가 지지는 직접 보이지만 규칙의 무릎 각도 대리값이 놓친 사례입니다.</p><p>새 오류·중간 등급 참여자를 추가하고 직접 초안을 별도로 검토해야 합니다. 현재 실험만으로 5종 개발·검증 완료를 주장하지 않습니다.</p><a href="plank_pannell_reinspection.json">재검토 기록</a> · <a href="plank_alignment_diagnostic.json">각도·좌표 민감도</a> · <a href="plank_context_trial.json">추적 문맥 실험</a></section></html>'
    )
    target = audit.OUT / "plank_improvement_trial.html"
    target.write_text("".join(parts), encoding="utf-8")
    for name in [
        "index.html",
        "plank_side_progress.html",
        "plank_error_review.html",
    ]:
        path = audit.OUT / name
        page = path.read_text(encoding="utf-8")
        if 'href="plank_improvement_trial.html"' not in page:
            page = page.replace(
                "</h1>",
                '</h1><p><a href="plank_improvement_trial.html">최신 플랭크 오류·균형 학습·규칙 실험</a></p>',
                1,
            )
            path.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
