"""스쿼트 부분 복귀 발 관측 실험의 개선·실패를 별도 로컬 웹에 표시한다."""

import html
import json

import audit
from allfive_round1_report import row_html
from held_squat_observation_trial import OUT


def build() -> dict:
    """공통 개발 교집합과 가용 비율을 분리한 후보 비교 HTML을 저장한다.

    Returns:
        dict: 보고서 절대 경로. 기존 5종 비교 영상·모델을 교체하지 않는다.

    Raises:
        ValueError: 실험이 미완료인 경우.
        OSError: 비교 결과 접근 또는 HTML 저장 실패.
    """
    data = audit.read_json(OUT / "comparison.json")
    if not data["completed"]:
        raise ValueError("Comparison must complete first")
    current = data["fixed"]
    paired = data["paired"]
    overview = [
        ["같은 전후 총점 교집합 수", paired["common_total_records"]],
        ["기존 AI / Rule MAE (점)", paired["total"]["baseline"]],
        [
            "부분 복귀 관측 후보 AI / Rule MAE (점)",
            paired["total"]["candidate"],
        ],
        ["후보 가용·전체·항목 오차", current],
        ["연결 그룹 3분할 교차검증", data["group_cv"]],
        ["발 관측 가능 변경", data["observation_changes"]],
    ]
    rows = []
    for p in data["fixed_predictions"]:
        rows.append(
            row_html(
                [
                    p["id"],
                    p["target_score"],
                    p["ai_score"],
                    p["rule_score"],
                    p["levels"],
                    p["ai_levels"],
                    p["rule_levels"],
                ]
            )
        )
    source = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>스쿼트 부분 복귀와 발 관측 분리</title><style>
body{font:16px sans-serif;max-width:1200px;margin:32px auto;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #aaa;padding:8px;overflow-wrap:anywhere}
</style><h1>스쿼트: 복귀 등급과 발 관측 분리</h1>
<p>발 변화 측정에 완전 신전을 요구하던 조건을, 최대 무릎 각도 125도 이상과
25도 이상 움직임이 확인되는 부분 복귀에서도 측정하도록 바꿨다.
필요 관절·발 길이·동시 관측률·추적 승인 조건은 그대로 적용한다.
정지 자세를 반복으로 만들거나 복귀 점수·직접 등급·완결 여부를 바꾸지 않는다.</p>
<p>기존 학습·개발 분할의 직접 초안으로만 재학습했다. 반복 사용한 개발 검증이며
새로운 독립 최종 시험·전문가 정확도가 아니다. 검증 라벨을 수치 학습에 넣지 않았다.
확대 원본에서 뒤꿈치 들림이 확정되지 않아 철회한 4개 가설 라벨은 사용하지 않았다.</p>
<p><a href="../held_round2/index.html">앞선 5종 모델 비교·관절 영상</a> ·
<a href="comparison.json">전체 비교 JSON</a> ·
<a href="candidate_input_manifest.json">이번 실제 입력</a> ·
<a href="../held_squat_errors/hypothesis_review.json">철회한 오류 가설</a></p><table>"""
    source += "".join(
        row_html([k, json.dumps(v, ensure_ascii=False)]) for k, v in overview
    )
    source += (
        "</table><table><tr><th>ID</th><th>직접 초안 /100</th>"
        "<th>AI /100</th><th>Rule /100</th><th>직접 /2</th>"
        "<th>AI /2</th><th>Rule /2</th></tr>"
        + "".join(rows)
        + "</table></html>"
    )
    (OUT / "index.html").write_text(source, encoding="utf-8")
    return dict(path=str(OUT / "index.html"), browser_verified=False)


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
