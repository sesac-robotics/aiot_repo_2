"""새 YTW 후보 3개의 규칙 결과 전 시간순 직접 검토 자료를 준비한다."""

import pathlib

import audit
from external_review import contact_sheet


def run() -> dict:
    """후보별 전 구간 36장 표본과 고정 규칙·단계 코드 해시를 저장한다.

    Returns:
        dict: 원본 SHA·FPS·표본 프레임과 동결 코드 해시. 등급·동작 구간
        정답은 생성하지 않으며 이미 5장 구도 검토에 노출된 개발 자료다.

    Raises:
        OSError: 원본·결과 접근 실패.
        ValueError: 원본 디코딩·구간 오류.
        RuntimeError: 자원 보호 중단.
    """
    rows = audit.read_json(audit.OUT / "next_view_candidates.json")["records"]
    result = dict(
        rule_sha256=audit.digest(
            pathlib.Path(__file__).parent / "w_morphology.py"
        ),
        phase_sha256=audit.digest(
            pathlib.Path(__file__).parent / "ytw_phases.py"
        ),
        records=[],
        labels_before_new_Heavy_or_rule=True,
        exposure="five_view_samples_already_reviewed; development_only",
    )
    for row in rows:
        if row["exercise"] != "ytw":
            continue
        sheet = contact_sheet(
            pathlib.Path(row["video_path"]),
            audit.OUT / f'review/next_w_{row["id"]}.jpg',
            0,
            None,
            36,
        )
        result["records"].append(dict(candidate=row, temporal_sheet=sheet))
    audit.save_json(audit.OUT / "next_w_review_preparation.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
