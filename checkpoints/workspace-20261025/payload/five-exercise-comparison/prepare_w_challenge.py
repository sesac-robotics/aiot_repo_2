"""규칙 경계를 고정한 뒤 새 W 검토 표본을 준비한다. 점수는 생성하지 않는다."""

import audit
from external_review import contact_sheet


def run() -> None:
    """기존 신규 구도 후보 앞 12개의 36장 시간순 표본을 보존한다.

    Returns:
        None: 규칙 해시·선정 ID·검토판 경로를 저장한다.

    Raises:
        OSError: 후보·영상 읽기 또는 출력 쓰기 실패.
        ValueError: 영상 해시·디코딩 불일치.
        RuntimeError: 자원 보호 조건 충족.
    """
    rows = audit.read_json(audit.OUT / "ytw_new_view_candidates.json")[
        "records"
    ][:12]
    record = dict(
        policy_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/w_morphology.py"
        ),
        selection="first 12 prior deterministic unique-person training-source candidates; frozen before dense review and rule results",
        independent_final_test=False,
        source_split="train",
        rows=[],
    )
    audit.save_json(audit.OUT / "w_challenge_freeze.json", record)
    for row in rows:
        path = audit.BASE / "five-exercise-comparison/w_morphology.py"
        if audit.digest(path) != record["policy_sha256"]:
            raise ValueError("Frozen rule changed during challenge")
        result = contact_sheet(
            __import__("pathlib").Path(row["video_path"]),
            audit.OUT / f'review/w_challenge_{row["id"]}.jpg',
            0,
            None,
            36,
        )
        record["rows"].append(
            dict(id=row["id"], person=row["person"], **result)
        )
        audit.save_json(audit.OUT / "w_challenge_freeze.json", record)


if __name__ == "__main__":
    run()
