"""낮은 팔 위치로 선별한 W 후보의 규칙 동결·시간순 검토판을 저장한다."""

import pathlib
import audit
from external_review import contact_sheet


def run() -> dict:
    """W 후보 12개의 36장 표본을 준비하고 코드 해시를 보존한다.

    Returns:
        dict: 원본·정확한 프레임·선정 출처·규칙 및 단계 코드 해시.
        다섯 장의 구도 표본은 이미 보았으므로 완전 미노출 시험이 아니다.

    Raises:
        OSError: 후보·원본·결과 파일 읽기 또는 쓰기 실패.
        ValueError: 영상 구간·디코딩·코드 동결 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    candidates = audit.read_json(audit.OUT / "error_batch_candidates.json")
    records = []
    rule = audit.BASE / "five-exercise-comparison/w_morphology.py"
    phase = audit.BASE / "five-exercise-comparison/ytw_phases.py"
    frozen = dict(
        rule_sha256=audit.digest(rule),
        phase_sha256=audit.digest(phase),
        source_split="train",
        independent_final_test=False,
        prior_exposure="five_temporal_view_screen_samples_per_clip",
        rows=records,
    )
    audit.save_json(audit.OUT / "w_targeted_freeze.json", frozen)
    for row in candidates["records"]:
        if row["exercise"] != "ytw":
            continue
        if (
            audit.digest(rule) != frozen["rule_sha256"]
            or audit.digest(phase) != frozen["phase_sha256"]
        ):
            raise ValueError("Frozen code changed")
        result = contact_sheet(
            pathlib.Path(row["video_path"]),
            audit.OUT / f'review/w_targeted_{row["id"]}.jpg',
            0,
            None,
            36,
        )
        records.append(dict(id=row["id"], person=row["person"], **result))
        audit.save_json(audit.OUT / "w_targeted_freeze.json", frozen)
    return frozen


if __name__ == "__main__":
    print(len(run()["rows"]))
