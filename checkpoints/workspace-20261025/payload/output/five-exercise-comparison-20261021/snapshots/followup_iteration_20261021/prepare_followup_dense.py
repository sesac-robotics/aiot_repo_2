"""선별 후 후보의 시간순 검토판을 준비하며 직접 등급은 생성하지 않는다."""

import pathlib

import audit
from acquire_cc_youtube import run as acquire
from external_review import contact_sheet
from prepare_source_ranges import run as prepare_ranges


def run() -> dict:
    """새 공개 플랭크 두 출처와 QEVD W·상지 사건의 상세 표본을 준비한다.

    Returns:
        dict: 36표본 검토판, 원본 계보와 규칙 동결 해시. 직접 확인 전에는
        점수·새 사람·오류 사례로 집계하지 않는다.

    Raises:
        OSError: 원본·목록·이미지 접근 실패.
        ValueError: 라이선스·원본 구간 또는 디코딩 오류.
        RuntimeError: 자원 보호·다운로드 실패.
    """
    acquisitions = [acquire(i, 360) for i in ["dXX3cNj9a0o", "fBywcddsrvw"]]
    candidates = audit.read_json(
        audit.OUT / "followup_candidate_preparation.json"
    )["records"]
    ids = [
        "qevd_00182380",
        "qevd_00175236",
        "qevd_00052083",
        "qevd_00192144",
        "qevd_00254079",
        "qevd_00059326",
        "qevd_00188392",
        "qevd_00262575",
    ]
    result = dict(
        rule_sha256=audit.digest(
            pathlib.Path(__file__).parent / "w_component_trial.py"
        ),
        labels_created=0,
        acquisitions=acquisitions,
        W_records=[],
    )
    for row in candidates:
        if row["id"] not in ids:
            continue
        sheet = contact_sheet(
            pathlib.Path(row["video_path"]),
            audit.OUT / f'review/followup_dense_{row["id"]}.jpg',
            0,
            None,
            36,
        )
        result["W_records"].append(dict(row, dense_sheet=sheet))
    ranges = []
    for record_id in ["qevd_00166139", "qevd_00233559", "qevd_00266575"]:
        row = next(r for r in candidates if r["id"] == record_id)
        ranges.append(
            dict(
                id="followup_" + record_id,
                source_id=record_id,
                video_path=row["video_path"],
                start=1.0,
                end=4.0,
            )
        )
    ranges.extend(
        [
            dict(
                id="followup_prone_y",
                source_id="GSItbmlKGL0",
                start=8.8,
                end=10.4,
            ),
            dict(
                id="followup_prone_w",
                source_id="GSItbmlKGL0",
                start=11.6,
                end=13.2,
            ),
        ]
    )
    plan = audit.OUT / "followup_dense_ranges.json"
    audit.save_json(plan, dict(records=ranges))
    result["range_preparation"] = prepare_ranges(plan)
    audit.save_json(audit.OUT / "followup_dense_preparation.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["W_records"]))
