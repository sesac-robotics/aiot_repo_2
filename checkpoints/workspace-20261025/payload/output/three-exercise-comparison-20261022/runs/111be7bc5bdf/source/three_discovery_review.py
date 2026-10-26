"""이번 외부 검색에서 재사용 후보로 확보한 원본의 검토판만 준비한다."""

import pathlib

import audit
import external_review
import three_protocol
from resources import check_resources


def run() -> dict:
    """원본·사전 역할·라이선스와 연결한 48장 개요 검토판을 저장한다.

    Returns:
        dict: 원본별 검토판 경로. 직접 검토·라벨 수는 생성하지 않는다.

    Raises:
        OSError: 원본·명세·검토판 파일 접근 실패.
        ValueError: 원본 해시 또는 역할·이용 조건 불일치.
        RuntimeError: 전원·온도·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    results = []
    for source in audit.read_json(root / "external_discovery_next.json")[
        "records"
    ]:
        if not source.get("video_path"):
            continue
        check_resources()
        if not source["reusable_metadata_gate"] or source[
            "evaluation_role"
        ].startswith("prospective"):
            raise ValueError("Unqualified source or reserved input")
        path = pathlib.Path(source["video_path"])
        if audit.digest(path) != source["video_sha256"]:
            raise ValueError("Source changed after discovery")
        sheet = external_review.contact_sheet(
            path,
            root / "review" / ("next_source_" + source["id"] + ".jpg"),
            0,
            None,
            48,
        )
        results.append(dict(**source, sheet=sheet, automatically_prepared=True))
        audit.save_json(
            root / "external_next_review_preparation.json",
            dict(records=results, directly_reviewed=False),
        )
    return dict(records=results, directly_reviewed=False)


if __name__ == "__main__":
    print(len(run()["records"]))
