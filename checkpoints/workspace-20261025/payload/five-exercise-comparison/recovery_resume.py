"""온도 보호로 멈춘 보강 추출을 동일 계획·검증된 캐시에서 재개한다."""

import json
import pathlib
import time

import numpy as np

import audit
import extract
from recovery_prepare import ROOT
from resources import check_resources


def run() -> dict:
    """완료 접두 구간의 원본·캐시 지문을 확인한 뒤 남은 추출만 수행한다.

    Returns:
        dict: 기존 완료 수·추가 완료 수·전체 완료 여부. 기존 캐시는 삭제하지
        않는다. 자원 보호 실패 시 자동 재시작하지 않고 체크포인트를 남긴다.

    Raises:
        OSError: 계획·원본·캐시 접근 실패.
        ValueError: 계획 순서·원본 SHA-256·구간·관절 캐시 지문 불일치.
        RuntimeError: 온도·메모리 보호 조건 위반.
    """
    plan = audit.read_json(ROOT / "input_plan.json")["records"]
    report = audit.read_json(ROOT / "extraction.json")
    count = len(report["records"])
    if [r["id"] for r in report["records"]] != [r["id"] for r in plan[:count]]:
        raise ValueError("Checkpoint is not a prefix of the current plan")
    hashes = {}
    for index, (row, completed) in enumerate(zip(plan, report["records"])):
        if index % 16 == 0:
            check_resources()
        for name, path, sha in [
            ("extraction", row["video_path"], row["video_sha256"])
        ] + (
            [("front_extraction", row["front_path"], row["front_sha256"])]
            if "synchronized_front" in row["required_views"]
            else []
        ):
            receipt = completed[name]
            config = receipt["config"]
            if path not in hashes:
                hashes[path] = audit.digest(pathlib.Path(path))
            if (
                hashes[path] != sha
                or config["source_sha256"] != sha
                or config["segment"] != row["segment"]
                or config["required_views"] != row["required_views"]
            ):
                raise ValueError("Checkpoint source or segment differs")
            with np.load(receipt["cache"]) as data:
                if str(data["fingerprint"]) != receipt["fingerprint"]:
                    raise ValueError("Checkpoint cache fingerprint differs")
        time.sleep(0.05)
    for row in plan[count:]:
        check_resources()
        record = dict(
            id=row["id"],
            extraction=extract.extract_view(
                row, "primary", row["video_path"], destination=ROOT / "heavy"
            ),
        )
        if "synchronized_front" in row["required_views"]:
            record["front_extraction"] = extract.extract_view(
                row, "front", row["front_path"], destination=ROOT / "heavy"
            )
        report["records"].append(record)
        audit.save_json(ROOT / "extraction.json", report)
        time.sleep(1)
    report.update(
        completed=True,
        input_plan_sha256=audit.digest(ROOT / "input_plan.json"),
        resumed_from_verified_records=count,
    )
    audit.save_json(ROOT / "extraction.json", report)
    return dict(previous=count, added=len(plan) - count, completed=True)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
