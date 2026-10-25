"""다종목 선택 원본의 사람·분할·라벨과 관절 캐시 연결을 검사합니다. 데이터 양뿐 아니라 사용 가능 범위와 누수 여부를 확인합니다.

Report available subjects and pose-quality attrition per error, not just video totals.
"""

import json
from collections import Counter

import numpy as np
from suite_config import SUITE, TASKS


def main():
    report = {}
    for exercise, config in TASKS.items():
        directory = SUITE / exercise
        rows = json.loads((directory / "selected.json").read_text())
        if not (directory / "data.npz").exists():
            continue
        with np.load(directory / "data.npz") as data:
            usable = set(data["ids"])
        kept = [r for r in rows if r["video_id"] in usable]
        entries = {}
        for j, target in enumerate(config["targets"]):
            source = [r for r in rows if r["target"][j]]
            pose = [r for r in kept if r["target"][j]]
            entries[target] = dict(
                source_videos=len(source),
                source_workers=len({r["worker_id"] for r in source}),
                pose_usable_videos=len(pose),
                pose_usable_workers=len({r["worker_id"] for r in pose}),
                source_test_positive_workers=len(
                    {
                        r["worker_id"]
                        for r in pose
                        if r["source_split"] == "test"
                    }
                ),
            )
        report[exercise] = dict(
            selected_videos=len(rows),
            usable_videos=len(kept),
            selected_workers=len({r["worker_id"] for r in rows}),
            usable_workers=len({r["worker_id"] for r in kept}),
            pose_drop_count=len(rows) - len(kept),
            source_split_counts=dict(Counter(r["source_split"] for r in kept)),
            per_target=entries,
        )
    (SUITE / "dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: {n: v for n, v in r.items() if n != "per_target"}
                for k, r in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
