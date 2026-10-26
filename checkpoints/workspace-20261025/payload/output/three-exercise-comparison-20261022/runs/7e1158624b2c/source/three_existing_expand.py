"""기존 라벨이 없는 보유 후보를 오류 유형별로 다시 준비한다.

과거 후보 목록에 언급된 영상도 재확인할 수 있다. 최초 직접 검토 여부는
추측하지 않으며 제공자 오류 문자열을 수치 라벨로 사용하지 않는다.
"""

import hashlib
import json
import pathlib

import audit
import external_review
from three_existing_review import ROOT


def run() -> dict:
    """기존 개발 분할의 미라벨 후보 20개를 시간순 24장 검토판으로 만든다.

    Returns:
        dict: 선정 후보와 원본 해시·프레임 번호. 자동 준비는 직접 검토가 아니다.

    Raises:
        OSError: 기존 입력·결과 읽기/쓰기 실패.
        ValueError: 영상 디코딩 또는 구간 계약 위반.
        RuntimeError: 자원 보호 조건 위반.
    """
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    excluded = {
        x["id"] for x in audit.read_json(ROOT / "preparation.json")["records"]
    }
    for folder, names in [
        (audit.OUT, ["labels.json", "supplemental_labels.json"]),
        (
            audit.BASE / "output/three-exercise-comparison-20261022",
            ["score_annotations.json", "additional_score_annotations.json"],
        ),
    ]:
        for name in names:
            excluded.update(x["id"] for x in audit.read_json(folder / name))
    priorities = {
        "pushups": [
            "legs on the floor",
            "elevated hips",
            "early stopping",
            "shallow",
        ],
        "squats": ["back not straight", "insufficient", "hold", "shallow"],
        "plank": [
            "high hips",
            "butt sinking",
            "on knees",
            "legs and hips on the floor",
        ],
        "ytw": ["torso parallel", "not low enough", "arms too low", "too fast"],
        "shoulder": ["bad", "bad", "good", "good"],
    }
    selected = []
    for exercise, terms in priorities.items():
        people = set()
        for term in terms:
            candidates = [
                x
                for x in pool
                if x["exercise"] == exercise
                and x["split"] == "train"
                and x["id"] not in excluded
                and x["person"] not in people
                and any(term in s for s in x["source_labels"])
                and (exercise != "shoulder" or x["id"].endswith("_front"))
            ]
            candidates.sort(
                key=lambda x: hashlib.sha256(
                    ("existing-errors-20261022" + x["id"]).encode()
                ).hexdigest()
            )
            if not candidates:
                continue
            row = candidates[0]
            people.add(row["person"])
            excluded.add(row["id"])
            selected.append(
                dict(
                    row,
                    selection_term=term,
                    prior_direct_review_not_asserted_absent=True,
                )
            )
    audit.save_json(
        ROOT / "expanded_selection.json",
        dict(records=selected, roles_frozen_before_pixels=True),
    )
    prepared = []
    for row in selected:
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + ".jpg"),
            0,
            None,
            24,
        )
        prepared.append(dict(row, preparation=receipt))
        audit.save_json(
            ROOT / "expanded_preparation.json",
            dict(records=prepared, direct_review_performed_by_script=False),
        )
    return dict(prepared=len(prepared))


if __name__ == "__main__":
    print(json.dumps(run()))
