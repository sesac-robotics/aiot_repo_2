"""보유 후보의 검토 근거를 대조하고 사전 분할을 유지한 검토판을 준비한다.

제공자 문자열은 후보 균형 선정에만 사용한다. 점수나 직접 검토 판정을
생성하지 않으며 기존 모델·라벨·플랭크·YTW 결과를 수정하지 않는다.
"""

import collections
import hashlib
import json
import pathlib
import re

import audit
import external_review
from resources import check_resources

ROOT = audit.BASE / "output/existing-data-audit-20261022"


def run() -> dict:
    """로컬 후보 전수 경로를 검사하고 종목별 미기록 개발 후보 6개를 준비한다.

    Returns:
        dict: 경로 존재 수·과거 산출물 언급 여부·선정 후보·시간순 검토판.
        언급이 없다는 것은 직접 검토가 없었다는 완전한 증명이 아니다.
        시험 분할은 읽거나 디코딩하지 않는다.

    Raises:
        OSError: 원본·기록·결과 접근 실패.
        ValueError: 개발/시험 분할 또는 디코딩 계약 위반.
        RuntimeError: 자원 보호 조건 위반.
    """
    ROOT.mkdir(exist_ok=True)
    (ROOT / "review").mkdir(exist_ok=True)
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    excluded_files = {
        "score_candidates.json",
        "video_inventory.json",
        "qevd_inventory.json",
        "fine_grained_labels.json",
        "fine_grained_labels_with_worker_ids.json",
        "candidate_manifest.json",
        "inventory_before_review.json",
    }
    mentions = collections.defaultdict(list)
    for path in (audit.BASE / "output").rglob("*.json"):
        if (
            path.name in excluded_files
            or any(x in path.parts for x in ("runs", "snapshots", "qevd-json"))
            or ROOT in path.parents
            or path.stat().st_size > 8_000_000
        ):
            continue
        text = path.read_text(errors="replace")
        for identifier in set(
            re.findall(r"qevd_\d{8}|subject_\d+_[a-z_]+", text)
        ):
            mentions[identifier].append(str(path))
    counts = collections.Counter()
    missing, selected = [], []
    terms = {
        "pushups": [
            "legs on the floor",
            "elevated hips",
            "shallow",
            "early stopping",
            "90 degrees",
            "deep",
        ],
        "squats": [
            "back not straight",
            "shallow",
            "insufficient",
            "hold",
            "90 degrees",
            "no obvious issue",
        ],
        "plank": [
            "legs and hips on the floor",
            "on knees",
            "high hips",
            "butt sinking",
            "average",
            "stopping early",
        ],
        "ytw": [
            "torso parallel",
            "not low enough",
            "arms too low",
            "bent legs",
            "too slow",
            "too fast",
        ],
        "shoulder": ["bad", "bad", "bad", "good", "good", "good"],
    }
    for row in pool:
        exists = pathlib.Path(row["path"]).is_file()
        counts[
            (row["exercise"], row["split"], exists, bool(mentions[row["id"]]))
        ] += 1
        if not exists:
            missing.append(row["id"])
    for exercise, priorities in terms.items():
        used_people = set()
        for index, term in enumerate(priorities):
            split = "train" if index < 4 else "validation"
            candidates = [
                x
                for x in pool
                if x["exercise"] == exercise
                and x["split"] == split
                and not mentions[x["id"]]
                and x["person"] not in used_people
                and any(term in label for label in x["source_labels"])
                and pathlib.Path(x["path"]).is_file()
            ]
            candidates.sort(
                key=lambda x: hashlib.sha256(
                    ("existing-first-20261022" + x["id"]).encode()
                ).hexdigest()
            )
            if not candidates:
                continue
            row = candidates[0]
            used_people.add(row["person"])
            selected.append(
                dict(row, selection_term=term, role_assigned_before_pixels=True)
            )
    manifest = dict(
        candidate_count=len(pool),
        missing_paths=missing,
        inventory=[
            dict(
                exercise=e,
                split=s,
                exists=x,
                mentioned_in_prior_artifact=m,
                count=n,
            )
            for (e, s, x, m), n in sorted(counts.items())
        ],
        selection=selected,
        external_discovery_paused=True,
        past_artifact_mention_is_not_proof_of_direct_review=True,
        new_direct_reviews=0,
        new_score_labels=0,
    )
    audit.save_json(ROOT / "inventory.json", manifest)
    prepared = []
    for row in selected:
        check_resources()
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + ".jpg"),
            0,
            None,
            18,
        )
        prepared.append(dict(row, preparation=receipt))
        audit.save_json(
            ROOT / "preparation.json",
            dict(records=prepared, direct_review_performed_by_script=False),
        )
    return dict(
        selected=len(selected), prepared=len(prepared), missing=len(missing)
    )


if __name__ == "__main__":
    print(json.dumps(run()))
