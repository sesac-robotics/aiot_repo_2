"""기존 80개와 다른 보유 QEVD 개발 플랭크의 구도 선별을 확대한다."""

import hashlib
import json

import audit
from recovery_plank_screen import run
from recovery_prepare import ROOT


def expand() -> dict:
    """새 참여자 그룹을 우선해 보유 학습 후보 200개의 자동 선별을 실행한다.

    Returns:
        dict: 자동 선별·검토판 준비 수. 직접 라벨·학습 자료 수가 아니다.

    Raises:
        OSError: 보유 목록·캐시·출력 접근 실패.
        ValueError: 영상·프레임 계약 오류.
        RuntimeError: 온도·메모리 보호 조건 위반.
    """
    old = audit.read_json(ROOT / "plank_screen.json")["records"]
    known = {x["id"] for x in old}
    people = {x["person"] for x in old}
    pool = [
        r
        for r in audit.read_json(
            audit.BASE / "output/overnight-20261020/score_candidates.json"
        )
        if r["exercise"] == "plank"
        and r["id"].startswith("qevd_")
        and r["split"] == "train"
        and r["id"] not in known
    ]
    pool.sort(
        key=lambda r: (
            r["person"] in people,
            hashlib.sha256(
                ("plank-expanded-local" + r["id"]).encode()
            ).hexdigest(),
        )
    )
    selected = []
    used = set()
    for row in pool:
        if row["person"] in used:
            continue
        used.add(row["person"])
        selected.append(row)
        if len(selected) == 200:
            break
    audit.save_json(
        ROOT / "plank_expanded_selection.json",
        dict(
            records=selected,
            automatic_only=True,
            original_screen_preserved=True,
        ),
    )
    return run(selected, "plank_screen_expanded_200.json")


if __name__ == "__main__":
    print(json.dumps(expand()))
