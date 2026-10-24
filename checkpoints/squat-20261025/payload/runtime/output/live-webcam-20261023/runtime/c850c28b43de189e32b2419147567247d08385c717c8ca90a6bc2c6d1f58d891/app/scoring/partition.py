"""참여자·원본·촬영 연결 성분으로 개발과 최종 시험의 누출을 차단한다.

운동 종류나 구간 경계가 달라도 같은 연결 성분은 한 역할만 가진다.
정답·제공자 문자열·모델 예측은 읽지 않는다. 사람 식별 자체를 수행하지는
않으므로 호출자는 확인한 참여자와 중복 촬영 묶음을 제공해야 한다.
"""

import collections
import re
from collections.abc import Sequence

ROLES = frozenset(("train", "validation", "final_test"))


def audit_partitions(
    records: Sequence[dict],
    *,
    historical_people: Sequence[str] = (),
    historical_sources: Sequence[str] = (),
    historical_recordings: Sequence[str] = (),
    historical_duplicate_groups: Sequence[str] = (),
) -> dict:
    """최종 시험의 역사적 노출과 역할 간 전이적 중복을 검사한다.

    Args:
        records: id·exercise·split·person·sources 필수 행. sources는 원본
            SHA-256 목록이다. recording_groups·duplicate_groups는 확인한
            동일 촬영·재인코딩 중복 묶음 목록이며 선택 입력이다. split은
            train·validation·final_test만 허용한다. 프레임·등급은 읽지 않는다.
        historical_people: 과거 학습·검증·모델 선택에 노출된 참여자 ID.
        historical_sources: 과거 노출 원본 SHA-256. 단위 없는 64자리 hex다.
        historical_recordings: 과거 노출된 동일 촬영 묶음 ID.
        historical_duplicate_groups: 과거 노출된 재인코딩·잘라낸 원본 묶음 ID.

    Returns:
        dict: 행 수·역할별 수·연결 그룹별 행 ID와 역할. 단위는 개수다.
        통과는 주어진 메타데이터의 분리를 뜻하며 실제 사람 식별 확인이나
        독립 최종 시험의 수행·성능을 보증하지 않는다. 입력은 변경하지 않는다.

    Raises:
        ValueError: 필수 메타데이터·지문 누락, 중복 ID, 역할 간 연결 성분
            중복, 최종 시험의 역사적 노출이 있으면 학습 전에 실패한다.
    """
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (
            *historical_people,
            *historical_recordings,
            *historical_duplicate_groups,
        )
    ) or any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in historical_sources
    ):
        raise ValueError(
            "Historical exposure needs valid IDs and source hashes"
        )
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[str, int] = {}
    identifiers: set[str] = set()
    exposed = {
        *("person:" + x for x in historical_people),
        *("source:" + x for x in historical_sources),
        *("recording:" + x for x in historical_recordings),
        *("duplicate:" + x for x in historical_duplicate_groups),
    }
    exposure_roots = set()
    for index, record in enumerate(records):
        identifier = record.get("id")
        person = record.get("person")
        sources = record.get("sources")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in identifiers
            or not isinstance(person, str)
            or not person.strip()
            or record.get("split") not in ROLES
            or not isinstance(record.get("exercise"), str)
            or not isinstance(sources, list)
            or not sources
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in sources
            )
        ):
            raise ValueError(
                "Valid unique ID, person, role and source hashes required"
            )
        identifiers.add(identifier)
        keys = ["person:" + person] + ["source:" + x for x in sources]
        for field, prefix in (
            ("recording_groups", "recording:"),
            ("duplicate_groups", "duplicate:"),
        ):
            values = record.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(x, str) or not x.strip() for x in values
            ):
                raise ValueError(
                    "Recording and duplicate groups must be ID lists"
                )
            keys.extend(prefix + x for x in values)
        if any(key in exposed for key in keys):
            exposure_roots.add(index)
        for key in keys:
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[dict]] = collections.defaultdict(list)
    for index, record in enumerate(records):
        groups[root(index)].append(record)
    historical_components = {root(index) for index in exposure_roots}
    output = []
    for key, rows in groups.items():
        roles = {row["split"] for row in rows}
        if len(roles) != 1:
            raise ValueError(
                "Connected participant/source/recording crosses roles"
            )
        role = next(iter(roles))
        if role == "final_test" and key in historical_components:
            raise ValueError(
                "Final test overlaps historical development exposure"
            )
        output.append(dict(role=role, ids=[row["id"] for row in rows]))
    return dict(
        records=len(records),
        role_counts=dict(collections.Counter(row["split"] for row in records)),
        connected_groups=output,
        metadata_disjoint=True,
        person_identity_verified=False,
        final_test_executed=False,
    )
