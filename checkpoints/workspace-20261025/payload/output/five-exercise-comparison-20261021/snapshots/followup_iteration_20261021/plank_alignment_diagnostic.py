"""플랭크 직접 초안·투영 관절·고정 각도 규칙의 불일치를 분해한다."""

import numpy as np

import audit
import geometry
import rule_score
from resources import check_resources


def summarize(sequence: np.ndarray) -> dict:
    """양쪽 몸통선 각도와 단일 관절 위치의 4픽셀 민감도를 반환한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 미관측은 NaN이다.

    Returns:
        dict: 측면별 유효율, 각도 도 단위 분위수, 골반 높이의 몸 길이
        비율, 4픽셀 관절 오차 시 등급. 이 변동은 실제 재촬영이 아니다.

    Raises:
        ValueError: 관절 배열 형상 불일치.
    """
    channels = geometry.channels(sequence)
    result = {}
    for side, shoulder, hip, ankle in [
        ("left", 11, 23, 27),
        ("right", 12, 24, 28),
    ]:
        line = channels[f"{side}_line_deg"]
        valid = np.isfinite(line)
        sh, hp, an = (sequence[:, i] for i in [shoulder, hip, ankle])
        axis = an - sh
        length = np.linalg.norm(axis, axis=1)
        cross = axis[:, 0] * (hp[:, 1] - sh[:, 1]) - axis[:, 1] * (
            hp[:, 0] - sh[:, 0]
        )
        offset = np.divide(
            cross, length**2, out=np.full(len(line), np.nan), where=length > 1
        )
        quantiles = (
            np.percentile(line[valid], [10, 50, 90]).tolist()
            if valid.any()
            else None
        )
        variants = []
        for joint in [shoulder, hip, ankle]:
            for dx, dy in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
                varied = sequence.copy()
                varied[:, joint] += [dx, dy]
                values = geometry.channels(varied)[f"{side}_line_deg"]
                finite = values[np.isfinite(values)]
                grade = None
                if len(finite) >= 5 and len(finite) / len(values) >= 0.8:
                    measured = float(np.percentile(finite, 10))
                    grade = (
                        2 if measured >= 168 else 1 if measured >= 155 else 0
                    )
                variants.append(
                    dict(joint=joint, dx_pixels=dx, dy_pixels=dy, grade=grade)
                )
        result[side] = dict(
            observed_fraction=float(valid.mean()),
            angle_p10_p50_p90_deg=quantiles,
            signed_hip_offset_body_ratio=(
                float(np.nanmedian(offset))
                if np.isfinite(offset).any()
                else None
            ),
            local_coordinate_perturbations=variants,
        )
    return result


def run() -> dict:
    """공통 학습 7개와 새 5개 구간을 라벨 변경 없이 진단한다.

    Returns:
        dict: 기존·추가 개발 사례별 관측·투영·규칙 비교. 직접 초안의
        타당성이나 전문가 정답을 확정하지 않고 불일치 원인을 보존한다.

    Raises:
        OSError: 캐시·결과 접근 실패.
        ValueError: 캐시 좌표 형상 불일치.
        RuntimeError: 자원 보호 중단.
    """
    check_resources()
    cases = []
    common = audit.read_json(audit.OUT / "comparison.json")["results"]["plank"]
    for row in common["rule_only_previews"]:
        path = audit.OUT / "heavy" / row["id"] / "primary.npz"
        if not path.exists():
            path = audit.OUT / "heavy" / row["id"] / "side.npz"
        cases.append((row["id"], path, "existing_train"))
    for row in audit.read_json(audit.OUT / "plank_error_results.json")[
        "results"
    ]:
        cases.append((row["id"], row["extraction"]["cache"], "new_item_only"))
    results = []
    for identifier, path, role in cases:
        with np.load(path) as cache:
            seq = cache["image_xy"].copy()
        rule = rule_score.evaluate(
            "plank",
            seq,
            None,
            [
                "trunk_line",
                "elbow_sagittal_position",
                "leg_and_body_support",
                "support_continuity",
            ],
            [40, 25, 20, 15],
        )
        results.append(
            dict(
                id=identifier,
                role=role,
                side_statistics=summarize(seq),
                fixed_rule=rule["evidence"]["trunk_line"],
            )
        )
    report = dict(
        records=results,
        source_drafts_modified=False,
        common_rule_modified=False,
        independent_final_test=False,
        expert_validated=False,
        findings_ko=[
            "규칙의 168·155도는 검증 전 경계이며 직접 초안의 시각적 등급과 동일하다고 보장하지 않는다.",
            "어깨–골반–발목 관절 중심의 화면 각도는 의복 외곽·척추 곡선과 다르다.",
            "4픽셀 변동은 원본 픽셀 단위 좌표 진단이며 실제 관절 오차 분포나 재촬영 검증이 아니다.",
            "직접 초안과 규칙 불일치의 원인을 하나로 확정하거나 규칙 출력을 정답으로 쓰지 않는다.",
        ],
    )
    audit.save_json(audit.OUT / "plank_alignment_diagnostic.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            (r["id"], r["side_statistics"])
            for r in run()["records"]
            if "pannell_normal" in r["id"]
        ]
    )
