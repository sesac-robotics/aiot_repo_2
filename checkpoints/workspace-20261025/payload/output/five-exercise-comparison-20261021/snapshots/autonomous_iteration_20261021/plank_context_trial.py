"""동일 플랭크 원본 프레임을 추적 시작 시점만 달리하여 비교한다."""

import copy

import numpy as np

import audit
from extract import extract_view
import geometry
import rule_score


def run() -> dict:
    """새 5개 구간에 1·3초 실제 이전 문맥을 추가해 같은 프레임을 대조한다.

    Returns:
        dict: 공통 원본 프레임의 관절 이동 픽셀·몸통 각도·규칙 등급.
        문맥 프레임은 채점에 넣지 않으며 직접 라벨·공통 추출기를 바꾸지
        않는다. 프레임 교집합만 비교하고 보간하지 않는다. 독립 시험이 아니다.

    Raises:
        OSError: 캐시·원본·결과 접근 실패.
        ValueError: 원본 프레임 교집합 누락 또는 형상 오류.
        RuntimeError: 자원 보호·추출 실패.
    """
    direct = audit.read_json(audit.OUT / "plank_error_direct_labels.json")
    results = []
    ids = direct["criterion_ids"]
    weights = direct["weights"]
    for row in direct["reviews"]:
        original = audit.OUT / "plank_error_cache" / row["id"] / "primary.npz"
        with np.load(original) as cache:
            frames = cache["frame_numbers"].copy()
            baseline = cache["image_xy"].copy()
        variants = []
        for seconds in [1, 3]:
            item = copy.deepcopy(row)
            item["id"] += f"_context{seconds}"
            item["required_views"] = ["item_only_context_trial"]
            item["segment"]["start_frame"] = max(
                0,
                item["segment"]["start_frame"]
                - round(seconds * item["segment"]["fps"]),
            )
            extraction = extract_view(
                item,
                "primary",
                row["video_path"],
                audit.OUT / "plank_context_cache",
            )
            with np.load(extraction["cache"]) as cache:
                common, old, new = np.intersect1d(
                    frames, cache["frame_numbers"], return_indices=True
                )
                contextual = cache["image_xy"][new].copy()
            if len(common) < 3:
                raise ValueError("Insufficient identical source frames")
            cold = baseline[old]
            observed = np.isfinite(cold).all(2) & np.isfinite(contextual).all(2)
            distance = np.linalg.norm(cold - contextual, axis=2)
            cold_rule = rule_score.evaluate("plank", cold, None, ids, weights)
            context_rule = rule_score.evaluate(
                "plank", contextual, None, ids, weights
            )
            variants.append(
                dict(
                    context_seconds=seconds,
                    extraction=extraction,
                    common_source_frames=common.tolist(),
                    compared_samples=len(common),
                    pixel_difference_median=(
                        float(np.median(distance[observed]))
                        if observed.any()
                        else None
                    ),
                    pixel_difference_max=(
                        float(np.max(distance[observed]))
                        if observed.any()
                        else None
                    ),
                    cold_levels=cold_rule["levels"],
                    context_levels=context_rule["levels"],
                    cold_trunk_evidence=cold_rule["evidence"]["trunk_line"],
                    context_trunk_evidence=context_rule["evidence"][
                        "trunk_line"
                    ],
                    context_joint_observed_fraction=np.isfinite(contextual)
                    .all(2)
                    .mean(0)
                    .tolist(),
                )
            )
        results.append(
            dict(
                id=row["id"],
                direct_levels=row["draft_levels"],
                variants=variants,
                total_score=None,
            )
        )
    report = dict(
        records=results,
        common_extractor_modified=False,
        labels_modified=False,
        adopted=False,
        independent_final_test=False,
        interpretation_ko="같은 픽셀이라도 VIDEO 추적 문맥으로 좌표가 변하는지 확인한다. 변화가 곧 정확도 개선을 뜻하지 않으며 라벨에 맞는 문맥을 선택하지 않는다.",
    )
    audit.save_json(audit.OUT / "plank_context_trial.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            (
                r["id"],
                [
                    (
                        v["context_seconds"],
                        v["cold_levels"],
                        v["context_levels"],
                    )
                    for v in r["variants"]
                ],
            )
            for r in run()["records"]
        ]
    )
