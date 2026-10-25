"""추가 직접 초안의 원본 구간을 Heavy 관절 캐시와 연결한다."""

import audit
from extract import extract_view


def run() -> dict:
    """W·전신 플랭크·상지 사건을 추출하고 초안의 불변 해시를 확인한다.

    Returns:
        dict: 각 과제별 캐시·원본·초안 계보. 동일 골반 대조 사건은 이미
        추출한 같은 원본 구간을 명시적으로 재사용하며 새 추출로 세지 않는다.

    Raises:
        OSError: 초안·원본·캐시 접근 실패.
        ValueError: 저장 중 초안·원본 해시 변경.
        RuntimeError: Heavy 추출 또는 자원 보호 실패.
    """
    files = {
        k: audit.OUT / n
        for k, n in [
            ("W_records", "followup_W_direct.json"),
            ("plank_records", "followup_plank_direct.json"),
            ("event_records", "followup_upper_event_direct.json"),
        ]
    }
    hashes = {k: audit.digest(p) for k, p in files.items()}
    result = dict(
        direct_sha256=hashes, W_records=[], plank_records=[], event_records=[]
    )
    for kind, path in files.items():
        for draft in audit.read_json(path)["reviews"]:
            if draft["id"] == "followup_fitness_hip_negative":
                previous = next(
                    r
                    for r in result["plank_records"]
                    if r["id"] == "followup_fitness_highhip"
                )
                extraction = dict(
                    previous,
                    id=draft["id"],
                    explicit_cache_reuse_from="followup_fitness_highhip",
                    reused=True,
                )
            else:
                view = "side" if kind == "plank_records" else "primary"
                row = dict(
                    draft,
                    required_views=(
                        ["side"]
                        if kind == "plank_records"
                        else [
                            (
                                "observed_W_shape_only"
                                if kind == "W_records"
                                else "upper_arm_event_only"
                            )
                        ]
                    ),
                )
                destination = (
                    None
                    if kind == "plank_records"
                    else audit.OUT
                    / (
                        "followup_W_cache"
                        if kind == "W_records"
                        else "followup_upper_cache"
                    )
                )
                extraction = extract_view(
                    row, view, row["video_path"], destination
                )
            result[kind].append(extraction)
            audit.save_json(audit.OUT / "followup_extraction.json", result)
    if hashes != {k: audit.digest(p) for k, p in files.items()}:
        raise ValueError("Direct drafts changed during extraction")
    result["completed"] = True
    audit.save_json(audit.OUT / "followup_extraction.json", result)
    return result


if __name__ == "__main__":
    print(run()["completed"])
