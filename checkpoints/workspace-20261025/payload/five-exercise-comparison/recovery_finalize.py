"""보강 후보의 원본 구간·모델 지문·로컬 표시 파일을 최종 대조한다."""

import json
import pathlib
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

import audit
from recovery_prepare import ROOT


class LocalLinks(HTMLParser):
    """HTML의 링크 경로를 수집한다. 브라우저 렌더링 검증은 수행하지 않는다."""

    def __init__(self):
        """입력 없는 링크 수집기를 초기화한다."""
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        """시작 태그의 파일 경로를 기록한다.

        Args:
            tag: HTML 태그 이름.
            attrs: 속성 이름·값 쌍. href와 src만 사용한다.

        Returns:
            None: 로컬 경로를 paths에 누적한다.
        """
        for key, value in attrs:
            if key in ("href", "src") and value:
                parts = urlsplit(value)
                if not parts.scheme and not parts.netloc and parts.path:
                    self.paths.append(unquote(parts.path))


def run() -> dict:
    """저장된 후보와 보존 계약을 검증하고 감사 JSON을 저장한다.

    Returns:
        dict: 원본 구간 중복·파일 지문·로컬 링크 대조 결과. 실제 사람의
        동일성·전문가 정답·독립 시험·브라우저 재생을 인증하지 않는다.

    Raises:
        OSError: 입력·연결 파일 접근 또는 저장 실패.
        ValueError: 구간 중복·지문·보존 계약·링크 불일치.
    """
    fitted = audit.read_json(ROOT / "comparison.json")
    if {r["exercise"] for r in fitted["exercises"]} != {
        "pushups",
        "squats",
        "shoulder",
        "plank",
        "ytw",
    }:
        raise ValueError("Comparison is still incomplete")
    plan = audit.read_json(ROOT / "input_plan.json")["records"]
    intervals = {}
    for row in plan:
        segment = row["segment"]
        first, last = segment["start_frame"], segment["end_frame"]
        sha = row["video_sha256"]
        if any(not (last < a or first > b) for a, b in intervals.get(sha, [])):
            raise ValueError("Overlapping labeled source intervals")
        intervals.setdefault(sha, []).append((first, last))
    fingerprints = {
        "input_plan": fitted["input_plan_sha256"]
        == audit.digest(ROOT / "input_plan.json"),
        "fit_input_manifest": fitted["fit_input_manifest_sha256"]
        == audit.digest(ROOT / "fit_input_manifest.json"),
    }
    if fitted.get("variant_policy_sha256") is not None:
        fingerprints["variant_policy"] = fitted[
            "variant_policy_sha256"
        ] == audit.digest(ROOT / "variant_policy_sources.json")
    preview = audit.read_json(ROOT / "preview_manifest.json")
    fingerprints["preview_comparison"] = preview[
        "comparison_sha256"
    ] == audit.digest(ROOT / "comparison.json")
    for row in fitted["exercises"]:
        fingerprints[row["exercise"]] = row["model_sha256"] == audit.digest(
            ROOT / "models" / (row["exercise"] + ".joblib")
        )
    for name, sha in fitted["code_sha256"].items():
        fingerprints[name] = sha == audit.digest(
            pathlib.Path(__file__).parent / name
        )
    for file, script in [
        ("trunk_direction_trial/comparison.json", "recovery_trunk_trial.py"),
        ("W_calibration_trial.json", "recovery_w_calibration.py"),
        ("W_phase_trial.json", "recovery_w_phase_trial.py"),
        ("W_shape_trial.json", "recovery_w_shape_trial.py"),
    ]:
        path = ROOT / file
        if not path.exists():
            continue
        trial = audit.read_json(path)
        fingerprints[file + ":comparison"] = trial[
            "comparison_sha256"
        ] == audit.digest(ROOT / "comparison.json")
        fingerprints[file + ":code"] = trial["code_sha256"] == audit.digest(
            pathlib.Path(__file__).parent / script
        )
        if "fit_input_manifest_sha256" in trial:
            fingerprints[file + ":inputs"] = trial[
                "fit_input_manifest_sha256"
            ] == audit.digest(ROOT / "fit_input_manifest.json")
        if "phase_code_sha256" in trial:
            fingerprints[file + ":phase_code"] = trial[
                "phase_code_sha256"
            ] == audit.digest(
                pathlib.Path(__file__).with_name("recovery_w_phase.py")
            )
        if "geometry_code_sha256" in trial:
            fingerprints[file + ":geometry_code"] = trial[
                "geometry_code_sha256"
            ] == audit.digest(
                pathlib.Path(__file__).with_name("recovery_trunk_geometry.py")
            )
        if "phase_trial_sha256" in trial:
            fingerprints[file + ":phase_trial"] = trial[
                "phase_trial_sha256"
            ] == audit.digest(ROOT / "W_phase_trial.json")
    w_preview_path = ROOT / "W_preview_manifest.json"
    if w_preview_path.exists():
        w_preview = audit.read_json(w_preview_path)
        for field, path in [
            ("comparison_sha256", ROOT / "comparison.json"),
            ("phase_trial_sha256", ROOT / "W_phase_trial.json"),
            ("shape_trial_sha256", ROOT / "W_shape_trial.json"),
            ("candidate_ai_model_sha256", ROOT / "W_phase_trial_model.joblib"),
        ]:
            fingerprints["W_preview:" + field] = w_preview[
                field
            ] == audit.digest(path)
        for record in w_preview["records"]:
            fingerprints["W_video:" + record["stage"]] = record[
                "sha256"
            ] == audit.digest(pathlib.Path(record["path"]))
    preserved = audit.read_json(
        audit.BASE / "output/three-exercise-comparison-20261022/protocol.json"
    )["preserved_sha256"]
    preservation = {
        path: sha == audit.digest(pathlib.Path(path))
        for path, sha in preserved.items()
    }
    parser = LocalLinks()
    parser.feed((ROOT / "index.html").read_text(encoding="utf-8"))
    missing = [p for p in parser.paths if not (ROOT / p).exists()]
    if (
        not all(fingerprints.values())
        or not all(preservation.values())
        or missing
    ):
        raise ValueError("Artifact fingerprint, preservation or link mismatch")
    interval_report = dict(
        records=len(plan),
        unique_primary_sha256=len(intervals),
        overlapping_label_intervals=[],
        camera_views_not_counted_as_independent_people=True,
    )
    audit.save_json(ROOT / "source_interval_audit.json", interval_report)
    audit.save_json(ROOT / "preservation_check.json", preservation)
    report = dict(
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
        fingerprints=fingerprints,
        preserved_files_unchanged=all(preservation.values()),
        interval_audit=interval_report,
        local_link_references_checked=len(parser.paths),
        missing_local_links=missing,
        browser_rendering_verified=False,
        verified_real_participant_identity=False,
        expert_ground_truth=False,
        independent_final_test_complete=False,
    )
    audit.save_json(ROOT / "artifact_verification.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
