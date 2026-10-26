"""사전 예약한 새 출처를 동결 모델의 전향적 개발 확인용으로 확보한다."""

import pathlib
import shutil

import audit
from acquire_cc_youtube import run as acquire


def run() -> dict:
    """모델·규칙·라벨을 먼저 동결하고 새 세 출처를 순차 확보한다.

    Returns:
        dict: 동결 해시·세 출처의 CC 근거·다운로드·검토판 경로.
        검토판 준비만 하며 직접 검토·점수 라벨을 생성하지 않는다.

    Raises:
        OSError: 동결 모델·출력 접근 실패.
        ValueError: 예약·라이선스 조건 불일치.
        RuntimeError: 자원 보호 중단.
    """
    policy_path = audit.OUT / "autonomous_prospective_freeze.json"
    folder = audit.OUT / "snapshots/autonomous_prospective_frozen"
    folder.mkdir(exist_ok=True, parents=True)
    if not policy_path.exists():
        files = {
            "full_ai": audit.OUT / "plank_balance_trial/balanced_leaf2.joblib",
            "common_rule": audit.BASE
            / "five-exercise-comparison/rule_score.py",
            "item_policy": audit.BASE
            / "five-exercise-comparison/plank_item_observation.py",
            "motion_rule": audit.BASE
            / "five-exercise-comparison/plank_support_motion.py",
        }
        for item in [
            "trunk_line",
            "elbow_sagittal_position",
            "leg_and_body_support",
            "support_continuity",
        ]:
            files[item] = (
                audit.OUT / "plank_support_item_trial" / (item + ".joblib")
            )
        frozen = {}
        for name, path in files.items():
            dest = folder / (name + path.suffix)
            shutil.copyfile(path, dest)
            frozen[name] = dict(path=str(dest), sha256=audit.digest(dest))
        audit.save_json(
            policy_path,
            dict(
                files=frozen,
                training_labels_sha256=audit.digest(audit.OUT / "labels.json"),
                reservation_sha256=audit.digest(
                    audit.OUT / "autonomous_source_reservation.json"
                ),
                one_pass_no_selection=True,
                full_ai_variant="balanced_leaf2",
                sources_not_used_for_fitting=[
                    "4UmPQEHqP-g",
                    "dypwCVza-58",
                    "49fNNAazn3Q",
                ],
                expert_validated=False,
            ),
        )
    result = []
    for source in ["4UmPQEHqP-g", "dypwCVza-58", "49fNNAazn3Q"]:
        result.append(acquire(source, 720))
        audit.save_json(
            audit.OUT / "autonomous_prospective_acquisition.json",
            dict(records=result, preparation_only=True, labels_created=0),
        )
    return dict(records=result, freeze=str(policy_path))


if __name__ == "__main__":
    print([(r["source_id"], r["bytes"]) for r in run()["records"]])
