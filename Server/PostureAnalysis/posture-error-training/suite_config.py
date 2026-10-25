"""다종목 실험의 경로·대상 오류·관측 조건·설정 버전을 정의합니다. 제공자 과제와 프로젝트의 다등급 채점 과제는 같은 라벨 체계가 아닙니다.

Explicit task definitions. Missing selected source labels are closed-world negatives.
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent
SUITE = BASE / "multi"
TASKS = {
    "pushups": dict(
        name="푸시업",
        dataset="qevd",
        targets=[
            "shallow",
            "butt sinking",
            "elevated hips",
            "head down",
            "moving head",
            "side to side",
        ],
        required=[11, 12, 13, 14, 23, 24],
        excluded=[
            "not visible",
            "head moving in an out of frame",
            "insufficient",
            "sloppy",
        ],
        normal_supported=False,
        scope="Six selected QEVD error annotations; absent labels do not certify normal form.",
    ),
    "elbow_plank": dict(
        name="엘보 플랭크",
        dataset="qevd",
        targets=[
            "arms facing inward",
            "arms too wide",
            "butt sinking",
            "head down",
            "head up",
            "high hips",
            "legs and hips on the floor",
            "shoulders shrugged up toward the ears",
        ],
        required=[11, 12, 13, 14, 23, 24],
        excluded=["average"],
        normal_supported=False,
        scope="Eight selected QEVD errors; other annotations include supported variants and timing, not verified normal controls.",
    ),
    "ytw": dict(
        name="YTW",
        dataset="qevd",
        targets=["arms too low", "wrong order", "too fast", "too slow"],
        required=[11, 12, 13, 14, 15, 16, 23, 24],
        excluded=["not low enough"],
        normal_supported=False,
        scope="Exploratory source-label reproduction. Upright/bent torso variants not reviewed; no verified upright normal controls.",
    ),
    "shoulder_press": dict(
        name="숄더 프레스",
        dataset="shoulder",
        targets=[
            "arms too wide at the top",
            "incomplete upward range",
            "hands move behind the head",
            "forearms tilt inward",
            "elbow path narrower than reference",
            "uneven arm height and timing",
        ],
        required=[11, 12, 13, 14, 15, 16, 23, 24],
        excluded=[],
        normal_supported=True,
        scope="Unverified visual/source-binary drafts; small pilot only. Source test has one positive video.",
    ),
}
JOINTS = [0, 7, 8] + list(range(11, 33))
SEQUENCE_LENGTH = 32
SAMPLE_FPS = 6
CONFIG_VERSION = "multi_v2_109_visibility_masked_6fps"
