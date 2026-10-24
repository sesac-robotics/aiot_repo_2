"""로컬 모델 Adapter와 관측된 기하의 AI·명시 규칙 추론을 조립한다.

팀 API·DB·통신 계약은 변경하지 않는다. 리뷰된 구간 점수만 계산하며
새 영상의 동작 완료·구도·사람 동일성을 자동 인증하지 않는다.
"""

import hashlib
import json
import pathlib
import time

import joblib
import numpy as np

from app.scoring import features, rules
from app.scoring.sequence import PoseSequence, verify_pair

DEFAULT_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2] / "artifacts/posture_score"
)


class ScoringEngine:
    """배포와 함께 고정된 모델만 로드하는 구간 점수 계산기다.

    모델은 생성 후 재사용한다. 파일 교체·학습·카메라 접근을 하지 않으며
    모델 bundle의 JSON·SHA-256을 검증한다. 외부의 임의 pickle은 받지 않는다.
    """

    def __init__(self, bundle: pathlib.Path = DEFAULT_BUNDLE) -> None:
        """번들 메타데이터를 검증한다. 실제 모델은 운동별 최초 사용 시 로드한다.

        Args:
            bundle: 신뢰하는 로컬 모델 디렉터리. manifest.json이 필요하다.

        Raises:
            OSError: 메타데이터 접근 실패.
            ValueError: 번들 스키마·경로 또는 운동 정의 불일치.
        """
        self.bundle = pathlib.Path(bundle).resolve()
        self.manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema") != "posture-score-bundle-v1":
            raise ValueError("Unsupported score model bundle")
        self._models: dict[str, dict] = {}

    def model(self, exercise: str) -> dict:
        """운동별 모델 해시를 확인하고 메모리 캐시를 반환한다.

        Args:
            exercise: 지원 운동 코드.

        Returns:
            dict: 항목·구도별 scikit-learn 모델과 독립 규칙 경계.

        Raises:
            ValueError: 미지원 운동·경로 이탈·해시 불일치.
            OSError: 모델 파일 접근 실패.
        """
        if exercise not in features.DEFINITIONS:
            raise ValueError("Unsupported exercise")
        if exercise not in self._models:
            entry = self.manifest["exercises"][exercise]
            path = (self.bundle / entry["file"]).resolve()
            if (
                path.parent != self.bundle
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise ValueError("Packaged model path or hash mismatch")
            self._models[exercise] = joblib.load(path)
        return self._models[exercise]

    def analyze_initial_return(
        self, exercise: str, primary: PoseSequence, *, state_models: dict,
        view: str, protocol: str, attempt_closed: bool = False,
        user_stopped: bool = False,
    ) -> dict:
        """학습된 시작 자세와 복귀 순서의 실험 근거를 계산한다.

        state_models는 검증된 로컬 상태 모델 객체다. 첫 프레임을 정상으로
        가정하거나 온라인 학습하지 않는다. 상태/순서 근거는 기존 0/1/2
        품질 점수와 구분하며, 관측 누락·사용자 종료를 실패로 만들지 않는다.
        """
        from app.scoring.initial_pose_return import InitialPoseAnalyzer

        return InitialPoseAnalyzer(state_models).run(
            exercise, primary.image_xy, primary.frame_numbers / primary.fps,
            view=view, protocol=protocol, attempt_closed=attempt_closed,
            user_stopped=user_stopped,
        )

    def score_with_initial_return(
        self, exercise: str, primary: PoseSequence, *, state_models: dict,
        return_severity_models: dict, protocol: str, view: str,
        user_stopped: bool = False, **score_options,
    ) -> dict:
        """동일 구간의 기존 항목·규칙과 새 시작/복귀 AI를 함께 검증한다.

        복귀 등급은 정상/오류를 따로 학습하고 개발 기준을 통과한 정확한
        특징 버전의 모델만 사용한다. 상태 복귀 확인을 2점으로 치환하지
        않는다. 이 실험 API는 기본 score나 라이브 모델을 교체하지 않는다.
        """
        from app.scoring.return_scoring import combine_return_score

        if type(user_stopped) is not bool:
            raise ValueError('user_stopped must be a boolean')
        complete = score_options.get('complete_attempt', False)
        score = self.score(exercise, primary, view=view, **score_options)
        evidence = self.analyze_initial_return(
            exercise, primary, state_models=state_models, view=view,
            protocol=protocol, attempt_closed=complete,
            user_stopped=user_stopped,
        )
        return combine_return_score(
            score, evidence, return_severity_models,
            complete_attempt=complete, user_stopped=user_stopped,
        )

    def score(
        self,
        exercise: str,
        primary: PoseSequence,
        *,
        view: str,
        front: PoseSequence | None = None,
        paired_view_reviewed: bool = False,
        lag_frames: int = 0,
        complete_attempt: bool = False,
        permitted_items: list[bool] | None = None,
        phase_windows: dict | None = None,
        projection_reviewed: bool = False,
    ) -> dict:
        """점수 정답을 받지 않고 검토 구간의 두 점수·근거·차이를 계산한다.

        Args:
            exercise: pushups·squats·shoulder·plank·ytw.
            primary: 10Hz 픽셀 관절. 호출자와 배열을 공유하지 않는다.
            view: 직접 확인한 촬영 구도.
            front: 스쿼트 동일 수행 정면 관절. 없으면 해당 항목은 미판정.
            paired_view_reviewed: 두 구도가 같은 수행이라는 직접 확인 여부.
            lag_frames: 정면 프레임 번호에서 측면을 뺀 원본 프레임 시차.
            complete_attempt: 구간에 완결 반복·유지 시도가 있다는 직접 확인.
            permitted_items: 보이는 네 항목의 추가 승인. None은 기하 관측만 사용.
            phase_windows: 직접 확인한 YTW 단계의 원본 프레임 구간.
            projection_reviewed: Y·전경 투영 대리값을 볼 수 있는 구도 승인.

        Returns:
            dict: 0~2 항목 등급과 0~100 가중 총점 또는 None, AI-Rule 차이,
            판정 근거·학습 등급별 구간 수·미학습 등급·모델 버전·밀리초 처리 시간.
            구간 수는 참여자 수나 예측 신뢰도가 아니다. 미판정은 환산하지 않는다.
            동일 입력은 항상 같은 결과이며 시간 값만 실행마다 달라진다.

        Raises:
            ValueError: 시점·정렬·승인 형상 또는 모델 특징 차원 불일치.
            OSError: 모델 접근 실패.
        """
        if (
            any(
                type(x) is not bool
                for x in (
                    paired_view_reviewed,
                    complete_attempt,
                    projection_reviewed,
                )
            )
            or type(lag_frames) is not int
        ):
            raise ValueError(
                "Review flags must be booleans and frame offset an integer"
            )
        if phase_windows is not None:
            if not isinstance(phase_windows, dict) or any(
                k not in ("y_reach", "t_reach", "w_shape")
                or not isinstance(v, list)
                or len(v) != 2
                or any(type(x) is not int for x in v)
                or not 0 <= v[0] <= v[1]
                for k, v in phase_windows.items()
            ):
                raise ValueError(
                    "Phase windows require reviewed nonnegative "
                    "integer frame intervals"
                )
        if permitted_items is not None and (
            len(permitted_items) != 4
            or any(type(x) is not bool for x in permitted_items)
        ):
            raise ValueError("Four explicit boolean item permissions required")
        if front is not None:
            if (
                exercise != "squats"
                or view != "side"
                or not paired_view_reviewed
            ):
                raise ValueError(
                    "Paired squat views need explicit same-attempt review"
                )
            verify_pair(primary, front, lag_frames)
        model = self.model(exercise)
        started = time.perf_counter()
        notes = features.describe(
            exercise,
            primary,
            model,
            view=view,
            front=front,
            phase_windows=phase_windows,
            projection_reviewed=projection_reviewed,
        )
        feature_ms = (time.perf_counter() - started) * 1000
        identifiers, weights = features.DEFINITIONS[exercise]
        ai, rule, items = [], [], []
        ai_ms, rule_ms = 0.0, 0.0
        for i, note in enumerate(notes):
            if permitted_items is not None and not permitted_items[i]:
                note["eligible"] = False
                note["holds"].append("item_not_reviewed_as_observable")
            key = (
                i
                if exercise == "ytw"
                else (exercise, identifiers[i], note["bank"])
            )
            head = model.get(key)
            a, b = None, None
            if head is None:
                note["holds"].append("unsupported_model_view_bank")
            if note["eligible"] and head is not None:
                estimator = head["model"]
                start = time.perf_counter()
                if isinstance(estimator, dict):
                    value = estimator["model"].predict([note["measurement"]])[0]
                else:
                    if estimator.n_features_in_ != len(note["feature"]):
                        raise ValueError("Model and feature version differ")
                    value = estimator.predict([note["feature"]])[0]
                a = float(np.clip(value, 0, 2))
                ai_ms += (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                kind = head.get("explicit_rule_kind")
                if exercise == "ytw":
                    b = int(
                        rules.grades(
                            np.asarray([note["feature"]]), head["policy"], i
                        )[0]
                    )
                elif kind == "W_shape_v1":
                    b = int(
                        rules.classify(
                            np.asarray([note["feature"]]),
                            np.asarray([head["W_policy"]]),
                        )[0, 0]
                    )
                elif kind in ("plank_scalar_v1", "knee_heel_pitch_v1"):
                    b = rules.rule_grade(note, head["boundary"])
                    if (
                        b is not None
                        and kind == "knee_heel_pitch_v1"
                        and head.get("pitch_boundary") is not None
                    ):
                        good, partial = head["pitch_boundary"]
                        b = min(
                            b,
                            (
                                2
                                if note["feature"][9] <= good
                                else 1 if note["feature"][9] <= partial else 0
                            ),
                        )
                elif "rule_grade" in note:
                    b = note["rule_grade"]
                elif head["boundary"] is not None:
                    b = rules.rule_grade(note, head["boundary"])
                rule_ms += (time.perf_counter() - start) * 1000
            single_grade_training = bool(
                head
                and (
                    head.get("single_grade_model", False)
                    or len(head.get("grade_counts", {})) == 1
                )
            )
            if single_grade_training:
                note["holds"].append(
                    "single_grade_training_error_discrimination_unverified"
                )
            counts = head.get("grade_counts") if head is not None else None
            grade_counts = (
                {
                    str(g): int(counts.get(str(g), counts.get(g, 0)))
                    for g in range(3)
                }
                if counts is not None
                else None
            )
            unseen_grades = (
                [g for g in range(3) if grade_counts[str(g)] == 0]
                if grade_counts is not None
                else None
            )
            if unseen_grades and not single_grade_training:
                note["holds"].append(
                    "training_grades_missing_"
                    + "_".join(map(str, unseen_grades))
                )
            ai.append(a)
            rule.append(b)
            items.append(
                dict(
                    id=identifiers[i],
                    weight=weights[i],
                    ai_grade=a,
                    rule_grade=None if b is None else int(b),
                    ai_minus_rule=None if a is None or b is None else a - b,
                    observed=bool(note["eligible"]),
                    bank=note["bank"],
                    measured=note.get("measurement"),
                    feature=note.get("feature") if note["eligible"] else None,
                    reasons=note["holds"],
                    single_grade_training=single_grade_training,
                    training_grade_counts=grade_counts,
                    unseen_training_grades=unseen_grades,
                    window=note.get("window"),
                )
            )
        total_allowed = complete_attempt and (
            exercise != "ytw"
            or projection_reviewed
            and phase_windows is not None
            and all(k in phase_windows for k in identifiers[:3])
        )

        def total(values: list) -> float | None:
            return (
                float(sum(w * g / 2 for w, g in zip(weights, values)))
                if total_allowed and None not in values
                else None
            )

        a_total, b_total = total(ai), total(rule)
        return dict(
            schema="posture-score-local-v1",
            exercise=exercise,
            model_version=self.manifest["version"],
            feature_version=self.manifest["feature_version"],
            assessment_scope="reviewed_clip",
            items=items,
            ai_score=a_total,
            rule_score=b_total,
            ai_minus_rule=(
                None
                if a_total is None or b_total is None
                else a_total - b_total
            ),
            complete_attempt_reviewed=complete_attempt,
            requires_manual_view_review=True,
            ytw_projection_proxy=exercise == "ytw",
            expert_validated=False,
            independent_final_test=False,
            timing_ms=dict(features=feature_ms, ai=ai_ms, rule=rule_ms),
            limitations=self.manifest["exercises"][exercise]["limitations"],
        )
