"""실험 점수에서 관측 불가·미검증 항목이 정상 점수로 바뀌지 않는지 검사한다."""
import unittest
import numpy as np
from predict import predict


class PredictionContractTest(unittest.TestCase):
    """결측·잘못된 입력·종합 점수 보류 계약을 확인한다."""

    def test_missing_pose_is_withheld(self):
        """관절이 전부 없으면 학습 평균으로 대신 채점하지 않아야 한다."""
        for exercise in ['pushups','squats']:
            r=predict(np.full(1836,np.nan),exercise)
            self.assertIsNone(r['total_score'])
            self.assertTrue(all(item['predicted_points'] is None for item in r['items']))

    def test_diagnostics_do_not_turn_into_scores(self):
        """진단 모드에서도 검증 미달 예측을 항목 점수·총점으로 바꾸지 않아야 한다."""
        r=predict(np.ones(1836),'pushups',image_vector=np.ones(112),diagnostics=True)
        self.assertIsNone(r['total_score'])
        self.assertTrue(any(i['diagnostic_grade'] is not None for i in r['items']))
        self.assertTrue(all(i['predicted_points'] is None for i in r['items']))

    def test_unknown_exercise_and_shape_are_rejected(self):
        """운동·입력 형식 오류를 다른 운동의 점수로 처리하지 않아야 한다."""
        with self.assertRaises(ValueError):predict(np.zeros(1836),'plank')
        with self.assertRaises(ValueError):predict(np.zeros(100),'squats')

    def test_disabled_items_never_receive_default_scores(self):
        """관절이 있어도 검증 기준 미달 항목은 보류되어야 한다."""
        r=predict(np.ones(1836),'pushups')
        by_id={item['id']:item for item in r['items']}
        for item in by_id.values():self.assertIsNone(item['predicted_points'])
        self.assertIsNone(r['total_score'])


if __name__=='__main__':unittest.main()
