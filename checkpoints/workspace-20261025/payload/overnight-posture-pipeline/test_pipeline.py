"""참여자 누출·초안 형식·점수 분포 부족을 차단하는 검사를 수행한다."""
import unittest
import numpy as np
from teacher import acceptable
from train import assert_splits,score_gate
from quality import observation_ok,GROUPS


class PipelineChecks(unittest.TestCase):
    """형식상 그럴듯한 잘못된 라벨이 학습에 들어가지 않는지 확인한다."""

    def test_invalid_score_rejected(self):
        """관측 불가·범위 밖·NaN·신뢰도 부족 점수를 제외한다."""
        good=dict(assessable=True,score=80,confidence=.8,evidence_en='Elbows bend with a visible straight trunk.')
        self.assertTrue(acceptable(good))
        for delta in [dict(assessable=False),dict(score=120),dict(score=float('nan')),dict(confidence=.5),dict(evidence_en='')]:
            self.assertFalse(acceptable(dict(good,**delta)))

    def test_leaked_person_rejected(self):
        """한 참여자의 다른 카메라가 서로 다른 분할에 들어가면 실패한다."""
        with self.assertRaises(AssertionError):assert_splits([dict(person='p1',split='train'),dict(person='p1',split='test')])

    def test_view_direction_is_not_an_exclusion(self):
        """정면이라는 이유로 배제하지 않고 두 시점의 실제 핵심 부위 관측을 요구한다."""
        self.assertTrue(observation_ok(2,4))
        self.assertFalse(observation_ok(1,4))
        self.assertFalse(observation_ok(2,3))
        self.assertNotIn((13,14),GROUPS['squats'])
        self.assertIn((25,26),GROUPS['squats'])

    def test_constant_scores_rejected(self):
        """표본 수가 많아도 점수가 모두 같으면 회귀 학습을 차단한다."""
        rows=[dict(split=s,person=f'{s}_{i}') for s,n in [('train',20),('validation',5),('test',5)] for i in range(n)]
        self.assertFalse(score_gate(rows,[85]*len(rows))[0])
        self.assertTrue(score_gate(rows,[40+(i%5)*10 for i in range(len(rows))])[0])


if __name__=='__main__':unittest.main()
