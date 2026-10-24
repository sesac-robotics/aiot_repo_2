"""운동 대기·완료·관측 실패를 구분하는 상태 머신 회귀 검사."""
import unittest
import numpy as np
from movement import MovementTracker


def squat(bent=False, shift=0):
    """정면 투영의 서기·굽힘 합성 관절을 만든다. 정확도 시험용은 아니다."""
    p = np.full((33,2),np.nan)
    for hip,knee,ankle,x,side in [(23,25,27,200,-1),(24,26,28,350,1)]:
        p[hip]=[x+shift,260 if bent else 200]
        p[knee]=[x+shift+side*(60 if bent else 5),330 if bent else 300]
        p[ankle]=[x+shift,400]
    p[11]=p[23]+[0,-100];p[12]=p[24]+[0,-100]
    return p


class MovementTest(unittest.TestCase):
    def run_steps(self,t,steps,start=0):
        completed=[]
        for n,p in enumerate(steps,start):
            r=t.push(n,p)
            if r:completed.append(r)
        return completed

    def test_waiting_does_not_count(self):
        t=MovementTracker('squats')
        self.run_steps(t,[squat()]*120)
        self.assertEqual(t.count,0)

    def test_full_cycle_and_waiting(self):
        t=MovementTracker('squats')
        result=self.run_steps(t,[squat()]*10+[squat(True)]*10+[squat()]*30)
        self.assertEqual(t.count,1);self.assertEqual(len(result),1)
        self.assertLess(len(result[0]),50)

    def test_start_crouching_not_a_rep(self):
        t=MovementTracker('squats')
        self.run_steps(t,[squat(True)]*10+[squat()]*10)
        self.assertEqual(t.count,0)

    def test_incomplete_on_visibility_loss(self):
        t=MovementTracker('squats')
        self.run_steps(t,[squat()]*10+[squat(True)]*8+[np.full((33,2),np.nan)]*5+[squat()]*10)
        self.assertEqual(t.count,0)

    def test_walking_is_not_a_squat(self):
        t=MovementTracker('squats')
        self.run_steps(t,[squat()]*10+[squat(True,100)]*10+[squat()]*10)
        self.assertEqual(t.count,0)

    def test_front_squat_without_both_ankles(self):
        t=MovementTracker('squats','front')
        stand,bent=squat(),squat(True)
        stand[27:]=np.nan;bent[27:]=np.nan
        result=self.run_steps(t,[stand]*10+[bent]*10+[stand]*20)
        self.assertEqual(t.count,1)
        self.assertEqual(len(result),1)

    def test_front_view_bending_with_straight_projection(self):
        t=MovementTracker('squats','front')
        stand,bent=squat(),squat()
        bent[[11,12,23,24],1]+=40
        stand[27:]=np.nan;bent[27:]=np.nan
        self.run_steps(t,[stand]*10+[bent]*10+[stand]*20)
        self.assertEqual(t.count,1)

    def test_waiting_without_ankles_does_not_count(self):
        t=MovementTracker('squats','front');stand=squat();stand[27:]=np.nan
        self.run_steps(t,[stand]*100)
        self.assertEqual(t.count,0)

    def test_higher_standing_pose_recalibrates_without_count(self):
        t=MovementTracker('squats','front')
        low=squat();low[[11,12,23,24],1]+=24
        low[27:]=np.nan
        stand=squat();stand[27:]=np.nan
        bent=squat(True);bent[27:]=np.nan
        self.run_steps(t,[low]*10+[stand]*10)
        self.assertEqual(t.count,0)
        self.assertAlmostEqual(t.base_hip,200)
        self.run_steps(t,[bent]*8+[stand]*8,start=20)
        self.assertEqual(t.count,1)

    def test_two_reps_preserve_standing_reference(self):
        t=MovementTracker('squats','front')
        self.run_steps(t,[squat()]*8+[squat(True)]*8+[squat()]*3+[squat(True)]*8+[squat()]*8)
        self.assertEqual(t.count,2)

if __name__=='__main__':unittest.main()
