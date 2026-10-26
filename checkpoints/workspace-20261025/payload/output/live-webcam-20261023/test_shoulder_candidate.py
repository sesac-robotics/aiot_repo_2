import unittest
import numpy as np
from shoulder_candidate import ShoulderTracker

def pose(high=False,curl=False,one=False):
    p=np.full((33,2),np.nan)
    for side,s in [(0,1),(1,-1)]:
        x=270+side*100
        p[11+side]=[x,240];p[23+side]=[x,390]
        raised=high and not (one and side==1)
        p[13+side]=[x-s*20,170] if raised else [x-s*50,280]
        p[15+side]=[x-s*40,100] if raised else [x-s*50,220]
        if curl:p[13+side]=[x,300];p[15+side]=[x,370]
    return p

def replay(poses,ticks=None):
    t=ShoulderTracker();segments=[]
    for k,p in zip(range(len(poses)) if ticks is None else ticks,poses):
        seg=t.push(k,p)
        if seg:segments.append(seg)
    return t,segments

class Tests(unittest.TestCase):
    def test_complete(self):
        t,s=replay([pose()]*6+[pose(True)]*4+[pose()]*3)
        self.assertEqual(t.count,1);self.assertEqual(len(s[0]),13)
    def test_no_head_or_ankles_required(self):
        self.assertEqual(replay([pose()]*6+[pose(True)]*3+[pose()]*3)[0].count,1)
    def test_curl_not_press(self):
        self.assertEqual(replay([pose()]*6+[pose(curl=True)]*5+[pose()]*4)[0].count,0)
    def test_one_arm_not_bilateral(self):
        self.assertEqual(replay([pose()]*6+[pose(True,one=True)]*5+[pose()]*4)[0].count,0)
    def test_incomplete(self):
        self.assertEqual(replay([pose()]*6+[pose(True)]*8)[0].count,0)
    def test_starts_high(self):
        self.assertEqual(replay([pose(True)]*6+[pose()]*3)[0].count,0)
    def test_border(self):
        p=pose();p[23,1]=490
        self.assertEqual(replay([p]*6+[pose(True)]*4+[pose()]*3)[0].count,0)
    def test_missing_resets(self):
        p=pose(True);p[15]=np.nan
        self.assertEqual(replay([pose()]*6+[pose(True)]*3+[p]+[pose()]*3)[0].count,0)
    def test_sparse_ready(self):
        seq=[pose()]*6+[pose(True)]*4+[pose()]*3
        self.assertEqual(replay(seq,[i*2 for i in range(len(seq))])[0].count,0)
    def test_duplicate_rejected(self):
        t=ShoulderTracker();t.push(1,pose())
        with self.assertRaises(ValueError):t.push(1,pose())
    def test_gap_no_return(self):
        seq=[pose()]*6+[pose(True)]*3+[pose()]*3
        self.assertEqual(replay(seq,list(range(9))+[15,16,17])[0].count,0)
    def test_two_reps(self):
        seq=([pose()]*6+[pose(True)]*3+[pose()]*3)*2
        self.assertEqual(replay(seq)[0].count,2)
if __name__=='__main__':unittest.main()
