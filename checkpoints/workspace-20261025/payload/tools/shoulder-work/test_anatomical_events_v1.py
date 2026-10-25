import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from anatomical_events_v1 import AnatomicalTurnDetector
class Tests(unittest.TestCase):
    def partial(self):
        p=pose(True);p[13]=[240,215];p[15]=[235,165];p[14]=[400,215];p[16]=[405,165];return p
    def replay(self,seq,ticks=None):
        d=AnatomicalTurnDetector();out=[]
        for t,p in zip(range(len(seq)) if ticks is None else ticks,seq):
            r=d.push(t,p,t)
            if r:out.append(r)
        return out
    def test_top_start_partial_turn(self):
        out=self.replay([pose(True)]*5+[self.partial()]*4+[pose(True)]*5)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['features']['status'],'features_ready');self.assertIsNone(out[0]['grade']);self.assertFalse(out[0]['label_created'])
    def test_cutoff_is_not_return_failure(self):self.assertEqual(self.replay([pose(True)]*5+[self.partial()]*4),[])
    def test_waiting_not_exercise(self):self.assertEqual(self.replay([pose()]*30),[])
    def test_first_partial_not_overwritten_by_later_normal(self):
        out=self.replay([pose(True)]*5+[self.partial()]*4+[pose(True)]*5+[pose()]*4+[pose(True)]*5)
        self.assertEqual(len(out),2);self.assertLess(out[0]['event']['turn_frames'][1],out[1]['event']['peak_frames'][0]);self.assertGreater(out[0]['features']['features'][2],out[1]['features']['features'][2])
    def test_missing_break_resets(self):
        s=[pose(True)]*5+[self.partial()]*4+[pose(True)]*5;s[6]=np.full((33,2),np.nan);self.assertEqual(self.replay(s),[])
    def test_time_gap_not_compacted(self):
        s=[pose(True)]*5+[self.partial()]*4+[pose(True)]*5;self.assertEqual(self.replay(s,[t*2 for t in range(len(s))]),[])
    def test_elbow_extension_not_upper_arm_descent(self):
        p=pose(True);p[15]=p[11];p[16]=p[12]
        self.assertEqual(self.replay([pose(True)]*5+[p]*4+[pose(True)]*5),[])
    def test_border_or_collapsed_torso_resets(self):
        s=[p.copy() for p in [pose(True)]*5+[self.partial()]*4+[pose(True)]*5];s[6][24]=s[6][23];self.assertEqual(self.replay(s),[])
    def test_duplicate_clock_rejected(self):
        d=AnatomicalTurnDetector();d.push(0,pose(True),0)
        with self.assertRaises(ValueError):d.push(0,pose(True),0)
if __name__=='__main__':unittest.main()
