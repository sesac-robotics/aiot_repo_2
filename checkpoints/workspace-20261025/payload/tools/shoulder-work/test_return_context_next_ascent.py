import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from return_context_v1 import ReturnContextQueue
class Tests(unittest.TestCase):
    def seed(self):return list(enumerate([pose().copy() for _ in range(6)]+[pose(True).copy() for _ in range(4)]+[pose().copy() for _ in range(2)]))
    def test_early_next_ascent_not_normal_return(self):
        q=ReturnContextQueue();q.submit('a',self.seed());out=[]
        for t in [12,13,14]:out+=q.push(t,pose(True))
        self.assertEqual(len(out),1);self.assertEqual(out[0]['features']['status'],'held');self.assertIsNone(out[0]['grade'])
    def test_expired_attempt_not_closed_by_later_normal(self):
        q=ReturnContextQueue();q.submit('a',self.seed());out=q.push(15,pose(True))
        self.assertEqual(out[0]['features']['status'],'held');self.assertEqual(q.push(16,pose()),[]);self.assertEqual(q.push(17,pose()),[]);self.assertEqual(q.finish(),[])
    def test_first_partial_requires_actual_following_ascent(self):
        p=pose(True);p[13]=[240,215];p[15]=[235,165];p[14]=[400,215];p[16]=[405,165]
        seed=list(enumerate([pose().copy() for _ in range(6)]+[pose(True).copy() for _ in range(3)]+[p.copy() for _ in range(3)]))
        q=ReturnContextQueue();self.assertIsNone(q.submit('a',seed));out=[]
        for t in [12,13,14]:out+=q.push(t,pose(True))
        self.assertEqual(len(out),1);r=out[0];self.assertEqual(r['features']['closure'],'lower_turnaround_with_observed_next_ascent');self.assertEqual(r['features']['return_ticks'],[9,10,11]);self.assertEqual(r['evaluated_at_tick'],14);self.assertIsNone(r['grade'])
if __name__=='__main__':unittest.main()
