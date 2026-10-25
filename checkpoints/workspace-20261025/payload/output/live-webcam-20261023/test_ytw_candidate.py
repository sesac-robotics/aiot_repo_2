"""합성 시간 계약 검사: 동작 정확도 검증과 별개다."""
import unittest
import numpy as np
from ytw_candidate import YTWTracker


def points(label):
    p=np.full((33,2),np.nan)
    p[[11,12,23,24]]=[[280,200],[360,200],[290,350],[350,350]]
    if label=='Y': p[[13,14,15,16]]=[[270,150],[370,150],[260,100],[380,100]]
    elif label=='T': p[[13,14,15,16]]=[[200,200],[440,200],[120,200],[520,200]]
    else: p[[13,14,15,16]]=[[240,240],[400,240],[240,160],[400,160]]
    return p


class TestYTW(unittest.TestCase):
    def tracker(self):
        # Synthetic deterministic labels test temporal logic only.
        return YTWTracker(lambda p: ('Y' if p[15,1]==100 else 'T' if p[15,0]==120 else 'W',True))
    def feed(self, phases, tracker=None):
        tracker=tracker or self.tracker();events=[]
        for tick,label in enumerate(phases):
            result=tracker.push(tick, np.full((33,2),np.nan) if label=='?' else points(label))
            if result:events.append(result)
        return tracker,events
    def test_order_and_repeat(self):
        t,e=self.feed(list('YYYYYTTTTTWWWWW')*2)
        self.assertEqual(t.count,2);self.assertEqual(e[0]['auto_phase_windows'],{'Y':[0,4],'T':[5,9],'W':[10,14]})
        self.assertFalse(e[0]['complete_attempt_reviewed']);self.assertIsNone(e[0]['grade'])
    def test_wrong_order(self):
        self.assertEqual(self.feed('YYYYYWWWWWTTTTT')[0].count,0)
    def test_no_initial_Y(self):
        self.assertEqual(self.feed('TTTTTWWWWW')[0].count,0)
    def test_held_single_pose(self):
        for phase in 'YTW':self.assertEqual(self.feed(phase*100)[0].count,0)
    def test_truncated(self):
        t,_=self.feed('YYYYYTTTTTWWW');t.finish();self.assertEqual(t.count,0)
        self.assertEqual(t.rejected[-1]['reason'],'stream_ended_incomplete')
    def test_missing_breaks_phase_dwell(self):
        self.assertEqual(self.feed('YYYYYTT?TTWWWWW')[0].count,0)
    def test_long_missing_invalidates(self):
        self.assertEqual(self.feed('YYYYYTTTTT????WWWWW')[0].count,0)
    def test_no_one_frame_completion(self):
        self.assertEqual(self.feed('YTW')[0].count,0)
    def test_border_excludes(self):
        t=self.tracker()
        for i,label in enumerate('YYYYYTTTTTWWWWW'):
            p=points(label);p[24]=[639,350];t.push(i,p)
        self.assertEqual(t.count,0)
    def test_head_ankles_not_required(self):
        self.assertEqual(self.feed('YYYYYTTTTTWWWWW')[0].count,1)
    def test_large_gap(self):
        t,_=self.feed('YYYYYTTTTT')
        for i in range(20,25):t.push(i,points('W'))
        self.assertEqual(t.count,0)
    def test_sparse_slots_coverage(self):
        t=self.tracker()
        for i,label in enumerate('YYYYYTTTTTWWWWW'):t.push(i*2,points(label))
        self.assertEqual(t.count,0)
    def test_actual_missing_slot_denominator(self):
        t,e=self.feed("YYYYY???TTTTT???WWWWW")
        self.assertEqual(t.count,0);self.assertFalse(e)
        self.assertEqual(t.rejected[-1]["reason"],"observation_below_80_percent")
    def test_short_gap_stays_nan(self):
        t=self.tracker();events=[]
        phases="YYYYYTTTTTWWWWW"
        for i,label in enumerate(phases):
            tick=i if i<5 else i+1
            e=t.push(tick,points(label))
            if e:events.append(e)
        self.assertEqual(t.count,1)
        self.assertTrue(np.isnan(events[0]["samples"][5][1]).all())
        self.assertAlmostEqual(events[0]["observation_coverage"],15/16)
    def test_duplicate_tick_rejected(self):
        t=self.tracker();t.push(0,points('Y'))
        with self.assertRaises(ValueError):t.push(0,points('Y'))
    def test_ready_must_persist(self):
        t=YTWTracker(lambda p:('Y',False));self.feed('Y'*30,t);self.assertFalse(t.active)
    def test_side_unsupported(self):
        t=self.tracker();t.view='side';self.feed('YYYYYTTTTTWWWWW',t);self.assertEqual(t.count,0)

if __name__=='__main__':unittest.main()
