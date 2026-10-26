import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/jayden/SeSACPAI-Mid/output/live-webcam-20261023')
from test_ytw_candidate import points
from ytw_ready_fix_v2 import YTWTracker as Fixed
from ytw_hold_onset_v4 import YTWTracker as Refined

class EntryTests(unittest.TestCase):
 def feed(self,labels='YYYYYTTTTTWWWWW',ready_false=(),missing=(),moving=(),cls=Refined):
  now=[0];current=['Y']
  def source(p):return current[0],now[0] not in ready_false
  t=cls(source);events=[];history=[]
  for i,label in enumerate(labels):
   now[0]=i;current[0]=label;p=points(label)
   if i in missing:p[:]=np.nan
   if i in moving:p[[15,16],0]+=40 if i%2 else -40
   e=t.push(i,p);history.append(t.status())
   if e:events.append(e)
  return t,events,history
 def test_ready_false_run_origin(self):
  t,e,h=self.feed('YYYYYYTTTTTWWWWW',ready_false=(0,),cls=Fixed)
  self.assertEqual(t.count,1);self.assertEqual(e[0]['start_tick'],1);self.assertEqual(e[0]['auto_phase_windows']['Y'][0],0)
 def test_refined_entry_is_actual_confirmation_tick(self):
  t,e,h=self.feed();self.assertEqual(t.count,1);self.assertEqual(e[0]['start_tick'],2)
  self.assertEqual(e[0]['Y_boundary_evidence']['confirmed_tick'],2);self.assertFalse(e[0]['Y_boundary_evidence']['backdated'])
  self.assertIsNone(h[1]['Y_stable_tick']);self.assertEqual(h[2]['Y_stable_tick'],2)
 def test_ready_release_clears_motion(self):
  t,e,h=self.feed('YYYYYYYYTTTTTWWWWW',ready_false=(2,))
  self.assertEqual(t.count,1);self.assertEqual(e[0]['start_tick'],5)
  self.assertIsNone(h[2]['Y_stable_tick']);self.assertIsNone(h[3]['Y_stable_tick'])
 def test_missing_does_not_bridge_stability(self):
  t,e,h=self.feed('YYYYYYYYTTTTTWWWWW',missing=(2,))
  self.assertEqual(t.count,1);self.assertEqual(e[0]['start_tick'],5)
 def test_gap_clears_stability(self):
  t=Refined(lambda p:('Y',True));t.push(0,points('Y'));t.push(1,points('Y'));t.push(3,points('Y'))
  self.assertIsNone(t.onset_tick);self.assertEqual(t.stable_count,0)
 def test_no_stability_withholds_boundary_preserves_order(self):
  t,e,h=self.feed('Y'*10+'T'*5+'W'*5,moving=tuple(range(10)))
  self.assertEqual(t.count,1);self.assertIsNone(e[0]['auto_phase_windows']);self.assertFalse(e[0]['Y_boundary_evidence']['confirmed'])
  self.assertFalse(e[0]['complete_attempt_reviewed']);self.assertIsNone(e[0]['grade'])
 def test_no_unlearned_frame_phase_inference(self):
  t=Refined(lambda p:('T',True))
  for i in range(50):t.push(i,points('Y'))
  self.assertEqual(t.count,0);self.assertIsNone(t.onset_tick)
 def test_no_future_detection(self):
  for labels in ['YYYYY','YYYYYTTTTT','YYYYYTTTTTWWWW']:
   self.assertEqual(self.feed(labels)[0].count,0)
 def test_new_attempt_does_not_keep_previous_velocity(self):
  t,e,h=self.feed('YYYYYTTTTTWWWWW'*2);self.assertEqual(t.count,2)
  self.assertEqual([x['start_tick'] for x in e],[2,17]);self.assertEqual([x['Y_boundary_evidence']['confirmed_tick'] for x in e],[2,17])
 def test_all_windows_inside_actual_segment(self):
  t,e,h=self.feed();event=e[0];N=len(event['samples'])
  self.assertTrue(all(0<=a<=b<N for a,b in event['auto_phase_windows'].values()))
  self.assertEqual(event['samples'][0][0],event['start_tick'])
 def test_refined_segment_must_also_observe_80_percent(self):
  labels="Y"*18+"T"*8+"W"*5
  t,e,h=self.feed(labels,missing=(15,16,17,23,24,25),moving=tuple(range(10)))
  self.assertEqual(t.count,1);self.assertIsNone(e[0]["auto_phase_windows"])
  self.assertEqual(e[0]["Y_boundary_evidence"]["reason"],"refined_segment_observation_below_80_percent")
 def test_no_correct_shape_required(self):
  # Classification remains source-owned, stability must not impose elbow angles.
  i=[0];t=Refined(lambda p:('Y' if i[0]<5 else 'T' if i[0]<10 else 'W',True));events=[]
  for j in range(15):
   i[0]=j;e=t.push(j,points('W'))
   if e:events.append(e)
  self.assertEqual(t.count,1);self.assertIsNone(events[0]['grade'])
 def test_wrong_order_missing_and_border(self):
  self.assertEqual(self.feed('YYYYYWWWWWTTTTT')[0].count,0)
  self.assertEqual(self.feed('YYYYYTTTTTWWWWW',missing=(5,6,7,8))[0].count,0)
  t=Refined(lambda p:('Y',True));p=points('Y');p[24]=[639,350]
  for i in range(20):t.push(i,p)
  self.assertEqual(t.count,0)
if __name__=='__main__':unittest.main()
