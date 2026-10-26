from pathlib import Path
import sys,json,unittest,hashlib,collections,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025'
sys.path[:0]=[str(B/'tools/four-exercise-source-phase'),str(B/'tools/error-enrichment')]
import features as sf
from split_guard import assert_training_allowed

class SourcePhaseContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.links=json.loads((O/'all-resolved-inputs.frozen.json').read_text());cls.rows=json.loads((O/'score-inputs.frozen.json').read_text());cls.byid={r['id']:r for r in cls.rows}
 def test_all_609_native_links_and_cache_fingerprints(self):
  self.assertEqual(len(self.links),609);self.assertEqual(len({r['id'] for r in self.links}),609)
  for r in self.links:
   z=np.load(r['cache_path']);m=json.loads(Path(r['cache_path']).with_suffix('.json').read_text());self.assertEqual(str(z['fingerprint']),r['cache_fingerprint']);self.assertEqual(hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest(),r['cache_fingerprint']);self.assertEqual(hashlib.sha256(Path(r['cache_path']).read_bytes()).hexdigest(),r['cache_sha256']);self.assertEqual(z['frame_numbers'][r['cache_indices']].tolist(),r['actual_native_frames'])
   ts=np.array(r['actual_native_frames'])/r['fps'];self.assertTrue(np.all(np.diff(ts)>0));self.assertTrue(np.all(ts>=r['window_seconds'][0]-1e-6));self.assertTrue(np.all(ts<r['window_seconds'][1]+1e-6))
 def test_crop_lineage_does_not_create_a_new_independent_source(self):
  derived=[r for r in self.links if r.get('original_lineage')];self.assertEqual(len(derived),9)
  for r in derived:
   lineage=r['original_lineage'];self.assertIn(lineage['original_sha256'],r['sources']);self.assertEqual(hashlib.sha256(Path(lineage['original_path']).read_bytes()).hexdigest(),lineage['original_sha256'])
   self.assertEqual(len(lineage['pixel_checks']),3)
   for check in lineage['pixel_checks']:self.assertTrue(check['exact_pixels_equal']);self.assertEqual(check['source_frame'],check['clip_frame']+lineage['offset_frames'])
 def test_protected_sources_and_nontrain_rejected(self):
  p=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text())
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='renamed',split='train',sources=[p['protected_source_sha256'][0]])])
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='fake',split='validation',sources=[])])
  assert_training_allowed(self.rows);self.assertNotIn('squats',{r['exercise'] for r in self.rows})
 def test_tracking_hold_is_not_a_bad_posture_label(self):
  for id in ['recheck_N003_000000_002750','recheck_N003_002750_005000','recheck_N011_000000_004750']:
   r=self.byid[id];self.assertTrue(all(not x['eligible'] for x in r['items']));self.assertTrue(any(y==2 for y in r['levels']))
  r=self.byid['paired-return-qevd_00026410'];self.assertTrue(all(not x['eligible'] for x in r['items'] if x['item']=='return_extension'));self.assertEqual(r['levels'][2],2)
 def test_clipped_ascent_stays_ungraded_across_every_bank(self):
  r=self.byid['paired-return-qevd_00269873']
  for x,y in zip(r['items'],r['levels']):
   if x['item']=='return_extension':self.assertIsNone(y);self.assertFalse(x['eligible'])
 def test_stage_window_cannot_bridge_missing_source_time(self):
  self.assertEqual(sf.continuous_windows(np.ones(8,bool),np.array([0,3,6,9,60,63,66,69]),30,5),[])
  v=np.ones(8,bool);v[3]=False;self.assertEqual(sf.continuous_windows(v,np.arange(8)*3,30,5),[])
 def test_confidence_and_outside_image_are_missing_not_zero(self):
  x=np.ones((10,33,2))*30;q=np.ones((10,33,2));q[0,11,0]=.1;x[1,15,0]=101
  out=sf.sanitize(x,q,100,100);self.assertTrue(np.isnan(out[0,11]).all());self.assertTrue(np.isnan(out[1,15]).all());self.assertEqual(out[2,11].tolist(),[30,30])
 def test_cut_return_does_not_become_closed_normal(self):
  angles=[170,170,170,150,120,90,80,90,120,150,170,170,170,170];x=np.full((len(angles),33,2),np.nan)
  for i,a in enumerate(angles):
   x[i,11]=[100,100];x[i,13]=[100,150];theta=np.radians(a);x[i,15]=[100+50*np.sin(theta),150-50*np.cos(theta)]
  q=np.ones((len(x),33,2));fs=np.arange(len(x))*3
  self.assertEqual(sf.return_turn(x[:10],q[:10],fs[:10],30,100)['status'],'held')
  r=sf.return_turn(x,q,fs,30,100);self.assertEqual(r['status'],'features_ready');self.assertEqual(r['available_at_source_frame'],39);self.assertFalse(r['future_after_supplied_window_used']);self.assertGreater(r['return_source_frame'],r['bottom_source_frame'])
 def test_stage_receipts_are_inside_supplied_source_window(self):
  for r in json.loads((O/'feature-observation-receipts.json').read_text()):
   if not r['eligible'] or '_stage_contiguous_' not in r['bank']:continue
   link=next(x for x in self.links if x['id']==r['id']);fs=r['result']['window_source_frames'];self.assertEqual(len(fs),5);self.assertTrue(set(fs)<=set(link['actual_native_frames']));self.assertTrue(np.all(np.diff(fs)/link['fps']<=.16))

if __name__=='__main__':unittest.main(verbosity=2)
