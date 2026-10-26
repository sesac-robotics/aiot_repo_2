from pathlib import Path
import json,unittest,hashlib
O=Path(__file__).parent
def read(name):return json.loads((O/name).read_text())
class Audit(unittest.TestCase):
 def test_full_frame_coverage(self):
  m=read('full-frame-manifest.json');self.assertEqual(len(m),176);self.assertEqual(sum(r['decoded_frames'] for r in m),26458);self.assertTrue(all(r['full_decode'] and r['decoded_frames']==r['expected_frames'] and r['frame_step']==1 for r in m))
 def test_no_guessed_labels(self):
  a=read('review-decisions.json');self.assertEqual(len(a['records']),176);self.assertFalse(a['continuous_playback']);self.assertEqual(a['new_training_labels'],0);self.assertTrue(all(not r['used_for_training'] for r in a['records']))
 def test_group_and_source_separation(self):
  r=read('toe-comparison.json');cases=r['cases']
  for f in r['folds']:
   tr=[a for a in cases if a['person'] in f['train_people']];te=[a for a in cases if a['id'] in f['held_ids']];self.assertFalse(set(f['train_people'])&set(f['held_people']));self.assertFalse({s for a in tr for s in a['source']}&{s for a in te for s in a['source']})
 def test_equal_held_cases(self):
  ref={a['id'] for a in read('phase-comparison.json')['predictions']}
  for k in ['phase','toe','linear','axis']:
   r=read(k+'-comparison.json');self.assertEqual(ref,{a['id'] for a in r['predictions']});self.assertEqual(len(ref),56);self.assertFalse(r['improved']);self.assertFalse(r['independent_final_test'])
 def test_misses_not_hidden(self):
  for k in ['phase','toe','linear','axis']:
   r=read(k+'-comparison.json');ps=r['predictions'];self.assertEqual(sum(p['target']<2 and p['after']>=1.5 for p in ps),r['after']['missed_errors']);self.assertEqual(r['after']['missed_errors'],2)
 def test_preserved(self):
  v=read('verification.json');self.assertTrue(v['source_preserved']);self.assertEqual(sum(v['weights'].values()),100)
  for p,h in v['original_model_sha256'].items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),h)
if __name__=='__main__':unittest.main()
