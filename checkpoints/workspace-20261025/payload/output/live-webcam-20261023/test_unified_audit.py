import unittest,json,hashlib
from pathlib import Path
from runtime_snapshot import read_source,digest
O=Path(__file__).parent.parent/'squat-unified-20261023'
class UnifiedAuditTests(unittest.TestCase):
 def test_outer_and_inner_data_selection_never_mix_person_source_groups(self):
  c=json.loads((O/'comparison.json').read_text());p=json.loads((O/'partition.json').read_text());source_groups={}
  for r in p['rows']:
   for source in r['sources']:
    if source in source_groups:self.assertEqual(source_groups[source],r['group'])
    source_groups[source]=r['group']
  for h in c['heads']:
   for fold in h.get('folds',[]):self.assertFalse(set(fold['train_groups'])&set(fold['held_groups']))
 def test_frozen_labels_are_the_ones_used_for_training(self):
  c=json.loads((O/'comparison.json').read_text());self.assertEqual(c['labels_sha256'],hashlib.sha256((O/'labels.aligned.frozen.json').read_bytes()).hexdigest());self.assertEqual(c['rubric_sha256'],hashlib.sha256((O/'rubric.json').read_bytes()).hexdigest())
  self.assertEqual(c['source_digest'],digest(read_source()))
 def test_all_raw_candidates_are_not_claimed_relabelled(self):
  receipt=json.loads((O/'verification.json').read_text());self.assertFalse(receipt['pending']['whole_corpus_fully_relabelled']);self.assertFalse(receipt['original_models_overwritten'])
  for path,sha in receipt['original_model_sha256'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)
 def test_new_visual_label_has_actual_frame_boundaries_and_unknowns(self):
  import numpy as np
  labels=json.loads((O/'new_visual_labels.json').read_text())
  for r in labels['records']:
   z=np.load(O/'observations'/r['id']/(r['view']+'.npz'));self.assertEqual(int(z['frame_numbers'][0]),r['start_frame']);self.assertLessEqual(int(z['frame_numbers'][-1]),r['end_frame']);self.assertIsNone(r['levels'][2]);self.assertFalse(labels['model_output_used_as_label'])
   if r['view']=='side':self.assertIsNone(r['levels'][3])
   else:self.assertEqual(r['levels'][:3],[None,None,None])
