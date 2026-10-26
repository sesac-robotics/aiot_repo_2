import unittest,json
from pathlib import Path
from source_admission_v1 import assess_source
B=Path('/home/jayden/SeSACPAI-Mid')
class Tests(unittest.TestCase):
    def setUp(self):
        self.sources={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
        self.groups={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/source-alias-audit-v1-20261025/source-groups.frozen.json').read_text())}
    def check(self,row,receipt=None):return assess_source(row,self.sources,self.groups,receipt)
    def row(self,sid):
        s=self.sources[sid];return {k:s[k] for k in ['id','source_sha256','split','person','view']}
    def test_true_review_flag_cannot_admit_validation(self):
        r=self.row('subject_004_shoulder_bad_front');r['train_eligible']=True
        a=self.check(r);self.assertFalse(a['training_source_allowed']);self.assertIn('canonical_validation_not_training',a['holds'])
    def test_role_relabel_does_not_admit_Test(self):
        r=self.row('subject_020_shoulder_bad_front');r['split']='train'
        a=self.check(r);self.assertIn('canonical_role_mismatch',a['holds']);self.assertIn('canonical_test_not_training',a['holds'])
    def test_person_or_hash_cannot_be_rewritten(self):
        r=self.row('subject_019_shoulder_bad_front');r.update(person='another',source_sha256='other')
        a=self.check(r);self.assertIn('canonical_person_mismatch',a['holds']);self.assertIn('source_hash_mismatch',a['holds'])
    def test_exact_aliases_share_group(self):
        a=self.check(self.row('subject_016_shoulder_bad_side'));b=self.check(self.row('subject_016_shoulder_good_side'))
        self.assertEqual(a['canonical_source_group'],b['canonical_source_group']);self.assertEqual(a['person_group'],b['person_group'])
    def test_view_conflict_not_resolved_by_directory_name(self):
        r=self.row('subject_023_shoulder_good_front');self.assertFalse(self.check(r)['training_source_allowed'])
        receipt=dict(source_sha256=r['source_sha256'],actual_view='front',review_status='direct_source_visual_review')
        a=self.check(r,receipt);self.assertTrue(a['training_source_allowed']);self.assertFalse(a['model_fit_approved'])
    def test_derived_source_resolves_to_parent(self):
        rows=json.loads((B/'data/error-enrichment-20261024/annotations/segments.frozen.json').read_text())
        shoulder=[r for r in rows if r['exercise']=='shoulder']
        results=[self.check(r) for r in shoulder]
        self.assertEqual(sum(r['training_source_allowed'] for r in results),7)
        self.assertEqual(sum(not r['training_source_allowed'] for r in results),3)
        self.assertTrue(all(not r['model_fit_approved'] for r in results))
    def test_unknown_source_does_not_default_train(self):
        self.assertFalse(self.check(dict(id='unknown',split='train',train_eligible=True))['training_source_allowed'])
if __name__=='__main__':unittest.main()
