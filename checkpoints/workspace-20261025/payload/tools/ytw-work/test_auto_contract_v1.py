import sys,unittest,copy
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from test_ytw_candidate import TestYTW
from auto_contract_v1 import create,existing_engine_options
class ContractTests(unittest.TestCase):
 def event(self):return TestYTW().feed('YYYYYTTTTTWWWWW')[1][0]
 def test_no_review_or_score_permission(self):
  d=create(self.event(),state_model_sha256='a'*64,source_sha256='b'*64)
  self.assertFalse(any(d['permissions'].values()));self.assertFalse(d['validated_for_activation'])
  self.assertEqual(existing_engine_options(d),dict(view='front',complete_attempt=False,phase_windows=None,projection_reviewed=False))
 def test_reversed_windows(self):
  e=self.event();e['auto_phase_windows']['T']=[0,4]
  with self.assertRaises(ValueError):create(e,state_model_sha256='a'*64,source_sha256='b'*64)
 def test_sparse_input(self):
  e=self.event();e['samples'].pop(5)
  with self.assertRaises(ValueError):create(e,state_model_sha256='a'*64,source_sha256='b'*64)
 def test_elevated_permission(self):
  d=create(self.event(),state_model_sha256='a'*64,source_sha256='b'*64);d['permissions']['complete_attempt_reviewed']=True
  with self.assertRaises(ValueError):existing_engine_options(d)
 def test_hash_required(self):
  with self.assertRaises(ValueError):create(self.event(),state_model_sha256='not-a-hash',source_sha256='b'*64)
 def test_observation_mismatch(self):
  e=self.event();e['observation_coverage']=.9
  with self.assertRaises(ValueError):create(e,state_model_sha256='a'*64,source_sha256='b'*64)
 def test_border_unsafe(self):
  e=self.event();e["samples"][0][1][24]=[639,350]
  with self.assertRaises(ValueError):create(e,state_model_sha256="a"*64,source_sha256="b"*64)
 def test_wrong_protocol(self):
  with self.assertRaises(ValueError):create(self.event(),state_model_sha256='a'*64,source_sha256='b'*64,protocol='hinged_YTW')
if __name__=='__main__':unittest.main()
