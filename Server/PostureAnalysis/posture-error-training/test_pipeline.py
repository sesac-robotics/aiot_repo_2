"""Regression checks for leakage, offline/live feature parity and visibility gates."""
from pathlib import Path
import json,unittest
import numpy as np
import torch
from features import features,resample
from inference import PostureErrorClassifier
BASE=Path(__file__).resolve().parent

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.classifier=PostureErrorClassifier()
    def test_worker_and_source_test_separation(self):
        splits=json.loads((BASE/'outputs/splits.json').read_text());sets={k:set(v['workers']) for k,v in splits.items()}
        self.assertFalse(sets['train']&sets['test'] or sets['train']&sets['validation'] or sets['test']&sets['validation'])
        source=json.loads((BASE/'data/selected.json').read_text());test_ids={r['video_id'] for r in source if r['source_split']=='test'}
        self.assertTrue(set(splits['test']['video_ids'])<=test_ids)
        self.assertFalse(set(splits['train']['video_ids'])&test_ids)
    def test_offline_and_serving_same_scores(self):
        splits=json.loads((BASE/'outputs/splits.json').read_text());vid=splits['test']['video_ids'][0]
        with np.load(BASE/'cache'/f'{vid}.npz') as cache:
            x=cache['X'];raw=cache['landmarks'];aspect=float(cache['aspect'])
        result=self.classifier.predict_landmarks(raw,aspect)
        with torch.inference_mode():expected=torch.sigmoid(self.classifier.model(torch.tensor(((x-self.classifier.mean)/self.classifier.std)[None],dtype=torch.float32))).numpy()[0]
        np.testing.assert_allclose(list(result['scores'].values()),expected,rtol=1e-5,atol=1e-6)
    def test_missing_pose_never_reports_normal(self):
        result=self.classifier.predict_landmarks(np.zeros((32,33,4),dtype=np.float32))
        self.assertEqual(result['status'],'insufficient_pose_visibility');self.assertNotIn('normal_candidate',result)
    def test_malformed_input_rejected(self):
        for array in [np.zeros((32,17,4)),np.full((32,33,4),np.nan)]:
            with self.assertRaises(ValueError):self.classifier.predict_landmarks(array)
    def test_resampling_linear_signal(self):
        x=np.array([[0,2],[10,12]],dtype=np.float32)
        np.testing.assert_allclose(resample(x,3),[[0,2],[5,7],[10,12]])
    @unittest.skipUnless((BASE/'data/squat.npz').exists(), 'Full extracted squat array is local-only; regenerate with prepare.py for normalization audit')
    def test_training_only_normalization(self):
        data=np.load(BASE/'data/squat.npz');splits=json.loads((BASE/'outputs/splits.json').read_text());train=set(splits['train']['video_ids']);idx=np.array([i for i,v in enumerate(data['ids']) if v in train]);normalization=np.load(BASE/'outputs/normalization.npz')
        np.testing.assert_allclose(normalization['mean'],data['X'][idx].mean(axis=(0,1)),atol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
