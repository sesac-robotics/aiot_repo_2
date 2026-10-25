import json
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from unified_assessment import apply, VERSION

ROOT = Path(__file__).parent
SESSION = ROOT/'sessions/1792740109492452117'


class UnifiedAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads((SESSION/'final-state.json').read_text())
        cls.row = cls.saved['repetitions'][0]

    def test_saved_window_points_and_coaching_share_sources(self):
        for row in self.saved['repetitions']:
            before = deepcopy(row)
            result = apply(row)
            self.assertEqual(row, before)
            score = result['score_100']
            self.assertEqual(score['version'], result['coaching']['version'])
            self.assertEqual([i['weight'] for i in score['items']], [30,20,30,20])
            self.assertIsNone(score['ai']['total'])
            self.assertEqual(score['ai']['assessed_weight'], 80)
            self.assertEqual(score['rule']['assessed_weight'], 100)
            self.assertEqual(result['ai']['earned'], score['ai']['earned_subtotal'])
            self.assertEqual(result['rule']['total'], score['rule']['total'])
            self.assertAlmostEqual(score['rule']['total'], sum(i['rule_points'] for i in score['items']))
            for item, check in zip(score['items'], result['coaching']['checks']):
                self.assertEqual(item['id'], check['id'])
                self.assertEqual(item['coaching']['message'], check['message'])
                self.assertEqual(check['message'] in result['coaching']['messages'], check['attention']) if check['attention'] else None
            self.assertEqual(result['legacy_score_100'], row['score_100'])

    def test_same_scale_difference_not_mixed_total(self):
        result = apply(self.row)
        for item in result['score_100']['items']:
            a,b = item['ai_points'], item['rule_points']
            self.assertEqual(item['comparison']['difference_points'], round(a-b,1) if a is not None and b is not None else None)

    def test_active_return_is_not_completed_and_missing_not_filled(self):
        s = apply(self.row, phase='active')['score_100']
        self.assertIsNone(s['rule']['total'])
        self.assertIsNone(s['items'][1]['rule_points'])
        self.assertEqual(s['rule']['assessed_weight'],80)
        row = deepcopy(self.row)
        row['method_comparison']['item_comparisons'] = []
        s = apply(row)['score_100']
        self.assertEqual(s['ai']['assessed_weight'],0)
        self.assertEqual(s['rule']['assessed_weight'],0)
        self.assertIsNone(s['rule']['total'])

    def test_other_exercises_and_views_keep_legacy_contract(self):
        from rubric100 import assessment
        for exercise,view in [('squats','side'),('pushups','front')]:
            r=apply(self.row,exercise,view)
            self.assertEqual(r['score_100'],assessment(self.row,exercise))
            self.assertEqual(r['coaching'],self.row['coaching'])

    def test_no_feet_real_pipeline_does_not_block_scores(self):
        from server import score_samples
        from movement import MovementTracker
        z=np.load(SESSION/'pose.npz');tr=MovementTracker('squats','front')
        segment=None
        for t,x in zip(z['frame_numbers'],z['image_xy']):
            segment=tr.push(int(t),x)
            if segment:break
        a=score_samples(segment,640,480,'squats','front')
        nofeet=[]
        for t,x in segment:
            x=x.copy();x[27:]=np.nan;nofeet.append((t,x))
        b=score_samples(nofeet,640,480,'squats','front')
        for i,j in zip(a['score_100']['items'],b['score_100']['items']):
            self.assertEqual((i['ai_points'],i['rule_points']),(j['ai_points'],j['rule_points']))
        noknees=[(t,np.concatenate([x[:25],np.full((2,2),np.nan),x[27:]])) for t,x in segment]
        c=score_samples(noknees,640,480,'squats','front')['score_100']
        self.assertIsNone(c['items'][2]['ai_points'])
        self.assertIsNotNone(c['items'][3]['ai_points'])

    def test_zero_is_an_observed_score_and_no_legacy_fallback(self):
        r=deepcopy(self.row)
        for item in r['method_comparison']['item_comparisons']:
            if item['item']=='knee_inward':item.update(ai_grade=0,rule_grade=0)
        s=apply(r)['score_100']
        self.assertEqual(s['items'][2]['ai_points'],0)
        r['method_comparison']['item_comparisons']=[]
        self.assertIsNone(apply(r)['score_100']['items'][3]['ai_points'])

    def test_final_snapshot_waits_for_pending_evaluation(self):
        import server
        import threading
        import tempfile
        from unittest.mock import patch
        entered=threading.Event();release=threading.Event()
        class Pending:
            def join(self):
                entered.set()
                if not release.wait(2):raise RuntimeError('test worker timeout')
        state=dict(generation=7,finalizing=True,repetitions=[],samples=[],jpeg=None)
        with tempfile.TemporaryDirectory() as tmp, patch.object(server,'STATE',state), patch.object(server,'REP_QUEUE',Pending()):
            p=Path(tmp)
            worker=threading.Thread(target=server.finalize_session,args=(p,7))
            worker.start();self.assertTrue(entered.wait(1))
            self.assertFalse((p/'final-state.json').exists())
            state['repetitions']=[apply(self.row)]
            release.set();worker.join(2)
            self.assertFalse(worker.is_alive())
            saved=json.loads((p/'final-state.json').read_text())
            self.assertEqual(saved['repetitions'],state['repetitions'])
            self.assertFalse(state['finalizing'])


if __name__=='__main__':unittest.main()
