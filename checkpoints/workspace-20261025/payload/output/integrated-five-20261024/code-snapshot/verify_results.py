import collections,sys,json
import numpy as np,joblib
from common import B,O,read,save,sha

def main():
 summary=read(O/'summary.json');rows=read(O/'score-inputs.frozen.json');score=read(O/'score-training.json');assert score['completed'];bundle=joblib.load(O/'score-candidates.joblib');assert bundle['labels_sha256']==sha(O/'score-inputs.frozen.json');assert len(read(O/'team-extraction.json'))==57
 bygroup=collections.defaultdict(set)
 for r in rows:bygroup[r['group']].add(r['split'])
 assert all(len(s)==1 for s in bygroup.values());assert all(r['split'] in ('train','validation') for r in rows)
 for key,h in bundle['heads'].items():
  tr=[r for r in rows if r['id'] in h['trained_ids']];assert all(r['split']=='train' for r in tr);assert len(tr)==len(h['trained_ids']);assert len(h['grade_support'])>=2
  for r in tr:
   i=next(i for i,n in enumerate(r['items']) if (r['exercise'],n['item'],n['bank'])==key);assert r['levels'][i] is not None and r['items'][i]['eligible'];x=r['items'][i]['feature'];assert len(x)==h['feature_dimensions'] and np.isfinite(x).all()
  loaded=h['model'];candidate=next(r for r in score['heads'] if tuple(r['key'])==key);assert loaded.n_features_in_==candidate['feature_dimensions']
 for p in summary['input_provenance']:assert sha(p['path'])==p['sha256']
 for p in summary['models']:assert sha(B/p['path'])==p['sha256']
 native=read(O/'qevd-native-inputs.json');catalog={r['id']:r for r in read(B/'mediapipe-pilot/qevd_coach/catalog.json')['items']};assert all(catalog[r['id']]['split']=='train' for r in native)
 ai=read(O/'aihub-native-inputs.json');train={r['person'] for r in ai if r['split']=='train'};val={r['person'] for r in ai if r['split']=='validation'};assert not train&val
 neural=read(O/'return-neural-comparison.json');assert neural['full_training_epochs']>1;assert not neural['final_test_used']
 # Confirm score inference can run without annotation targets or provider strings.
 from score_video import score as predict
 z=np.load(O/'team-observations/V01/side.npz');from team_features import height_features
 result=predict('squats',z['image_xy'],'front',3.15,1080,1920,bundle,height_features.reference(z['image_xy']),True);assert len(result['items'])==4;assert all(not i['rule_used_as_AI'] for i in result['items']);save('inference-smoke.json',result)
 verification=dict(status='passed',source_observations=57,score_records=len(rows),trained_score_heads=len(bundle['heads']),models_verified=len(summary['models']),input_hashes_match=True,no_person_source_component_crosses_roles=True,no_final_test_used=True,no_null_label_filled=True,no_rule_output_used_as_AI=True,neural_training_more_than_one_epoch=True,aihub_train_validation_people_disjoint=True,qevd_official_split_preserved=True,production_replaced=False,independent_accuracy_certified=False)
 save('verification.json',verification);print(json.dumps(verification,ensure_ascii=False))
if __name__=='__main__':main()
