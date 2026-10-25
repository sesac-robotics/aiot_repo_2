"""Compare frozen AI, same-feature scalar rules and existing compatible rules.
All targets, source assignments, temporal windows and item permissions stay fixed.
"""
import json,sys,collections,hashlib,os
from pathlib import Path
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');BASE=Path(os.environ.get('PAIRED_AI_RUN',str(B/'output/pushup-repaired-five-v2-20261024')));O=Path(os.environ.get('PAIRED_RULE_OUTPUT',str(B/'output/ai-rule-paired-v2-20261024')));sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'team-posture-scoring/Server/AIService')]
from common import sha
from train_scores import metrics
from split_guard import POLICY,assert_training_allowed
from thresholds import predict,transform_input
from app.scoring import ytw_features,rules as native_rules

def read(p):return json.loads(Path(p).read_text())
def legacy(row,criterion,items):
 ex=row['exercise'];candidates=[i for i in items if i['item']==criterion and i['eligible'] and i['feature'] is not None]
 def scalar(bank,idx,good,partial,higher):
  item=next((i for i in candidates if i['bank']==bank),None)
  if item is None:return None,'legacy_measurement_not_observed'
  value=item['feature'][idx];g=2 if (value>=good if higher else value<=good) else 1 if (value>=partial if higher else value<=partial) else 0
  return dict(prediction=float(g),bank=bank,scalar=float(value),thresholds=[good,partial],higher=higher),'ok'
 if ex=='pushups':
  settings={'trunk_line':(1,168,150,True),'descent_range':(1,110,145,False),'return_extension':(5,160,140,True),'support':(1,.15,.05,True)}
  return scalar('side_legacy_roi_v1',*settings[criterion]) if criterion in settings else (None,'unsupported_criterion')
 if ex=='shoulder':
  settings={'overhead_reach':(5,145,100,True),'lower_return':(3,110,140,False),'bilateral_timing':(5,.08,.3,False),'frontal_trunk_hold':(5,8,15,False)}
  return scalar('front',*settings[criterion]) if criterion in settings else (None,'unsupported_criterion')
 if ex=='squats':
  if criterion=='descent_range':return scalar('side',1,.3,.8,False)
  if criterion=='trunk_balance':return scalar('front',5,8,15,False)
  return None,'legacy_rule_target_or_measurement_not_equivalent'
 if ex=='plank':
  if criterion=='trunk_line':return scalar('side_video',1,168,155,True)
  if criterion=='elbow_sagittal_position':return scalar('side_video',5,.1,.5,False)
  item=next((i for i in candidates if i['bank']==('side_support_clearance_v1' if criterion=='leg_and_body_support' else 'side_video')),None)
  if item is None:return None,'legacy_measurement_not_observed'
  f=item['feature']
  if criterion=='leg_and_body_support':
   vals=[f[1],f[8],f[15]];g=min(2 if x>=a else 1 if x>=b else 0 for x,a,b in zip(vals,[165,.12,.03],[145,.04,.005]));return dict(prediction=float(g),bank=item['bank'],measurements=vals),'ok'
  if criterion=='support_continuity':
   v=f[6]-f[0];return dict(prediction=float(2 if v<=8 else 1 if v<=20 else 0),bank=item['bank'],measurement=v),'ok'
 if ex=='ytw':
  if criterion not in ['y_reach','t_reach','w_shape']:return None,'hinge_target_does_not_equal_upright_trunk'
  from fractions import Fraction
  d=TEAM[row['id'][5:]];z=np.load(B/'output/integrated-five-20261024/team-observations'/d['source_id']/'side.npz');times=z['frame_numbers']/float(Fraction(INVENTORY[d['source_id']]['fps']));seq=z['image_xy'][(times>=d['start'])&(times<d['end'])]
  notes=ytw_features.describe_ytw(seq);w=ytw_features.refined_w(seq)
  if w:notes[2].update(w,eligible=True)
  index=['y_reach','t_reach','w_shape'].index(criterion);n=notes[index]
  if not n['eligible'] or n['feature'] is None:return None,'legacy_stage_geometry_unobserved'
  value=float(native_rules.grades(np.asarray([n['feature']]),LEGACY_YTW[index]['policy'],index)[0]);return dict(prediction=value,bank='legacy_front_projected',policy=[float(x) for x in LEGACY_YTW[index]['policy']]),'ok'
 return None,'legacy_no_compatible_rule'

TEAM={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};INVENTORY={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};LEGACY_YTW=joblib.load(B/'team-posture-scoring/Server/AIService/artifacts/posture_score/ytw.joblib')
rulebundle=read(O/'rules.frozen.json');heads={tuple(h['key']):h for h in rulebundle['heads']};assert rulebundle['source_ai_model_sha256']==sha(BASE/'score-candidates.joblib');assert rulebundle['policy_sha256']==sha(POLICY)
training=read(BASE/'score-inputs.frozen.json');by_train={r['id']:r for r in training};assert_training_allowed([by_train[i] for h in heads.values() for i in h['training_ids']]);test=read(BASE/'test-inputs.frozen.json');rows={r['id']:r for r in test};assert not {s for r in test for s in r['sources']}&{s for r in training for s in r['sources']}
ai=read(BASE/'test-evaluation.json');paired=[];legacy_predictions=[];held=[]
for p in ai['predictions']:
 r=rows[p['id']];item=next(i for i,g in zip(r['items'],r['levels']) if i['item']==p['criterion'] and i['bank']==p['bank']);key=(p['exercise'],p['criterion'],p['bank']);target=p['target'];head=heads.get(key)
 if head is None:held.append(dict(id=p['id'],exercise=p['exercise'],criterion=p['criterion'],reason='no_interpretable_same_bank_rule'))
 else:
  rule=head['rule'];assert rule['index']<len(item['feature'])-int('_aihub_transfer' in item['bank']);x=transform_input(item['bank'],item['item'],item['feature']) if rulebundle.get('geometry_transform_version') else [item['feature']];v=float(predict(rule,x)[0]);paired.append(dict(p,ai_prediction=p['prediction'],rule_prediction=v,rule=rule,same_feature_bank=True))
 old,reason=legacy(r,p['criterion'],r['items'])
 if old is None:held.append(dict(id=p['id'],exercise=p['exercise'],criterion=p['criterion'],reason=reason,method='existing_rule'))
 else:legacy_predictions.append(dict(p,ai_prediction=p['prediction'],rule_prediction=old['prediction'],legacy_rule=old))
def summarize(rs):
 def calc(method):return metrics([dict(r,prediction=r[method]) for r in rs])
 return dict(paired_items=len(rs),ai=calc('ai_prediction'),rule_base=calc('rule_prediction'))
result=dict(schema='same_test_same_target_paired_ai_rule_v1',ai_version=str(BASE),rule_version=rulebundle['schema'],new_rules_not_deployed_legacy_rules_unchanged=True,source_partition_unchanged=True,source_overlap=0,Test_targets_unchanged=True,historical_test_exposure=True,independent_final_test=False,ai_predicted_items=len(ai['predictions']),same_bank_paired_items=len(paired),existing_compatible_rule_paired_items=len(legacy_predictions),ai_held_items=ai['held_item_count'],same_bank_by_exercise={e:summarize([r for r in paired if r['exercise']==e]) for e in sorted({r['exercise'] for r in paired})},existing_rules_by_exercise={e:summarize([r for r in legacy_predictions if r['exercise']==e]) for e in sorted({r['exercise'] for r in legacy_predictions})},same_bank_by_criterion={e+'/'+c:summarize([r for r in paired if r['exercise']==e and r['criterion']==c]) for e,c in sorted({(r['exercise'],r['criterion']) for r in paired})},paired_predictions=paired,existing_rule_paired_predictions=legacy_predictions,held=held,fingerprints=dict(ai_model=sha(BASE/'score-candidates.joblib'),training_inputs=sha(BASE/'score-inputs.frozen.json'),test_inputs=sha(BASE/'test-inputs.frozen.json'),rules=sha(O/'rules.frozen.json'),policy=sha(POLICY)),selection_on_test_forbidden=True,production_replacement=False)
p=O/'comparison.frozen.json';assert not p.exists(),'Do not overwrite frozen Test comparison';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['same_bank_paired_items','existing_compatible_rule_paired_items','same_bank_by_exercise','existing_rules_by_exercise']},indent=2))
