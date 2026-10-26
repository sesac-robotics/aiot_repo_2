"""Use actual stored pixels/observations and native AI encoders as extra features.
New features never change grade labels or their observation permissions.
"""
from pathlib import Path
import sys,json,copy,hashlib,collections,joblib,numpy as np
from common import B,O,read,save,sha
sys.path.insert(0,str(B/'tools/integrated-five'));from team_features import describe
sys.path.insert(0,str(B/'five-exercise-comparison'));import recovery_features
from cache_sequences import Resolver
from pose_features import arm_pose
from aihub_transfer import describe_common,JOINTS

def matches(old,new):
 for a in old:
  if not a['eligible'] or a['feature'] is None:continue
  for b in new:
   if a['item']!=b['item'] or b['feature'] is None:continue
   if len(a['feature'])==len(b['feature']) and np.allclose(a['feature'],b['feature'],rtol=1e-5,atol=1e-4):return True
 return False

def main():
 rows=read(O/'score-inputs.frozen.json');save('score-inputs-no-transfer.frozen.json',rows);resolver=Resolver();enc=joblib.load(O/'aihub-transfer-encoders.joblib')['models'];audit=[]
 for r in rows:
  if r['exercise'] not in ['pushups','plank','ytw','shoulder']:continue
  candidates=resolver.resolve(r);selected=None
  trusted=r['family']=='R2D2' or r['origin']=='error_enrichment'
  for seq,p in candidates:
   view='front' if any(n['bank'].startswith('front') for n in r['items']) else 'side'
   try:notes=recovery_features.describe_ytw(seq) if r['exercise']=='ytw' else describe(r['exercise'],seq,view,max(1,len(seq)/10))
   except (ValueError,IndexError):continue
   if trusted or matches(r['items'],notes):selected=seq,p;break
  if selected is None:audit.append(dict(id=r['id'],status='not_added_exact_source_or_feature_parity_unresolved'));continue
  seq,p=selected;added=[]
  if r['exercise']=='ytw' and r['family']!='R2D2' and r['origin']!='error_enrichment':
   f=arm_pose(seq)
   for i,name in enumerate(['y_reach','t_reach','w_shape']):
    forbidden=any('direct_review_item_unobservable' in h or 'tracking_unreliable' in h for h in r['items'][i].get('holds',[]))
    if f is None or r['levels'][i] is None or forbidden or not r.get('whole_attempt_observed'):continue
    bank='hinged_arm_pose_v2' if r['items'][i]['bank']=='front' else 'hinged_arm_pose_projected_v2';r['levels'].append(r['levels'][i]);r['items'].append(dict(item=name,bank=bank,feature=f,eligible=True,holds=[],features_contain_no_rule_grades=True));added.append(name+':'+bank)
  elif r['exercise'] in ['plank','pushups']:
   f=describe_common(seq[:,JOINTS])
   if f is not None:
    tasks={'trunk_line':'plank_alignment'} if r['exercise']=='plank' else {'trunk_line':'pushup_trunk','descent_range':'pushup_descent'}
    original=list(zip(r['items'],r['levels']))
    for n,g in original:
     if n['item'] not in tasks or not n['eligible'] or g is None or n['feature'] is None or '_aihub_transfer' in n['bank']:continue
     task=tasks[n['item']];m=enc[task];prob=float(m.predict_proba([f])[0,list(m.classes_).index(1)]);r['levels'].append(g);bank=n['bank']+'_aihub_transfer_v1';r['items'].append(dict(item=n['item'],bank=bank,feature=n['feature']+[prob],eligible=True,holds=[],native_condition_encoder=task,pretraining='AIHub_Training_only',provider_label_not_used_as_project_target=True));added.append(n['item']+':'+bank)
  elif r['exercise']=='shoulder':
   f=arm_pose(seq)
   for i in [0,2]:
    n=r['items'][i];g=r['levels'][i]
    if f is None or g is None or not n['eligible']:continue
    bank='front_press_arm_pose_v2';r['levels'].append(g);r['items'].append(dict(item=n['item'],bank=bank,feature=f,eligible=True,holds=[],features_contain_no_rule_grades=True));added.append(n['item']+':'+bank)
  audit.append(dict(id=r['id'],status='features_added' if added else 'retained_without_new_features',added=added,cache=str(p),cache_sha256=sha(p),source_hash_verified=True,original_interval_verified=trusted or matches(r['items'],notes)))
 save('score-inputs.frozen.json',rows);save('feature-transfer-audit.json',dict(records=audit,native_training_only=True,grades_unchanged=True,feature_schema='geometry_plus_native_AI_probability_not_rule_output',counts=dict(collections.Counter(a['status'] for a in audit))));print('AUGMENTED',collections.Counter(a['status'] for a in audit),flush=True)
if __name__=='__main__':main()
