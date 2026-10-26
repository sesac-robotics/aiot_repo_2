"""Apply reviewed labels and compact geometry; preserve all source assignments."""
import collections,copy,json,sys,hashlib
from pathlib import Path
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');OLD=B/'output/per-person-stratified-80-20-20261024';O=B/'output/label-verified-five-20261024';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from cache_sequences import Resolver
from pose_features import arm_pose
from common import sha
sys.path.insert(0,str(B/'tools/integrated-five'));from team_features import describe
from features import shape,bilateral,side_body

def main():
 canonical=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];team={r['id']:r for r in canonical}
 rows=json.loads((OLD/'score-inputs.frozen.json').read_text())+json.loads((OLD/'test-inputs.frozen.json').read_text())
 replaced={'r2d2_V38_014000_020750','r2d2_V38_020750_025750'};base=copy.deepcopy(next(r for r in rows if r['id']=='r2d2_V38_014000_020750'))
 rows=[r for r in rows if r['id'] not in replaced];base['id']='r2d2_V38_014000_025750';base['replaces']=sorted(replaced);rows.append(base)
 resolver=Resolver();audit=json.loads((B/'output/error-enrichment-20261024/feature-transfer-audit.json').read_text())['records'];verified={r['id']:r['cache'] for r in audit if r.get('cache') and r.get('original_interval_verified')}
 records=[]
 def resolve(row):
  if row['family']=='R2D2':
   d=team[row['id'][5:]];p=B/'output/integrated-five-20261024/team-observations'/d['source_id']/'side.npz';z=np.load(p)
   inventory=resolver.inventory[d['source_id']];from fractions import Fraction
   times=z['frame_numbers']/float(Fraction(inventory['fps']));return z['image_xy'][(times>=d['start'])&(times<d['end'])],p
  pairs=resolver.resolve(row)
  if row['origin']=='error_enrichment':return pairs[0] if pairs else None
  target=verified.get(row['id']);return next(((seq,p) for seq,p in pairs if str(p)==target),None)
 for row in rows:
  if row['family']=='R2D2':
   d=team[row['id'][5:]];row['annotation_review']='sampled_second_visual_review_v2';row['whole_attempt_observed']=d['complete_cycle'];row['reference_total_allowed']=d['total_score'] is not None
   if row['id']==base['id']:
    resolved=resolve(row);seq,p=resolved;notes=describe('ytw',seq,'front',d['end']-d['start']);f=arm_pose(seq)
    for name in ['y_reach','t_reach','w_shape']:notes.append(dict(item=name,bank='upright_arm_pose_v2',feature=f,eligible=f is not None,holds=[]))
    row['items']=notes;grades={n:g['grade'] for n,g in zip(['y_reach','t_reach','w_shape','upright_trunk_hold'],d['items'])};row['levels']=[grades[n['item']] for n in notes]
    for item,g in zip(row['items'],row['levels']):
     if g is None:item.update(feature=None,eligible=False,holds=['mixed_Y_pose_after_boundary_review'])
  if row['exercise'] not in ['ytw','shoulder','pushups','plank']:continue
  resolved=resolve(row)
  if resolved is None:records.append(dict(id=row['id'],status='not_augmented_unresolved_original_interval'));continue
  seq,p=resolved;added=[]
  def append(index,bank,feature):
   old=row['items'][index];grade=row['levels'][index]
   if grade is None or not old['eligible'] or feature is None:return
   row['items'].append(dict(item=old['item'],bank=bank,feature=feature,eligible=True,holds=[],features_contain_no_rule_grades=True));row['levels'].append(grade);added.append(old['item']+':'+bank)
  if row['exercise']=='ytw':
   variant='upright' if row['family']=='R2D2' else 'hinged' if row['items'][0]['bank']=='front' else 'projected'
   for i,kind in enumerate(['Y','T','W']):append(i,variant+'_compact_shape_v3',shape(seq,kind))
  elif row['exercise']=='shoulder':
   append(0,'front_compact_press_v3',shape(seq,'press'));append(2,'front_compact_timing_v3',bilateral(seq))
  elif row.get('view') in ['side','side_oblique'] or row['items'][0]['bank'].startswith('side'):
   append(0,'side_body_alignment_v3',side_body(seq))
  records.append(dict(id=row['id'],status='augmented' if added else 'retained',added=added,cache=str(p),cache_sha256=sha(p),original_interval_verified=True))
 training=[r for r in rows if r['split']!='test'];test=[r for r in rows if r['split']=='test']
 assert len({r['id'] for r in rows})==len(rows)
 assert not {s for r in training for s in r['sources']} & {s for r in test for s in r['sources']}
 # Test labels and source membership cannot change during training-side improvements.
 oldtest={r['id']:r for r in json.loads((OLD/'test-inputs.frozen.json').read_text())}
 assert set(oldtest)=={r['id'] for r in test}
 for r in test:assert r['levels'][:len(oldtest[r['id']]['levels'])]==oldtest[r['id']]['levels']
 for name,value in [('score-inputs.frozen.json',training),('test-inputs.frozen.json',test),('feature-provenance.json',records)]:
  p=O/name
  if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Would overwrite frozen input '+name)
  p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
 (O/'preparation.json').write_text(json.dumps(dict(previous_run=str(OLD),team_review=str(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'),source_partition_unchanged=True,test_targets_unchanged=True,all_existing_data_retained_except_two_superseded_train_windows=True,rows=len(rows),training_and_external_validation=len(training),test_rows=len(test),augmentation_counts=dict(collections.Counter(r['status'] for r in records))),ensure_ascii=False,indent=2))
 print('PREPARED',len(rows),len(test),collections.Counter(r['status'] for r in records),flush=True)
if __name__=='__main__':main()
