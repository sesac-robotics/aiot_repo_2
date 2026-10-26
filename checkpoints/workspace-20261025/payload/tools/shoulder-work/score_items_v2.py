"""실제 재학습/개발검증이 있는 항목만 점수를 허용하는 후보 인터페이스.

이 모듈은 모델이 아니며 기본적으로 두 항목을 보류한다. 기존 Rule을 AI로 바꾸지 않는다.
"""
import numpy as np
from phase_features_v2 import VERSION,extract
WEIGHTS={'lower_return':25,'frontal_trunk_hold':20}
TARGETS={'lower_return':'observed_ready_relative_lower_return',
 'frontal_trunk_hold':'observed_frontal_trunk_stability_only'}

def score(samples,models=None,width=640,height=480):
 features=extract(samples,width,height);out=[]
 for item,weight in WEIGHTS.items():
  row={'id':item,'weight':weight,'grade':None,'points':None,'reasons':[]}
  artifact=(models or {}).get(item)
  if features['status']!='features_ready':row['reasons'].append(features['reason'])
  elif artifact is None:row['reasons'].append('no_validated_multi_grade_model')
  else:
   counts=artifact.get('training_grade_counts',{});support=sorted(int(g) for g,n in counts.items() if n>0)
   if 2 not in support or not any(g<2 for g in support):row['reasons'].append('normal_only_or_missing_normal_training')
   if artifact.get('feature_schema')!=VERSION:row['reasons'].append('feature_schema_mismatch')
   if artifact.get('criterion')!=TARGETS[item]:row['reasons'].append('criterion_mismatch')
   if artifact.get('view')!='front':row['reasons'].append('view_mismatch')
   if not artifact.get('development_gate',{}).get('passed'):row['reasons'].append('development_gate_not_passed')
   if not artifact.get('training_ids') or not artifact.get('training_source_hashes'):row['reasons'].append('missing_training_provenance')
   if not row['reasons']:
    key='return_features' if item=='lower_return' else 'trunk_features'
    value=float(artifact['model'].predict(np.asarray([features[key]],float))[0])
    if not np.isfinite(value) or value<min(support) or value>max(support):row['reasons'].append('prediction_outside_observed_grade_support')
    else:row.update(grade=value,points=value/2*weight)
  out.append(row)
 return {'feature_schema':VERSION,'features':features,'items':out,
 'all_items_supported':all(r['points'] is not None for r in out),
 'production_replacement':False,'independent_final_test':False}
