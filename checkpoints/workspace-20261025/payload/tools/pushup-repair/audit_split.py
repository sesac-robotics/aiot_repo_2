"""Audit reviewed person/quality splits without changing any assignment."""
import json,collections,hashlib
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-repaired-five-v2-20261024';P=B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json';policy=json.loads(P.read_text());labels=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];inputs=json.loads((O/'score-inputs.frozen.json').read_text())+json.loads((O/'test-inputs.frozen.json').read_text()); operational={r['id'][5:]:r['levels'][:4] for r in inputs if r['family']=='R2D2'};counts=collections.Counter();cells=collections.Counter();criteria=collections.Counter();sources=collections.defaultdict(set)
for r in labels:
 s=policy['assignments'][r['source_id']];g=operational.get(r['id'],[]);q='error' if any(x is not None and x<2 for x in g) else 'normal' if len(g)==4 and all(x==2 for x in g) else 'unknown';counts[r['person'],q,s]+=1;cells[r['person'],r['exercise'],q,s]+=1;sources[s].add(r['source_sha256'])
 for item in r['items']:
  if item['grade'] is not None:criteria[r['person'],r['exercise'],item['id'],'normal' if item['grade']==2 else 'error',s]+=1
assert not sources['train']&sources['test'];assert len(sources['train']|sources['test'])==57
people=[]
for p in ['P01','P02','P03','P04']:
 row={'person':p}
 for q in ['normal','error','unknown']:
  a=counts[p,q,'train'];b=counts[p,q,'test'];row[q]={'train':a,'test':b,'test_fraction':b/(a+b) if a+b else None}
  if q!='unknown':assert a and b
 people.append(row)
gaps=[]
for p in ['P01','P02','P03','P04']:
 for ex in ['squats','pushups','shoulder','ytw','plank']:
  for q in ['normal','error']:
   for s in ['train','test']:
    if cells[p,ex,q,s]==0:gaps.append(dict(person=p,exercise=ex,quality=q,split=s))
for ex in ['squats','pushups','shoulder','ytw','plank']:
 for q in ['normal','error']:assert sum(cells[p,ex,q,'test'] for p in ['P01','P02','P03','P04'])>0
train=json.loads((O/'score-inputs.frozen.json').read_text());test=json.loads((O/'test-inputs.frozen.json').read_text());assert not {s for r in train for s in r['sources']}&{s for r in test for s in r['sources']}
result=dict(policy=str(P),policy_sha256=hashlib.file_digest(P.open('rb'),'sha256').hexdigest(),reviewed_team_segments=len(labels),assigned_train=sum(v for (*_,s),v in counts.items() if s=='train'),assigned_test=sum(v for (*_,s),v in counts.items() if s=='test'),people=people,whole_source_overlap=0,all_five_test_has_normal_and_error=True,person_exercise_quality_missing_cells=gaps,criterion_support=[dict(person=p,exercise=e,criterion=i,quality=q,split=s,n=n) for (p,e,i,q,s),n in sorted(criteria.items())],all_training_candidate_families=dict(collections.Counter(r['family'] for r in train if r['split']=='train')),test_targets_unchanged=json.loads((B/'output/label-verified-five-20261024/test-inputs.frozen.json').read_text())==test,test_has_historical_exposure=True,exact_80_20_achieved=False,quality_definition='First four operational item targets; unsupported variants have unknown class. Direct visual labels retained separately.',ratio_deviation_reason='Frozen source groups and global exercise-class coverage retained; no neighboring windows split across Train/Test.')
(O/'split-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['criterion_support','person_exercise_quality_missing_cells']},indent=2))
