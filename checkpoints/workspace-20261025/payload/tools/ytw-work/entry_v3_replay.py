"""Fixed Y-entry development/recheck. Original sources/models/labels read-only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,importlib.util
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/y-entry-v3';L=B/'output/ytw-work-20261024/logo-v1';V=B/'output/ytw-work-20261024/boundary-v2';A=B/'output/live-webcam-20261023';S=A/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7'
sys.path[:0]=[str(A),str(S)]
from live_observation import mask_frame_border
from app.scoring.initial_pose_return import pose
from ytw_ready_fix_v2 import YTWTracker as ReadyFixed
from ytw_stable_entry_v3 import YTWTracker as Stable
spec=importlib.util.spec_from_file_location('ytw_saved_baseline',O/'baseline_v1.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.load(open(O/'plan.frozen.json'));refs=json.load(open(V/'references-v2.frozen.json'));assert sha(V/'references-v2.frozen.json')==plan['reference_sha256']
models=json.load(open(L/'replay.json'));inv={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'));rows=json.load(open(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'))['labels'];data={};trace_receipts=[]
for sid in ['V37','V38','V39','V40','V41','V43','V44']:
 s=inv[sid];assert policy['assignments'][sid]=='train';assert sha(s['copy'])==s['sha256'];person='r2d2_'+next(r['person'] for r in rows if r['source_id']==sid);f=next(f for f in models['folds'] if f['excluded_person']==person);assert person not in f['train_people'];assert sha(f['model'])==f['model_sha256'];heads=joblib.load(f['model'])['heads']
 cp=B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz';z=np.load(cp);fps=float(Fraction(s['fps']));ticks=np.rint(z['frame_numbers']/fps*10).astype(int);rot=abs(float(s['rotation'][0]['rotation']))%180;w,h=(s['height'],s['width']) if rot==90 else (s['width'],s['height']);xy=mask_frame_border(z['image_xy'],w,h)
 vectors=[pose('ytw',p) for p in xy];ii=[i for i,v in enumerate(vectors) if v is not None];phase=[None]*len(xy);ready=[False]*len(xy)
 if ii:
  X=[vectors[i] for i in ii];labels=heads['phase'].predict(X);states=heads['target'].predict(X)
  for i,label,state in zip(ii,labels,states):phase[i]=str(label);ready[i]=bool(state==1)
  # Verify batched deterministic inference agrees with original scalar calls.
  for i in ii[::max(1,len(ii)//10)]:assert phase[i]==str(heads['phase'].predict([vectors[i]])[0]) and ready[i]==bool(heads['target'].predict([vectors[i]])[0]==1)
 data[sid]=(ticks,xy,w,h,phase,ready)
 trace_receipts.append(dict(source_id=sid,person=person,source_sha256=s['sha256'],cache_sha256=sha(cp),fold_model_sha256=f['model_sha256'],excluded_person_not_fit=True,batched_scalar_parity=True,trace=[dict(tick=int(t),phase=p,ready=r) for t,p,r in zip(ticks,phase,ready)]))
(O/'phase-traces.json').write_text(json.dumps(trace_receipts,ensure_ascii=False,indent=2)+'\n')
def replay(sid,name):
 ticks,xy,w,h,phase,ready=data[sid];current=[0]
 def source(points):return phase[current[0]],ready[current[0]]
 cls=base.YTWTracker if name=='baseline' else ReadyFixed if name=='ready_origin_fix' else Stable
 kwargs={} if name in ['baseline','ready_origin_fix'] else dict(max_step=float(name.split('_')[1]))
 tracker=cls(source,**kwargs);events=[]
 for i,(tick,p) in enumerate(zip(ticks,xy)):
  current[0]=i;e=tracker.push(int(tick),p,w,h)
  if e:events.append({k:v for k,v in e.items() if k!='samples'})
 tracker.finish();return dict(source_id=sid,candidate=name,events=events,final=tracker.status())
def err(pred,bracket):
 lo,hi=bracket;return dict(predicted=pred,reference=bracket,min_error=round(max(lo-pred,pred-hi,0),6),worst_error=round(max(abs(pred-lo),abs(pred-hi)),6),definite_within_0_2=max(abs(pred-lo),abs(pred-hi))<=.200001)
def measure(replays,sids):
 emap={r['source_id']:r['events'] for r in replays};positive=[];negative=[];invalid=[]
 for p in refs['positive_attempts']:
  if p['source_id'] not in sids:continue
  w=p['phase_entry_seconds'];ev=[e for e in emap[p['source_id']] if w['Y'][0]-.8<=e['start_tick']/10<=w['T'][0] and w['T'][0]<=e['end_tick']/10<=w['W'][1]+1.5];a=dict(id=p['id'],matched=len(ev)==1)
  if len(ev)==1:
   e=ev[0];a['phase_errors']={k:err((e['start_tick']+e['auto_phase_windows'][k][0])/10,w[k]) for k in 'YTW'};a['terminal_error']=err(e['end_tick']/10,p['confirmation_seconds']);a['event_ticks']=[e['start_tick'],e['end_tick']]
  positive.append(a)
 for n in refs['natural_negative_episodes']:
  if n['source_id'] not in sids:continue
  a,b=n['interval_seconds'];ev=[e for e in emap[n['source_id']] if a<=e['end_tick']/10<=b];negative.append(dict(id=n['id'],kind=n['kind'],false_completion=bool(ev)))
 for sid,events in emap.items():
  for e in events:
   N=e['end_tick']-e['start_tick']+1;win=e['auto_phase_windows']
   if any(not 0<=a<=b<N for a,b in win.values()) or not win['Y'][1]<win['T'][0]<=win['T'][1]<win['W'][0]:invalid.append(dict(source_id=sid,event_ticks=[e['start_tick'],e['end_tick']],windows=win))
 phases=[v for p in positive if p['matched'] for v in p['phase_errors'].values()];ys=[p['phase_errors']['Y'] for p in positive if p['matched']];term=[p['terminal_error'] for p in positive if p['matched']]
 summary=dict(reference_attempts=len(positive),matched_attempts=sum(p['matched'] for p in positive),natural_negative_episodes=len(negative),false_completions=sum(n['false_completion'] for n in negative),invalid_phase_windows=len(invalid),Y_definite_within_0_2=sum(y['definite_within_0_2'] for y in ys),Y_worst_error_sum=round(sum(y['worst_error'] for y in ys),6),Y_minimum_error_gt_0_2=sum(y['min_error']>.200001 for y in ys),phase_definite_within_0_2=sum(v['definite_within_0_2'] for v in phases),phase_comparisons=len(phases),terminal_definite_within_0_2=sum(t['definite_within_0_2'] for t in term),terminal_comparisons=len(term),source_event_counts={s:len(e) for s,e in emap.items()})
 return dict(summary=summary,positive=positive,negative=negative,invalid_windows=invalid,replays=replays)
# Development first. Recheck candidate is fixed before these references measured.
dev={name:measure([replay(sid,name) for sid in plan['development_sources']],plan['development_sources']) for name in plan['candidates']}
base_matched=dev['baseline']['summary']['matched_attempts'];eligible=[n for n in plan['candidates'] if n!='baseline' and dev[n]['summary']['matched_attempts']==base_matched and dev[n]['summary']['false_completions']==0 and dev[n]['summary']['invalid_phase_windows']==0]
rank={'stable_0.08':0,'stable_0.10':1,'stable_0.05':2,'ready_origin_fix':3}
choice=min(eligible,key=lambda n:(dev[n]['summary']['Y_worst_error_sum'],rank[n])) if eligible else 'ready_origin_fix'
selection=dict(frozen_before_recheck=True,development_only=True,selected=choice,eligible=eligible,development_summaries={n:r['summary'] for n,r in dev.items()},plan_sha256=sha(O/'plan.frozen.json'),candidate_code_sha256={p.name:sha(p) for p in [B/'tools/ytw-work/ytw_ready_fix_v2.py',B/'tools/ytw-work/ytw_stable_entry_v3.py']})
sp=O/'selection.frozen.json'
if sp.exists():assert json.load(open(sp))==selection
else:sp.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
(O/'development.json').write_text(json.dumps(dev,ensure_ascii=False,indent=2)+'\n')
checks={name:measure([replay(sid,name) for sid in plan['recheck_sources']],plan['recheck_sources']) for name in ['baseline','ready_origin_fix',choice]}
(O/'recheck.json').write_text(json.dumps(dict(scope=plan['exposure'],selection_sha256=sha(sp),checks=checks,production_replacement=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(selection=selection,recheck={n:r['summary'] for n,r in checks.items()}),ensure_ascii=False,indent=2))
