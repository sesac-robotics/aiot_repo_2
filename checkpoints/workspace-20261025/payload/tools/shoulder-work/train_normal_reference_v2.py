"""Normal-only research reference; deliberately no grade, score or error probability."""
from pathlib import Path
import os,json,sys,collections,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/shoulder-work'));from phase_features_v3 import extract,VERSION
sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
O=B/'output/shoulder-work-20261024/normal-reference-v2';O.mkdir(exist_ok=True)
assert not (O/'inputs.frozen.json').exists() and not (O/'normal-reference.joblib').exists()
R=B/'output/shoulder-work-20261024/full-review-v1';labels=json.loads((R/'dense-visual-labels.json').read_text());inv={r['id']:r for r in json.loads((R/'inventory.frozen.json').read_text())['sources']};dims={r['id']:r for r in json.loads((R/'render-receipts.json').read_text())};cache=collections.defaultdict(list)
for p in (B/'output').glob('**/primary.json'):
 if 'shoulder' not in str(p) or any(s in str(p) for s in ['/snapshots/','/before/','/code-snapshot/']):continue
 try:
  d=json.loads(p.read_text());c=d.get('config',{});sha=c.get('source_sha256');zpath=p.with_suffix('.npz')
  if sha not in {s['source_sha256'] for s in inv.values()} or not zpath.exists():continue
  z=np.load(zpath)
  if str(z['fingerprint'])!=d.get('fingerprint') or c.get('sample_hz')!=10:continue
  cache[sha].append((p,d,z['frame_numbers'],z['image_xy']))
 except (ValueError,KeyError):continue
rows=[];holds=[]
for l in labels:
 s=inv[l['id']]
 if s['view'] not in ['front','diagonal'] or l['lower_return_grade']!=2:continue
 found=0
 for a,b in l['observed_events']:
  success=None
  for p,d,fs,xy in sorted(cache[s['source_sha256']],key=lambda t:len(t[2]),reverse=True):
   fps=d['config']['segment']['fps'];times=fs/fps
   if times.min()>a-.5 or times.max()<a:continue
   for context in [3,2,1.5,4,5]:
    mask=(times>=max(0,a-context))&(times<=b+.1);slots={}
    for t,x in zip(times[mask],xy[mask]):
     tick=int(round((t-times[0])*10))+int(round(times[0]*10));err=abs(t-tick/10)
     if tick not in slots or err<slots[tick][0]:slots[tick]=(err,x)
    samples=[(k,v[1]) for k,v in sorted(slots.items())];f=extract(samples,dims[s['id']]['width'],dims[s['id']]['height'])
    if f['status']!='features_ready':continue
    if not a-.14<=np.median(f['return_ticks'])/10<=b+.14:continue
    success=(p,d,f);break
   if success:break
  if not success:holds.append(dict(id=l['id'],event=[a,b],reason='no_matching_source_verified_observed_phase_v2_cache'));continue
  p,d,f=success;found+=1
  for item,key in [('lower_return','return_features'),('frontal_trunk_hold','trunk_features')]:
   if item=='frontal_trunk_hold' and l['frontal_trunk_grade']!=2:continue
   # The seated participant has a separate reference bank, never pooled into standing.
   bank=('front_oblique_seated' if 'subject_023_' in s['id'] else 'front_oblique_standing') if s['view']=='diagonal' else ('front_seated' if 'subject_023_' in s['id'] else 'front_standing')
   rows.append(dict(id=s['id']+'_'+item+'_'+str(a),source_id=s['id'],source_sha256=s['source_sha256'],sources=[s['source_sha256']],person=s['person'],recording_group=s['recording_group'],split=s['split'],item=item,bank=bank,feature=f[key],phase=f,visual_return_event=[a,b],visual_label_path=str(R/'dense-visual-labels.json'),cache=str(p.with_suffix('.npz')),cache_fingerprint=d['fingerprint']))
 if not found:holds.append(dict(id=l['id'],reason='no_phase_v2_features_for_any_reviewed_return'))
# Independent four-joint frontal body windows: arm observability is irrelevant.
for l in labels:
 s=inv[l['id']]
 if s['view']!='front' or l['frontal_trunk_grade']!=2:continue
 for windex,(a,b) in enumerate(l['reviewed_windows']):
  choices=[]
  for p,d,fs,xy in cache[s['source_sha256']]:
   times=fs/d['config']['segment']['fps'];mask=(times>=a)&(times<=b)
   t=times[mask];x=xy[mask].copy()
   if len(t)<8 or t[-1]-t[0]<.7:continue
   wh=np.array([dims[s['id']]['width'],dims[s['id']]['height']]);bad=~np.isfinite(x).all(2)|((x<wh*.02)|(x>wh*.98)).any(2);x[bad]=np.nan
   valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2));coverage=float(valid.sum()/max(len(t),round((t[-1]-t[0])*10)+1))
   if coverage<.8 or valid.sum()<8:continue
   sh=x[valid][:,[11,12]].mean(1);hip=x[valid][:,[23,24]].mean(1);tilt=np.degrees(np.arctan2(sh[:,0]-hip[:,0],hip[:,1]-sh[:,1]));baseline=np.median(tilt[:5]);f=np.r_[np.percentile(abs(tilt),[50,90,95]),np.percentile(abs(tilt-baseline),[50,90,95])]
   choices.append((valid.sum(),p,d,f,coverage,[float(t[0]),float(t[-1])]))
  if not choices:holds.append(dict(id=l['id'],item='frontal_trunk_hold_window_reference',window=[a,b],reason='no_source_verified_cache_with_80_percent_four_joint_window'));continue
  _,p,d,f,cov,times=max(choices,key=lambda x:x[0]);rows.append(dict(id=s['id']+'_body_window_'+str(windex),source_id=s['id'],source_sha256=s['source_sha256'],sources=[s['source_sha256']],person=s['person'],recording_group=s['recording_group'],split=s['split'],item='frontal_trunk_hold_window_reference',bank='front_seated' if 'subject_023_' in s['id'] else 'front_standing',feature=f.tolist(),feature_schema='frontal-four-joint-absolute-and-window-initial-relative-tilt-v1',baseline_semantics='first_five_valid_window_samples_not_initial_ready_state',coverage=cov,observed_time_range=times,visual_reviewed_window=[a,b],visual_label_path=str(R/'dense-visual-labels.json'),cache=str(p.with_suffix('.npz')),cache_fingerprint=d['fingerprint']))

# Test inputs are inventory only; no test inference or selection in this development fit.
heads={};metrics={}
def model():return make_pipeline(StandardScaler(),OneClassSVM(nu=.05,gamma='scale'))
for key in sorted({(r['item'],r['bank']) for r in rows}):
 train=[r for r in rows if (r['item'],r['bank'])==key and r['split']=='train'];val=[r for r in rows if (r['item'],r['bank'])==key and r['split']=='validation'];assert_training_allowed(train)
 people=sorted({r['person'] for r in train});name='/'.join(key)
 if len(people)<2:metrics[name]={'status':'held','reason':'fewer_than_two_Train_people','Train_rows':len(train),'Train_people':people};continue
 fit=model().fit([r['feature'] for r in train]);heads[key]=fit;loo=[]
 for person in people:
  tr=[r for r in train if r['person']!=person];te=[r for r in train if r['person']==person];m=model().fit([r['feature'] for r in tr]);pred=m.predict([r['feature'] for r in te]);loo.extend([dict(id=r['id'],person=person,normal_reference_inlier=bool(v==1)) for r,v in zip(te,pred)])
 vp=fit.predict([r['feature'] for r in val]).tolist() if val else []
 metrics[name]=dict(status='normal_reference_fitted',Train_rows=len(train),Train_people=people,development_validation_rows=len(val),development_normal_inlier_count=sum(v==1 for v in vp),leave_person_out_normal_inlier_count=sum(r['normal_reference_inlier'] for r in loo),leave_person_out_rows=len(loo),leave_person_out=loo,false_error_detection_claim=False,error_examples=0)
assert not {r['source_sha256'] for r in rows if r['split']=='train'} & {r['source_sha256'] for r in rows if r['split']!='train'}
joblib.dump(dict(schema='shoulder-normal-reference-v2',feature_version=VERSION,heads=heads,semantics='normal_training_distribution_support_only',grade_prediction_enabled=False,score_prediction_enabled=False,error_probability_enabled=False,production_eligible=False),O/'normal-reference.joblib')
(O/'inputs.frozen.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');(O/'holds.json').write_text(json.dumps(holds,ensure_ascii=False,indent=2)+'\n');(O/'metrics.json').write_text(json.dumps(dict(fixed_configuration={'nu':.05,'gamma':'scale'},metrics=metrics,roles_preserved=True,test_inference=False,production_changed=False,expert_adjudicated=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps({'rows':len(rows),'heads':list(metrics),'metrics':metrics},ensure_ascii=False))
