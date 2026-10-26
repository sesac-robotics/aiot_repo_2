"""Describe existing Train motion evidence; no fitting or parameter selection."""
import json,sys,hashlib,datetime,collections
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/squat-count-Train-profile-20261025';R=B/'output/squat-return-20261023';S=B/'output/squat-standing-state-20261023'
sys.path.insert(0,str(B/'output/live-webcam-20261023'));sys.path.insert(0,str(B/'tools/error-enrichment'))
from live_observation import mask_frame_border
from split_guard import assert_training_allowed

def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 O.mkdir(exist_ok=True)
 labels=json.loads((R/'labels-v2.frozen.json').read_text())['labels'];labels=[r for r in labels if r['split']=='train'];assert_training_allowed(labels)
 records=[];held=[];sources={}
 for r in labels:
  name=Path(r['path']).stem;cache=R/(name+'-heavy.npz');refp=S/(name+'-reference.npz')
  if not cache.exists() or not refp.exists():held.append(dict(id=r['id'],reason='existing_cache_or_reviewed_initial_reference_missing'));continue
  if name not in sources:
   z=np.load(cache);x=mask_frame_border(z['image_xy'],int(z['width']),int(z['height']));f=z['frame_numbers'];fps=float(z['fps']);ref=np.load(refp)['image_xy'];ref=ref[np.isfinite(ref[:,[11,12,23,24]]).all((1,2))];p=np.median(ref,axis=0);h=p[[23,24]].mean(0);s=p[[11,12]].mean(0);scale=float(np.linalg.norm(s-h));ticks=np.rint(f*10/fps).astype(int);assert len(set(ticks))==len(ticks)
   sources[name]=dict(x=x,frames=f,ticks=ticks,fps=fps,hip0=h,scale=scale,source_sha256=sha(Path(r['path'])),cache_sha256=sha(cache),reference_sha256=sha(refp))
  z=sources[name];sel=(z['frames']>=r['first_frame'])&(z['frames']<=r['last_frame']);x=z['x'][sel];ticks=z['ticks'][sel]
  torso=np.isfinite(x[:,[11,12,23,24]]).all((1,2));knee=np.isfinite(x[:,[25,26]]).all(2);ok=torso&knee.any(1);both=torso&knee.all(1)
  drop=(x[:,[23,24]].mean(1)[:,1]-z['hip0'][1])/z['scale'];good=drop[ok];span=int(ticks[-1]-ticks[0]+1)if len(ticks)else 0;run=best=0;prev=None
  for tick,value,valid in zip(ticks,drop,ok):
   run=run+1 if valid and value>.08 and prev is not None and tick==prev+1 else 1 if valid and value>.08 else 0
   best=max(best,run);prev=tick
  records.append(dict(id=r['id'],source=name,source_sha256=z['source_sha256'],split='train',person=r['person'],existing_return_grade=r['grade'],grade_is_not_depth_or_completed_count_truth=True,first_frame=r['first_frame'],last_frame=r['last_frame'],actual_10Hz_slots=len(ticks),span_10Hz_slots=span,body_one_knee_valid_slots=int(ok.sum()),body_both_knees_valid_slots=int(both.sum()),observation_fraction=float(ok.sum()/span)if span else 0,hip_drop_p95_p05=float(np.ptp(np.percentile(good,[5,95])))if len(good)else None,hip_drop_max=float(np.max(good))if len(good)else None,max_consecutive_drop_over_existing_point08=best,scale_is_existing_initial_torso_not_live_tracker_baseline=True))
 out=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='existing_Train_reviewed_intervals_only_descriptive_no_threshold_selection',input_labels_sha256=sha(R/'labels-v2.frozen.json'),reference_policy='existing reviewed initial-reference files only; missing holds',records=records,held=held,sources=[dict(id=k,**{n:v for n,v in z.items()if n.endswith('sha256')})for k,z in sources.items()],fit_rows=0,new_labels=0,source_roles_changed=False,Test_or_validation_content_opened=False)
 p=O/'profile.frozen.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 eligible=[r for r in records if r['hip_drop_p95_p05']is not None]
 print(json.dumps(dict(profiled_intervals=len(records),sources=len(sources),held=len(held),people=sorted({r['person']for r in records}),range_percentiles=np.percentile([r['hip_drop_p95_p05']for r in eligible],[0,5,10,50,95]).tolist(),smallest=sorted(eligible,key=lambda r:r['hip_drop_p95_p05'])[:7]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
