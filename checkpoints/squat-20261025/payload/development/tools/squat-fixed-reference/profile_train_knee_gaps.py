"""Actual Train knee gaps from existing cached observations, without filling."""
import sys,json,hashlib,datetime,collections
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/squat-count-Train-profile-20261025'
sys.path.insert(0,str(B/'output/live-webcam-20261023'));sys.path.insert(0,str(B/'tools/error-enrichment'))
from live_observation import mask_frame_border
from split_guard import assert_training_allowed

def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 policy=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text());inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};labels=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];ids=sorted({r['source_id']for r in labels if r['exercise']=='squats'and policy['assignments'][r['source_id']]=='train'})
 sources=[];runs=[]
 for sid in ids:
  inv=inventory[sid];p=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);raw=z['image_xy'];width,height=inv['width'],inv['height']
  if any(abs(int(r.get('rotation',0)))==90 for r in inv.get('rotation',[])):width,height=height,width
  assert m['config']['source_sha256']==inv['sha256'];assert str(z['fingerprint'])==m['fingerprint'];assert m['config']['required_views']==['front']
  row=dict(id=sid,split='train',source_sha256=inv['sha256'],sources=[inv['sha256']]);assert_training_allowed([row])
  x=mask_frame_border(raw,width,height);ticks=np.rint(z['time_seconds']*10).astype(int);assert len(ticks)==len(set(ticks))
  core=np.isfinite(x[:,[11,12,23,24]]).all((1,2));knee=np.isfinite(x[:,[25,26]]).all(2);rawknee=np.isfinite(raw[:,[25,26]]).all(2);missing=core&~knee.any(1)
  indices=np.flatnonzero(missing);parts=[]
  for ix in indices:
   if not parts or ticks[ix]!=ticks[parts[-1][-1]]+1:parts.append([int(ix)])
   else:parts[-1].append(int(ix))
  for ix in parts:
   a,c=ix[0],ix[-1];bracketed=a>0 and c<len(x)-1 and ticks[a]-ticks[a-1]==1 and ticks[c+1]-ticks[c]==1 and core[a-1]and core[c+1]and knee[a-1].any()and knee[c+1].any()
   rec=dict(id=f'{sid}_gap_{int(ticks[a])}',source_id=sid,split='train',start_tick=int(ticks[a]),end_tick=int(ticks[c]),slots=len(ix),seconds=len(ix)/10,body_observed_whole_gap=True,bracketed_by_actual_knee=bool(bracketed),raw_both_knees_missing_slots=int((~rawknee[ix].any(1)).sum()),core_only_gap_not_completion_truth=True,coordinate_interpolation=False)
   runs.append(rec)
  sources.append(dict(row,width=width,height=height,cache_sha256=sha(p),actual_ticks=len(ticks),core_valid=int(core.sum()),one_knee_valid=int((core&knee.any(1)).sum()),both_knees_valid=int((core&knee.all(1)).sum()),knee_only_gap_slots=int(missing.sum()),gap_runs=len(parts),pixels_visual_QA_not_repeated=True))
 out=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_roles_unchanged=True,new_fit=0,Test_or_validation_content_opened=False,sources=sources,gaps=runs,caveat='Observed knee-only gap is not a completed squat label; gap lengths alone cannot establish safe hold duration or missed-cycle accuracy.')
 p=O/'natural-knee-gaps.frozen.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(Train_sources=len(sources),actual_ticks=sum(r['actual_ticks']for r in sources),knee_only_gap_slots=sum(r['knee_only_gap_slots']for r in sources),gap_runs=len(runs),bracketed_runs=sum(r['bracketed_by_actual_knee']for r in runs),gap_length_histogram=dict(collections.Counter(r['slots']for r in runs)),bracketed_histogram=dict(collections.Counter(r['slots']for r in runs if r['bracketed_by_actual_knee']))),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
