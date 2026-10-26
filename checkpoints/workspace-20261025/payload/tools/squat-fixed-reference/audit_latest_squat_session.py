"""Read-only independent phase and arithmetic audit of the latest squat test."""
import json,hashlib,datetime,sys,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];L=B/'output/live-webcam-20261023';SID='1792858741575615610';S=L/'sessions'/SID;O=B/'output/squat-retest-review-20261025'/SID
sys.path.insert(0,str(L));from live_observation import mask_frame_border

def main():
 O.mkdir(parents=True,exist_ok=True);raw=hashlib.sha256((S/'pose.npz').read_bytes()).hexdigest();d=json.loads((S/'final-state.json').read_text());assert d['exercise']=='squats'and not d['collecting']and not d['finalizing']
 spec=importlib.util.spec_from_file_location('latest_squat_review_tracker',L/'movement.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tracker=m.MovementTracker('squats','front');z=np.load(S/'pose.npz');width=int(z['width']);height=int(z['height']);fps=float(z['fps']);assert fps==10;ts=z['frame_numbers'].astype(int);x=z['image_xy'];masked=mask_frame_border(x,width,height);lookup={int(t):q for t,q in zip(ts,masked)};segments=[]
 for t,q in zip(ts,x):
  s=tracker.push(int(t),q,width=width,height=height)
  if s:
   r=tracker.completion_receipts[-1];bt=r['bottom_confirmed_tick'];rt=r['return_first_tick'];ct=r['completed_tick'];assert bt<rt and rt+2==ct
   for k in [bt-2,bt-1,bt,rt,rt+1,ct]:
    q=lookup[k];assert np.isfinite(q[[11,12,23,24]]).all()and np.isfinite(q[[25,26]]).all(1).any()
   for k,q in s:assert np.allclose(q,lookup[k],equal_nan=True)
   body=np.array([q[[23,24]].mean(0)for _,q in s]);sh=np.array([q[[11,12]].mean(0)for _,q in s]);scale=float(np.nanmedian(np.linalg.norm(body-sh,axis=1)));drop=(body[:,1]-body[0,1])/scale;knees=np.array([q[[25,26]]for _,q in s]);distance=np.abs(knees[:,0,0]-knees[:,1,0]);ratio=distance/np.median(distance[:3]);segments.append(dict(number=tracker.count,start_tick=int(s[0][0]),end_tick=int(s[-1][0]),duration_seconds=(s[-1][0]-s[0][0])/10,hip_range_p95_p05=float(np.ptp(np.nanpercentile(drop,[5,95]))),minimum_knee_width_ratio_to_first3=float(np.min(ratio)),receipt=r))
 assert tracker.count==d['movement']['count']==len(d['repetitions'])==7;assert tracker.completion_receipts==d['movement']['completion_receipts']
 weights={'descent_range':30,'standing_return':20,'knee_inward':30,'trunk_balance':20};scores=[]
 for r in d['repetitions']:
  s=r['score_100'];summary={}
  for method in ['ai','rule']:
   vals=[(i['id'],i[method+'_points'])for i in s['items']if i[method+'_points']is not None];assert all(0<=v<=weights[k]for k,v in vals);assert round(sum(v for k,v in vals),1)==s[method]['earned_subtotal'];assert sum(weights[k]for k,v in vals)==s[method]['assessed_weight']==100;assert s[method]['total']==s[method]['earned_subtotal'];summary[method]=s[method]['total']
  scores.append(dict(number=r['number'],**summary,items=[dict(id=i['id'],ai_points=i['ai_points'],rule_points=i['rule_points'])for i in s['items']]))
 assert hashlib.sha256((S/'pose.npz').read_bytes()).hexdigest()==raw
 v=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session=SID,samples=len(ts),duration_seconds=(ts[-1]-ts[0])/10,stored_count=7,replay_count=tracker.count,all_phase_receipts_match=True,actual_consecutive_one_knee_phase_evidence=True,no_interpolation=True,score_rows_arithmetic_verified=7,all_four_items_scored_in_saved_rows=True,raw_pose_sha256=raw,movement_sha256=hashlib.sha256((L/'movement.py').read_bytes()).hexdigest(),segments=segments,scores=scores,uncertain_attempts=d['movement']['uncertain_attempts'],recovery_receipts=d['movement'].get('recovery_receipts',[]),actual_human_count_confirmed=False,saved_RGB_video_present=False,expert_posture_truth_available=False,new_fit=0,Train_Test_changed=False)
 (O/'independent-review.frozen.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v[k]for k in ['session','samples','duration_seconds','stored_count','replay_count','score_rows_arithmetic_verified','all_phase_receipts_match','recovery_receipts']},ensure_ascii=False))
 print(json.dumps([{'number':r['number'],'hip_range':r['hip_range_p95_p05'],'knee_min_ratio':r['minimum_knee_width_ratio_to_first3']}for r in segments],ensure_ascii=False))
if __name__=='__main__':main()
