"""Independently check actual Train evidence and NaN preservation for a fixed candidate."""
import importlib.util,json,sys,hashlib,datetime
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];L=B/'output/live-webcam-20261023';C=B/'project-documents/squat-motion-recovery-20261025/movement_stage_candidate.py';O=B/'output/squat-count-Train-profile-20261025'
sys.path.insert(0,str(L));from live_observation import mask_frame_border

def main():
 before=hashlib.sha256(C.read_bytes()).hexdigest();spec=importlib.util.spec_from_file_location('reviewed_motion_stage_candidate',C);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};results=[]
 for sid,expected in [('V10',4),('V18',1)]:
  inv=inventory[sid];width,height=inv['height'],inv['width'];z=np.load(B/'output/integrated-five-20261024/team-observations'/sid/'side.npz');ticks=np.rint(z['time_seconds']*10).astype(int);masked=mask_frame_border(z['image_xy'],width,height);lookup={int(t):x for t,x in zip(ticks,masked)};tracker=m.MovementTracker('squats','front');segments=[]
  for t,x in zip(ticks,z['image_xy']):
   segment=tracker.push(int(t),x,width=width,height=height)
   if segment:
    receipt=tracker.completion_receipts[-1];bt=receipt['bottom_confirmed_tick'];rt=receipt['return_first_tick'];ct=receipt['completed_tick'];assert bt<rt and rt+2==ct
    # Every phase confirmation requires three actual consecutive valid slots.
    for k in [bt-2,bt-1,bt,rt,rt+1,rt+2]:
     q=lookup[k];assert np.isfinite(q[[11,12,23,24]]).all()and np.isfinite(q[[25,26]]).all(1).any()
    st=[int(t)for t,_ in segment];assert st==sorted(set(st))
    for k,q in segment:assert np.allclose(q,lookup[k],equal_nan=True)
    cov=receipt['observation'];assert cov['detector_core_coverage']>=.8 and cov['actual_time_slot_coverage']>=.8
    segments.append(dict(start_tick=st[0],end_tick=st[-1],samples=len(st),missing_knee_samples=sum(not np.isfinite(q[[25,26]]).all(1).any()for _,q in segment),receipt=receipt))
  assert tracker.count==expected
  if sid=='V10':assert not any(s['end_tick']>=143 for s in segments)
  results.append(dict(source_id=sid,split='train',count=tracker.count,segments=segments,actual_phase_causality_and_original_nan_values_verified=True))
 assert hashlib.sha256(C.read_bytes()).hexdigest()==before
 p=O/'independent-stage-review.json';p.write_text(json.dumps(dict(reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_path=str(C),candidate_sha256=before,results=results,source_roles_changed=False,new_fit=0,independent_final_accuracy=False),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(candidate_sha256=before,Train_sources=2,counts=[r['count']for r in results],verified_complete_cycles=sum(r['count']for r in results),all_actual_phase_evidence_consecutive=True,raw_NaN_values_preserved=True,final_V10_arm_lowering_not_counted=True),ensure_ascii=False))
if __name__=='__main__':main()
