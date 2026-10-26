"""Explicitly selected exercise/view candidate inference, without label input."""
import argparse,sys,hashlib,json
from pathlib import Path
import numpy as np,cv2,joblib
from common import B,O,sha
from team_features import describe,height_features

def score(exercise,seq,view,duration,width,height,bundle,baseline=None,experimental=False):
 notes=describe(exercise,seq,view,duration,baseline,width,height);items=[]
 for n in notes:
  head=bundle['heads'].get((exercise,n['item'],n['bank']));value=None;reason=None
  if not n['eligible']:reason='required_observation_unavailable'
  elif head is None:reason='insufficient_normal_error_training_groups'
  elif not experimental and not head['candidate_gate_passed']:reason='candidate_validation_gate_not_passed'
  else:value=float(np.clip(head['model'].predict([n['feature']])[0],0,2))
  items.append(dict(item=n['item'],bank=n['bank'],ai_grade=value,hold_reason=reason,rule_used_as_AI=False))
 weights={'squats':[30,20,30,20],'pushups':[40,30,20,10],'shoulder':[30,25,25,20],'plank':[40,25,20,15],'ytw':[30,25,25,20]}[exercise]
 return dict(exercise=exercise,view=view,items=items,ai_total_score=sum(w*i['ai_grade']/2 for w,i in zip(weights,items)) if all(i['ai_grade'] is not None for i in items) else None,status='experimental_candidate_not_production',supported_return_grades=[1,2] if exercise=='squats' else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--exercise',choices=['squats','pushups','shoulder','plank','ytw'],required=True);p.add_argument('--view',choices=['front','side'],required=True);p.add_argument('--experimental',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();path=Path(a.video).resolve();cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release();h=sha(path)
 import extract
 rec=extract.extract_view(dict(id=h[:16],video_sha256=h,front_sha256=h,required_views=[a.view],segment=dict(start_frame=0,end_frame=count-2,fps=fps)),a.view,str(path),O/'inference-cache');z=np.load(rec['cache']);seq=z['image_xy'];baseline=height_features.reference(seq);result=score(a.exercise,seq,a.view,(count-1)/fps,width,height,joblib.load(O/'score-candidates.joblib'),baseline,a.experimental);result.update(video_sha256=h,model_sha256=sha(O/'score-candidates.joblib'));Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
