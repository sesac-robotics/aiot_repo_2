"""Decode owned original videos once, preserving 5Hz pixel evidence for review."""
import collections,datetime,json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
O=B/'output/shoulder-common-support-audit-20261024/full-review-v1'
I=B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json'
def main():
 O.mkdir(exist_ok=True)
 records=[r for r in json.loads(I.read_text())['sources'] if r['owner']=='model_evaluation']
 assert len(records)==72
 p=O/'plan.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(
   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inventory_path=str(I),inventory_sha256=sha(I),
   owned_sources=72,subjects=list(range(1,13)),all_roles_preserved=True,review_protocol=[
    'Whole-video1Hz contact sheets are screening only; never sole completed event-grade evidence.',
    'Preserve5Hz original-frame thumbnails for event-centered dense inspection; refine raw frames if boundaries remain uncertain.',
    'Grade each completed return against observed initial-ready arm state. A next ascent may close a partial return; clipping/missing is held.',
    'Frontal trunk target is lateral trunk stability. Side/diagonal sagittal motion is a separate observation, not frontal supervision.',
    'Provider good/bad does not assign grades. Existing labels are not overwritten. Source and same-recording views stay in their old role.',
    'Only Train may fit after combined review. Historical Test/development are evaluation only. No final-Test or live service change.'],
   no_fit=True),indent=2))
 decoded=[]
 for r in records:
  assert sha(r['path'])==r['source_sha256']
  cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(cv2.CAP_PROP_FPS);N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  targets=sorted(set([min(N-1,round(t*fps/5)) for t in range(int(N/fps*5)+1)]+[N-1]));wanted=set(targets)
  dest=O/'frames'/r['id'];dest.mkdir(parents=True,exist_ok=True);frames=[];idx=-1
  while cap.grab():
   idx+=1
   if idx not in wanted:continue
   ok,a=cap.retrieve();assert ok
   im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((200,250))
   p=dest/f'f{idx:06}.jpg';assert not p.exists();im.save(p,quality=94)
   frames.append(dict(frame=idx,time=idx/fps,path=str(p.relative_to(O))))
  cap.release();assert len(frames)==len(targets),(r['id'],len(frames),len(targets))
  decoded.append(dict(r,fps=fps,frame_count=N,duration=N/fps,frames=frames))
  print('DECODED',r['id'],N,round(N/fps,2),len(frames),flush=True)
  subj=r['person'];others=[a for a in decoded if a['person']==subj]
  if len(others)!=6:continue
  others=sorted(others,key=lambda a:(a['provider_tag'],a['view']));duration=max(a['duration'] for a in others)
  for page,start in enumerate(range(0,int(duration)+1,8),1):
   canvas=Image.new('RGB',(1600,6*285),'white');draw=ImageDraw.Draw(canvas)
   for row,a in enumerate(others):
    for col,t in enumerate(range(start,start+8)):
     x=col*200;y=row*285;draw.text((x+2,y+3),f"{a['id'][8:11]} {a['provider_tag']} {a['view']} {t}s",fill='black')
     if t>a['duration']:continue
     f=min(a['frames'],key=lambda z:abs(z['time']-t));im=Image.open(O/f['path']);canvas.paste(im,(x+(200-im.width)//2,y+28))
   canvas.save(O/f'{subj}-overview-{page:02}.jpg',quality=94)
  print('OVERVIEW_READY',subj,flush=True)
 p=O/'decoded-manifest.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(source_count=len(decoded),sources=decoded),indent=2))
 print('READY72',flush=True)
if __name__=='__main__':main()
