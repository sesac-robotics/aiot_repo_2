from standing_state import *
from PIL import Image,ImageDraw
refs=read(O/'standing-calibrations.json');state=read(O/'standing-state-inputs.frozen.json');b=joblib.load(O/'standing-state-candidate.joblib');by={r['review_id']:r for r in state if r['split']=='train'};chosen=[];people=set()
for e,r in by.items():
 if r['person'] in people:continue
 xy,fs=load_frames(r);ref=refs[e];candidates=[]
 for i,x in enumerate(xy):
  f=frame_x(x,ref)
  if f is None or not .03<f[0]<.3:continue
  p=float(probability(b['model'],[dict(x=f)],b['dims'])[0])
  if .45<p<.99:candidates.append((p,i,f))
 if not candidates:continue
 candidates.sort(reverse=True);p,i,f=candidates[0];remaining=[a for a in candidates if abs(fs[a[1]]-fs[i])/r['fps']>=.25];other=min(remaining,key=lambda a:abs(a[0]-.65)) if remaining else candidates[-1];selected=[i,other[1]];chosen.append(dict(review_id=e,person=r['person'],path=r['path'],source_sha256=r['source_sha256'],split='train',fps=r['fps'],frames=[ref['calibration_frame']]+[int(fs[j]) for j in selected],not_a_target=True));people.add(r['person'])
for k,r in enumerate(chosen):
 if k%4==0:canvas=Image.new('RGB',(900,4*310),'#dddddd')
 import cv2
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(r['frames']):
  c.set(1,f);ok,a=c.read();assert ok
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((300,280));x=j*300;y=(k%4)*310;canvas.paste(im,(x+(300-im.width)//2,y+25));draw.text((x+3,y+3),r['review_id']+' f'+str(f)+(' REF' if j==0 else ''),fill='black')
 c.release();r['sheet']=str(O/f'hard-states-{k//4+1:02}.jpg');canvas.save(r['sheet'],quality=92)
save('hard-state-candidates.json',chosen);print('hard state sources',len(chosen))
