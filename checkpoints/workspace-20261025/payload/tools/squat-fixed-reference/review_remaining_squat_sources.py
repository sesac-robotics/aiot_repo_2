"""Prepare remaining MultiView Train side sources for human/AI visual review."""
import collections
import hashlib
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

B=Path(__file__).resolve().parents[2]
R=B/'output/squat-pose-alignment-v5-20261024/remaining-Train'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    R.mkdir(exist_ok=True)
    rows=json.loads((B/'output/squat-fixed-reference-combined-v3-20261024/score-inputs.frozen.json').read_text())
    roles=collections.defaultdict(set)
    for r in rows:
        if r['family']=='MultiView':roles[r['person']].add(r['split'])
    prior=json.loads((B/'output/squat-fixed-reference-v1-20261024/multiview-review/intake.frozen.json').read_text())['records']
    done={r['id'] for r in prior}
    root=B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise/squat/bad/side'
    sources=[]
    for path in sorted(root.glob('*.mp4')):
        sid=path.stem;person='shoulder_'+sid.split('_')[1]
        if roles[person]!={'train'} or sid in done:continue
        sources.append(dict(id=sid,person=person,split='train',path=str(path),source_sha256=sha(path),
            sources=[sha(path)],selection='remaining_canonical_Train_side_sources_not_prediction_based',
            provider_tag_not_target=True,grade=None,fitted=False))
    intake=R/'intake.frozen.json';assert not intake.exists();intake.write_text(json.dumps(sources,indent=2))
    for r in sources:
        c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);n=int(c.get(7))
        frames=list(range(0,n,round(fps)));frames=sorted(set(frames+[n-1]))
        out=Image.new('RGB',(1600,((len(frames)+7)//8)*280+30),'white');draw=ImageDraw.Draw(out)
        draw.text((4,4),r['id']+' whole1Hz screening; NOT grades',fill='black')
        for j,f in enumerate(frames):
            c.set(1,f);ok,a=c.read();assert ok
            im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((200,250));x=j%8*200;y=j//8*280+30
            out.paste(im,(x+(200-im.width)//2,y+25));draw.text((x+3,y+3),f'f{f} {f/fps:.2f}s',fill='black')
        c.release();out.save(R/(r['id']+'-screen.jpg'),quality=96)
    print('New return-focused screening sources:',len(sources))

if __name__=='__main__':main()
