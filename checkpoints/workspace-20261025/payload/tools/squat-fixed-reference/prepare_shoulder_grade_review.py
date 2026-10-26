"""Fresh middle-cycle review, preserving the canonical source roles."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw

B=Path(__file__).resolve().parents[2]
S=B/'output/shoulder-common-support-audit-20261024/full-review-v1'
R=B/'output/shoulder-grade-model-v1-20261025'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import assert_training_allowed

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    R.mkdir(exist_ok=True)
    registry={x['id']:x for x in json.loads((S/'decoded-manifest.frozen.json').read_text())['sources']}
    sources=[]
    for n in [1,5,6,7,8,9,10,11]:
        sid=f'subject_{n:03d}_shoulder_bad_front';r=registry[sid];assert r['split']=='train'
        a,b=(20,26) if n==1 else (12,18)
        frames=[f for f in r['frames'] if a<=f['time']<=b];assert len(frames)<=32
        out=Image.new('RGB',(1600,1150),'white');draw=ImageDraw.Draw(out)
        draw.text((4,4),sid+' Train / NEW middle window / original5Hz evidence, no grades yet',fill='black')
        for j,f in enumerate(frames):
            im=Image.open(S/f['path']);x=j%8*200;y=j//8*280+30
            out.paste(im,(x+(200-im.width)//2,y+25));draw.text((x+2,y+2),f"f{f['frame']} {f['time']:.2f}s",fill='black')
        page=R/(sid+'-middle.jpg');assert not page.exists();out.save(page,quality=96)
        source=dict(id=sid,person=r['person'],split=r['split'],view='front',path=r['path'],
            source_sha256=r['source_sha256'],sources=[r['source_sha256']],
            start=a,end=b,frames=[f['frame'] for f in frames],sheet=str(page),
            sheet_sha256=sha(page),selection='all_model_owned_Train_bad_front_middle_windows',
            grades_not_assigned=True)
        assert sha(r['path'])==r['source_sha256'];sources.append(source)
    assert_training_allowed(sources)
    p=R/'middle-review-intake.frozen.json';assert not p.exists()
    p.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=sources,not_Test=True,selection_not_predictions=True),ensure_ascii=False,indent=2)+'\n')
    print('prepared middle windows',len(sources))

if __name__=='__main__':main()
