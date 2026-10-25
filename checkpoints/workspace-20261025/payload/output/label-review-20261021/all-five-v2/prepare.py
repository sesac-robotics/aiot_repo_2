"""미검토 참여자의 시간순 검토판과 QEVD 잔여량을 저장한다.

입력은 로컬 후보 목록과 기존 직접 라벨이다. 점수를 자동 생성하지 않는다.
출력 장면에는 제공자 오류명을 표시하지 않으며 수치는 초·프레임 단위이다.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
BASE=ROOT.parents[2]
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)


def frames(row, number, screening=False):
    """한 후보의 시간순 장면 또는 촬영 구도 확인 장면을 추출한다.

    Args:
        row: 로컬 path를 가진 영상 메타데이터.
        number: 검토판 정수 번호.
        screening: True이면 구도 확인용 3장만 저장한다.
    Returns:
        dict: 초·프레임 단위 구간과 파일 해시를 추가한 메타데이터.
    Raises:
        RuntimeError: 영상 프레임을 읽지 못하는 경우.
    """
    row=dict(row);cap=cv2.VideoCapture(row['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);count=int(cap.get(7));last=min(count-1,round(12*fps))
    indices=np.unique(np.linspace(0,last,36).round().astype(int)) if not screening else np.array([round(last*x) for x in [.2,.5,.8]])
    images=[]
    for i in range(last+1):
        if not cap.grab():raise RuntimeError(row['id'])
        if i not in indices:continue
        ok,im=cap.retrieve()
        if not ok:raise RuntimeError(row['id'])
        images.append((i,Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))))
    cap.release()
    columns=3 if screening else 6;w=320 if screening else 256;h=220 if screening else 216
    canvas=Image.new('RGB',(columns*w,(len(images)+columns-1)//columns*h+35),'#14212a');draw=ImageDraw.Draw(canvas)
    draw.text((5,4),f'{"SCREEN" if screening else "REVIEW"} {number:03d} | {row["exercise"]} | TRAIN',font=FONT,fill='white')
    for j,(i,im) in enumerate(images):
        im.thumbnail((w,h-31));x=j%columns*w;y=35+j//columns*h;canvas.paste(im,(x+(w-im.width)//2,y));draw.text((x+3,y+h-27),f'f{i} {i/fps:.2f}s',font=FONT,fill='white')
    path=ROOT/f'{"screen" if screening else "case"}_{number:03d}.jpg';canvas.save(path,quality=94)
    row.update(review_number=number,evidence=dict(path=str(path),fps=fps,total_frames=count,start_frame=0,end_frame=last,sample_frames=[int(i) for i in indices]),video_sha256=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest())
    return row


def main():
    """새 후보 32개와 플랭크 추가 60명의 검토 자료를 저장한다. 반환값은 없으며 영상 읽기 오류는 전파한다."""
    pool=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text());old=json.loads((PARENT/'all-five-v1/labels.json').read_text());used={r['person'] for r in old};rng=np.random.default_rng(21212026);chosen=[]
    targets={'ytw':[('torso parallel',4),('arms too low',2),('not low enough',2),('thumbs pointed down',2),('too fast',1),('too slow',1)],'pushups':[('shoulder-width',2),('elevated hips',2),('shallow',2),('butt sinking',2)],'squats':[('no obvious issue',2),('shallow',2),('back not straight',2),('narrow',2)]}
    for ex,spec in targets.items():
        for text,n in spec:
            choices=[r for r in pool if r['exercise']==ex and r['split']=='train' and r['label_source'].startswith('QEVD') and any(text in s for s in r['source_labels'])];rng.shuffle(choices);added=0
            for r in choices:
                if r['person'] in used:continue
                chosen.append(r);used.add(r['person']);added+=1
                if added==n:break
    choices=[r for r in pool if r['exercise']=='shoulder' and r['split']=='train' and '_front' in r['id'] and r['person'] not in used]
    chosen.extend(sorted(choices,key=lambda r:r['id']))
    manifests=[]
    for n,r in enumerate(chosen,1):manifests.append(frames(r,n));print('review',n,r['exercise'],flush=True)
    (ROOT/'candidate_manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
    screened=json.loads((PARENT/'other-three-v1/screen_manifest.json').read_text());used.update(r['person'] for r in screened)
    choices=[r for r in pool if r['exercise']=='plank' and r['split']=='train' and r['person'] not in used];rng.shuffle(choices);screen=[]
    for r in choices:
        if r['person'] in used:continue
        screen.append(frames(r,len(screen)+1,True));used.add(r['person'])
        if len(screen)==60:break
    (ROOT/'screen_manifest.json').write_text(json.dumps(screen,ensure_ascii=False,indent=2))
    for start in range(0,len(screen),12):
        canvas=Image.new('RGB',(1600,4*205),'#14212a');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(screen[start:start+12]):
            im=Image.open(r['evidence']['path']);im.thumbnail((530,200));x=j%3*533;y=j//3*205;canvas.paste(im,(x,y));draw.text((x+3,y+180),f'P{r["review_number"]:03d}',font=FONT,fill='yellow')
        canvas.save(ROOT/f'overview_{start//12+1:02d}.jpg',quality=94)
    inventory={}
    known={r['id'] for r in old};screen_ids={r['id'] for r in screened}
    for ex in ['pushups','squats','plank','ytw','shoulder']:
        rs=[r for r in pool if r['exercise']==ex and r['label_source'].startswith('QEVD')]
        inventory[ex]=dict(local_candidate_clips=len(rs),previous_direct_review_clips=sum(r['id'] in known for r in rs),previous_screening_only=sum(r['id'] in screen_ids and r['id'] not in known for r in rs),new_direct_review_candidates=sum(r['exercise']==ex for r in manifests),new_screening_candidates=len(screen) if ex=='plank' else 0,all_candidates_reviewed=False)
    (ROOT/'inventory_before_review.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2));print('prepared',len(manifests),len(screen),flush=True)

if __name__=='__main__':main()
