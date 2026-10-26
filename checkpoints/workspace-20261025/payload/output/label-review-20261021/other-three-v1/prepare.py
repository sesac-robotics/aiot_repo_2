"""나머지 세 운동의 학습 분할 대표 영상을 직접 검토할 근거를 준비한다.

점수와 등급을 자동 생성하지 않는다. 제공자 정상·오류 표시는 검토판에서 숨긴다.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)


def choose(rows, ex, labels):
    """서로 다른 참여자를 우선하여 지정 제공자 라벨별 두 후보를 고른다.

    Args:
        rows: 로컬 영상 후보 목록.
        ex: 영어 운동 코드.
        labels: 표집에만 사용하는 제공자 라벨 문자열 목록.
    Returns:
        list: 참여자가 중복되지 않는 학습 분할 후보.
    """
    result=[];people=set()
    for label in labels:
        a=sorted([r for r in rows if r['exercise']==ex and r['split']=='train' and label in r['source_labels']],key=lambda r:r['id'])
        count=0
        for r in a:
            if r['person'] in people:continue
            result.append(copy.deepcopy(r));people.add(r['person']);count+=1
            if count==2:break
    return result


def prepare():
    """후보의 시간순 장면과 파일 해시를 저장한다.

    Returns:
        None: candidate_manifest.json, case_*.jpg를 저장한다.
    Raises:
        RuntimeError: 영상 읽기 실패 또는 목표 수 부족 시.
    """
    rows=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text());chosen=[]
    chosen+=choose(rows,'plank',['elbow plank - average','elbow plank - high hips','elbow plank - butt sinking','elbow plank - on knees','elbow plank - legs and hips on the floor','elbow plank - stopping early','elbow plank - arms too wide','elbow plank - shaking'])
    chosen+=choose(rows,'ytw',['standing ytw - torso parallel to the ground, thumbs pointed up','standing ytw - arms too low','standing ytw - not low enough','standing ytw - bent legs','standing ytw - too fast','standing ytw - too slow'])
    sh=sorted([r for r in rows if r['exercise']=='shoulder' and r['split']=='train' and '_front' in r['id']],key=lambda r:r['id']);people=sorted(set(r['person'] for r in sh))[:6]
    chosen += [copy.deepcopy(r) for r in sh if r['person'] in people]
    for n,r in enumerate(chosen,1):
        r['review_number']=n;r['video_sha256']=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
        cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);total=int(cap.get(7));last=min(total-1,int(fps*12));indices=np.unique(np.linspace(0,last,24).round().astype(int));targets=set(int(i) for i in indices);ims=[]
        for i in range(last+1):
            if not cap.grab():raise RuntimeError(r['id'])
            if i not in targets:continue
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(r['id'])
            ims.append((i,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        cap.release();canvas=Image.new('RGB',(1536,4*216+35),'#14212a');d=ImageDraw.Draw(canvas);d.text((8,6),f'{n:03d} | {r["exercise"]} | TRAIN | time ordered',font=FONT,fill='white')
        for j,(idx,im) in enumerate(ims):
            im.thumbnail((256,185));x=j%6*256;y=35+j//6*216;canvas.paste(im,(x+(256-im.width)//2,y));d.text((x+4,y+187),f'f{idx} {idx/fps:.2f}s',font=FONT,fill='white')
        canvas.save(ROOT/f'case_{n:03d}.jpg',quality=94)
        r['evidence']=dict(path=str(ROOT/f'case_{n:03d}.jpg'),fps=fps,total_frames=total,review_start=0,review_end=last,indices=indices.tolist(),times_seconds=[round(int(i)/fps,3) for i in indices]);(ROOT/f'{n:03d}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
        print(n,r['exercise'],r['id'],flush=True)
    (ROOT/'candidate_manifest.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));print('대표 검토 후보',len(chosen),flush=True)

if __name__=='__main__':prepare()
