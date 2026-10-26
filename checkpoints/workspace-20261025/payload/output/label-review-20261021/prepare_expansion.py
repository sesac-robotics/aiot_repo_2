"""참여자 분할을 유지하며 푸시업·스쿼트의 점수 검토 후보를 확대한다."""
from pathlib import Path
import json
import collections
import cv2
import numpy as np

BASE=Path('/home/jayden/SeSACPAI-Mid')
ROOT=BASE/'output/label-review-20261021'
OUT=ROOT/'expanded';OUT.mkdir(exist_ok=True)


def frames(row,number):
    """선택 구간을 순차 디코딩해 시간 표시 장면을 저장한다.

    Args:
        row: 영상 경로·시작·종료 프레임이 있는 후보 정보.
        number: 보고서용 일련 식별자.
    Returns:
        dict: 이미지 경로와 실제 표본 프레임 번호·초 단위 시각.
    Raises:
        RuntimeError: 메타데이터나 디코딩 오류.
    """
    cap=cv2.VideoCapture(row['path']);fps=cap.get(5);count=int(cap.get(7))
    if fps<=0:raise RuntimeError(row['id'])
    first=int(row.get('start_frame',0));last=min(count-1,int(row.get('end_frame',round(11.67*fps))));indices=np.linspace(first,last,36).round().astype(int)
    canvas=np.zeros((6*190,6*320,3),dtype=np.uint8);cap.set(cv2.CAP_PROP_POS_FRAMES,first);j=0
    for index in range(first,last+1):
        if not cap.grab():raise RuntimeError(row['id'])
        if j<36 and index==indices[j]:
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(row['id'])
            while j<36 and indices[j]==index:
                image=frame
                h,w=image.shape[:2]
                if h>w:
                    image=image[int(.30*h):int(.98*h),int(.10*w):int(.98*w)] if row['exercise']=='squats' else image[int(.30*h):int(.73*h),:]
                h,w=image.shape[:2];scale=min(320/w,165/h);image=cv2.resize(image,(round(w*scale),round(h*scale)))
                x=j%6*320;y=j//6*190;canvas[y:y+image.shape[0],x:x+image.shape[1]]=image
                cv2.putText(canvas,f'{(index-first)/fps:.2f}s',(x+4,y+184),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);j+=1
    cap.release();p=OUT/f'{number:03d}.jpg';cv2.imwrite(str(p),canvas)
    return dict(image=str(p),first_frame=first,last_frame=last,fps=fps,indices=indices.tolist(),times=[round((int(i)-first)/fps,3) for i in indices])


def prepare():
    """푸시업 측면 영상 48개와 스쿼트 40반복 후보를 저장한다. 채점하지 않는다."""
    c=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text())
    push=sorted([r for r in c if r['exercise']=='pushups' and r['label_source']=='MultiView_provider' and r['id'].endswith('_side')],key=lambda r:(r['person'],not r['reference']))
    squat=[]
    for split,n in [('train',20),('validation',10),('test',10)]:
        pool=[r for r in c if r['exercise']=='squats' and r['label_source']=='REHAB26_provider' and r['split']==split]
        selected=[]
        for ref,quota in [(True,n//2),(False,n//2)]:
            groups=collections.defaultdict(list)
            for r in pool:
                if r['reference']==ref:groups[(r['person'],str(r['source_labels']))].append(r)
            while sum(x['reference']==ref for x in selected)<quota:
                added=False
                for key in sorted(groups):
                    if groups[key] and sum(x['reference']==ref for x in selected)<quota:
                        selected.append(groups[key].pop(0));added=True
                if not added:raise RuntimeError('후보 부족')
        squat+=selected
    records=[]
    for i,r in enumerate(push+squat,1):
        record=dict(r,review_number=i)
        target=OUT/f'{i:03d}.jpg'
        if target.exists():continue
        record['evidence']=frames(r,i);records.append(record)
        (OUT/f'{i:03d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    allrecords=[json.loads(p.read_text()) for p in sorted(OUT.glob('[0-9][0-9][0-9].json'))]
    (OUT/'manifest.json').write_text(json.dumps(allrecords,ensure_ascii=False,indent=2))
    print('검토 후보',len(allrecords))


if __name__=='__main__':prepare()
