"""정상·오류 영상 20개를 선정하고 CPU로 시각 검토용 장면을 저장한다.

원본 영상·라벨은 변경하지 않는다. 출력 장면은 정답 점수나 전체 영상 검토의 대체물이 아니다.
"""
from pathlib import Path
import json
import cv2
import numpy as np

BASE=Path('/home/jayden/SeSACPAI-Mid')
OUT=BASE/'output/label-review-20261021'


def prepare():
    """운동별 정상 5개·오류 5개의 경로와 시간 표시 장면을 저장한다.

    Returns:
        None: manifest.json과 20개의 JPG 파일을 출력한다.
    Raises:
        RuntimeError: 필요한 영상 수나 디코딩 프레임이 부족한 경우.
    """
    candidates=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text())
    selected=[]
    people=['006','014','018','023','024']
    for ref in [True,False]:
        pool=[r for r in candidates if r['exercise']=='pushups' and r['label_source']=='MultiView_provider' and r['reference']==ref and '_side' in r['id']]
        pool.sort(key=lambda r:(r['person'][-3:] not in people,r['id']))
        selected+=pool[:5]
    for ref in [True,False]:
        pool=[r for r in candidates if r['exercise']=='squats' and r['label_source']=='REHAB26_provider' and r['reference']==ref]
        chosen=[];seen=set()
        for r in pool:
            key=(r['person'],tuple(r['source_labels'][1:]))
            if key not in seen:
                chosen.append(r);seen.add(key)
            if len(chosen)==5:break
        if len(chosen)<5:raise RuntimeError('독립 표본 부족')
        selected+=chosen
    records=[]
    for i,row in enumerate(selected,1):
        cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start=int(row.get('start_frame',0));end=min(count-1,int(row.get('end_frame',count-1)))
        canvas=np.zeros((3*240,4*320,3),dtype=np.uint8);times=[]
        for j,idx in enumerate(np.linspace(start,end,12).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,frame=cap.read()
            if not ok:raise RuntimeError(f'디코딩 실패: {row["id"]}')
            h,w=frame.shape[:2];scale=min(320/w,214/h);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
            y=(j//4)*240;x=(j%4)*320;canvas[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
            sec=(idx-start)/fps;times.append(round(sec,3))
            cv2.putText(canvas,f'{i:02d} | {sec:.2f}s',(x+5,y+233),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        cap.release();image=OUT/f'{i:02d}.jpg';cv2.imwrite(str(image),canvas)
        records.append(dict(row,review_id=i,montage_path=str(image),relative_seconds=times,
                            provider_class='normal' if row['reference'] else 'error',review_status='pending',score=None))
    (OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print([(r['review_id'],r['id'],r['source_labels']) for r in records])


if __name__=='__main__':prepare()
