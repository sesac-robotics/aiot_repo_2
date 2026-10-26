"""스쿼트 반복의 동기화 정면 근거를 추출해 측면 장면과 대조한다."""
from pathlib import Path
import csv
import json
import cv2
import numpy as np

OUT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021/expanded')
BASE=OUT.parents[2]


def prepare():
    """방향 메타데이터에서 정면에 가장 가까운 카메라를 선택하고 12장씩 저장한다.

    Returns:
        None: squat_front/*.jpg와 카메라·시각 JSON을 출력한다.
    Raises:
        RuntimeError: 메타데이터 누락 또는 영상 읽기 실패.
    """
    rows=list(csv.DictReader((BASE/'data/rehab26-squat/raw/annotations/Segmentation.csv').open(),delimiter=';'))
    dest=OUT/'squat_front';dest.mkdir(exist_ok=True)
    for n in range(49,89):
        r=json.loads((OUT/f'{n:03d}.json').read_text());annotation=next(a for a in rows if a['person']==r['person'].removeprefix('rehab_') and int(a['first_frame'])==r['start_frame'] and int(a['last_frame'])==r['end_frame'])
        orientation=float(annotation['person_orientation']);cam=min([1,2,3],key=lambda c:abs((c-1)*45-orientation));p=Path(r['path']).with_name(r['original_id'].rsplit('_cam',1)[0]+f'_cam{cam}.mp4')
        cap=cv2.VideoCapture(str(p));fps=cap.get(5);first=r['start_frame'];last=r['end_frame'];targets=np.linspace(first,last,12).round().astype(int);cap.set(cv2.CAP_PROP_POS_FRAMES,first);canvas=np.zeros((3*260,4*300,3),np.uint8);j=0
        for index in range(first,last+1):
            if not cap.grab():raise RuntimeError(str(p))
            if j<12 and index==targets[j]:
                ok,frame=cap.retrieve()
                if not ok:raise RuntimeError(str(p))
                h,w=frame.shape[:2];frame=frame[int(h*.30):int(h*.98),int(w*.10):int(w*.98)];h,w=frame.shape[:2];s=min(300/w,235/h);frame=cv2.resize(frame,(round(w*s),round(h*s)));x=j%4*300;y=j//4*260;canvas[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
                cv2.putText(canvas,f'cam{cam} {(index-first)/fps:.2f}s',(x+5,y+253),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);j+=1
        cap.release();cv2.imwrite(str(dest/f'{n:03d}.jpg'),canvas);(dest/f'{n:03d}.json').write_text(json.dumps(dict(path=str(p),camera=cam,relative_view_degrees=(cam-1)*45-orientation,times=[round((int(x)-first)/fps,3) for x in targets]),ensure_ascii=False,indent=2))
    print('정면 근거 40개 저장')


if __name__=='__main__':prepare()
