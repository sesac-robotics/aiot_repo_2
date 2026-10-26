"""푸시업 첫 12초를 초당 3장으로 추출해 자세 전환 구간을 검토한다.

이 샘플링으로 순간 반동·속도·정확한 반복 수를 확정하지 않는다.
"""
from pathlib import Path
import json
import cv2

OUT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021')


def prepare():
    """푸시업 10개 영상의 첫 12초 장면을 순차 디코딩해 저장한다.

    Returns:
        None: dense/*.jpg와 시간·프레임 대응 JSON을 저장한다.
    Raises:
        RuntimeError: 영상 프레임을 읽을 수 없는 경우.
    """
    import numpy as np
    dest=OUT/'dense';dest.mkdir(exist_ok=True);records=[]
    for r in json.loads((OUT/'manifest.json').read_text())[:10]:
        cap=cv2.VideoCapture(r['path']);fps=cap.get(5);count=int(cap.get(7))
        targets=[int(round(t*fps/3)) for t in range(36) if int(round(t*fps/3))<count]
        canvas=np.zeros((6*190,6*320,3),dtype=np.uint8);i=0;index=0;times=[]
        while i<len(targets):
            if not cap.grab():raise RuntimeError(r['path'])
            if index==targets[i]:
                ok,frame=cap.retrieve()
                if not ok:raise RuntimeError(r['path'])
                h,w=frame.shape[:2]
                if h>w:frame=frame[int(.30*h):int(.73*h),:]
                h,w=frame.shape[:2];s=min(320/w,165/h);frame=cv2.resize(frame,(round(w*s),round(h*s)))
                y=(i//6)*190;x=(i%6)*320;canvas[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
                sec=index/fps;times.append(round(sec,3));cv2.putText(canvas,f'{sec:.2f}s',(x+5,y+184),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);i+=1
            index+=1
        cap.release();p=dest/f'{r["review_id"]:02d}.jpg';cv2.imwrite(str(p),canvas)
        records.append(dict(id=r['id'],image=str(p),times=times,window_seconds=[0,times[-1]],sample_fps=3))
    (OUT/'dense_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('완료',len(records))


if __name__=='__main__':prepare()
