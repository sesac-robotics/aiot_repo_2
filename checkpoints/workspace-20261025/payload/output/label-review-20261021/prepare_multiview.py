"""원본 동기화 설명에 따라 세 카메라의 시간 표시 장면을 CPU로 추출한다.

서로 다른 시점을 새 독립 표본으로 세지 않는다. MultiView는 문서에 동기화 보장이
없으므로 각 영상의 상대 진행률로만 대조하며 같은 순간이라고 단정하지 않는다.
"""
from pathlib import Path
import json
import cv2
import numpy as np

BASE=Path('/home/jayden/SeSACPAI-Mid')
OUT=BASE/'output/label-review-20261021'


def prepare():
    """20개 수행의 세 시점을 12개 시각으로 추출하고 대응 관계를 저장한다.

    Returns:
        None: multiview/*.jpg 및 multiview_manifest.json을 기록한다.
    Raises:
        RuntimeError: 영상 누락·디코딩 실패가 발생한 경우.
    """
    manifest=json.loads((OUT/'manifest.json').read_text());result=[]
    for r in manifest:
        paths=[]
        if r['exercise']=='pushups':
            root=Path(r['path']).parent.parent
            for view in ['side','front','diagonal']:
                paths.append((view,root/view/(r['id'].removesuffix('_side')+'_'+view+'.mp4')))
        else:
            for cam in [1,2,3]:
                paths.append((f'cam{cam}',Path(r['path']).with_name(r['original_id'].rsplit('_cam',1)[0]+f'_cam{cam}.mp4')))
        canvas=np.zeros((6*250,6*256,3),dtype=np.uint8);views=[]
        for v,(view,p) in enumerate(paths):
            cap=cv2.VideoCapture(str(p));fps=cap.get(5);count=int(cap.get(7))
            start=r.get('start_frame',0);end=min(count-1,r.get('end_frame',count-1))
            indices=np.linspace(start,end,12).astype(int);times=[]
            for j,idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,frame=cap.read()
                if not ok:raise RuntimeError(str(p))
                h,w=frame.shape[:2]
                if r['exercise']=='squats':frame=frame[int(h*.30):int(h*.98),int(w*.1):int(w*.98)]
                h,w=frame.shape[:2];scale=min(256/w,225/h);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
                y=(v*2+j//6)*250;x=(j%6)*256
                canvas[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
                t=(idx-start)/fps;times.append(round(t,3))
                cv2.putText(canvas,f'{view} {t:.2f}s',(x+3,y+243),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
            cap.release();views.append(dict(view=view,path=str(p),fps=fps,frames=count,times=times))
        image=OUT/'multiview'/f'{r["review_id"]:02d}.jpg';cv2.imwrite(str(image),canvas)
        result.append(dict(review_id=r['review_id'],id=r['id'],image=str(image),views=views,synchronization='provider_frame_sync' if r['exercise']=='squats' else 'relative_progress_only'))
    (OUT/'multiview_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('완료:',len(result),'개 수행 /',len(result)*3,'개 카메라 영상')


if __name__=='__main__':prepare()
