"""부족한 수행 사례를 골라 직접 라벨링할 장면 표본을 저장한다. 자동 채점은 하지 않는다."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
BASE=Path('/home/jayden/SeSACPAI-Mid')
OUT=Path(__file__).resolve().parent


def prepare():
    """학습·검증 분할을 유지한 QEVD 후보와 시각 검토용 36장 이미지를 저장한다.

    Returns:
        None: manifest.json과 개별 JPEG 파일을 저장한다.
    Raises:
        RuntimeError: 필요한 후보 또는 영상 프레임이 없는 경우.
    """
    candidates=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text());selected=[];used=set();people=set()
    specs={'pushups':[('legs on the floor',4),('insufficient',4),('shallow',4),('early stopping',2),('90 degrees',2)],'squats':[('insufficient',3),('shallow',3),('90 degrees',2)]}
    for ex,spec in specs.items():
        for token,quota in spec:
            pool=[r for r in candidates if r['exercise']==ex and r['split']=='train' and r['label_source']=='QEVD_provider' and any(token in t for t in r['source_labels'])]
            count=0
            for row in sorted(pool,key=lambda r:r['id']):
                if row['id'] in used or (ex,row['person']) in people:continue
                used.add(row['id']);people.add((ex,row['person']));selected.append(dict(row,selection_reason=token));count+=1
                if count==quota:break
            if count<quota:raise RuntimeError('후보 부족: '+ex+token)
    extra=[r for r in candidates if r["exercise"]=="pushups" and r["split"]=="train" and r["label_source"]=="QEVD_provider" and any("legs on the floor" in t for t in r["source_labels"])]
    count=0
    for row in sorted(extra,key=lambda r:r["id"]):
        if row["id"] in used or ("pushups",row["person"]) in people:continue
        used.add(row["id"]);people.add(("pushups",row["person"]));selected.append(dict(row,selection_reason="support_failure_followup"));count+=1
        if count==4:break
    save_selected(selected)


def save_selected(selected):
    """선택된 원본의 장면을 저장하고 기존 검토 후보도 보존한다.

    Args:
        selected: 원본 경로·ID·분할이 포함된 후보 목록. 기존 식별 순서를 유지한다.
    Returns:
        None: JPEG, 개별 JSON, 누적 manifest.json을 저장한다.
    Raises:
        RuntimeError: 영상 디코딩에 실패한 경우.
        AssertionError: 원본 중복이 발견된 경우.
    """
    manifest=[]
    for n,row in enumerate(selected,1):
        imagepath=OUT/f'{n:03d}.jpg';meta=OUT/f'{n:03d}.json'
        if meta.exists():manifest.append(json.loads(meta.read_text()));continue
        cap=cv2.VideoCapture(row['path']);fps=cap.get(5);count=int(cap.get(7));last=min(count-1,round(15*fps));indices=np.linspace(0,last,36).round().astype(int)
        canvas=np.zeros((6*210,6*350,3),np.uint8);images=[]
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
            if not ok:raise RuntimeError(row['id'])
            images.append(frame)
        cap.release()
        for j,frame in enumerate(images):
            h,w=frame.shape[:2];scale=min(350/w,185/h);frame=cv2.resize(frame,(round(w*scale),round(h*scale)));x=j%6*350;y=j//6*210
            canvas[y:y+frame.shape[0],x:x+frame.shape[1]]=frame;cv2.putText(canvas,f'{indices[j]/fps:.2f}s',(x+4,y+204),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(imagepath),canvas)
        row.update(review_number=n,evidence=dict(image=str(imagepath),fps=fps,frame_count=count,indices=indices.tolist(),times=[round(int(i)/fps,3) for i in indices]),video_sha256=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest())
        meta.write_text(json.dumps(row,ensure_ascii=False,indent=2));manifest.append(row)
    manifest=[json.loads(p.read_text()) for p in sorted(OUT.glob('[0-9][0-9][0-9].json'))]
    assert len(set(r['video_sha256'] for r in manifest))==len(manifest)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('직접 검토 후보',len(manifest),flush=True)


if __name__=='__main__':prepare()
