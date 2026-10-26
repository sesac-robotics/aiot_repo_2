"""보유 자료의 시험 예약과 미검토 학습 후보 표본을 분리해 준비한다."""
import collections
import hashlib
import json
import pathlib
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from allfive_round1 import known_ids
from resources import check_resources

ROOT = audit.BASE / 'output/five-exercise-recovery-20261023'


def prepare() -> dict:
    """시험 원본은 열지 않고 메타 그룹을 예약한 뒤 학습 후보만 표본화한다.

    Returns:
        dict: 시험 예약·자동 준비 수. 직접 라벨 수나 독립 시험 완료 수가 아니다.

    Raises:
        ValueError: 메타 역할 충돌·영상 해독 실패.
        OSError: 보유 자료나 출력 접근 실패.
    """
    ROOT.mkdir(exist_ok=True)
    pool=audit.read_json(audit.BASE/'output/overnight-20261020/score_candidates.json')
    old=audit.read_json(audit.BASE/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json')
    known=known_ids()
    # 준비판에 등장한 원본도 보수적으로 역사적 노출로 취급한다.
    for path in (audit.BASE/'output').glob('**/*preparation*.json'):
        if 'site-packages' in path.parts or ROOT in path.parents:continue
        known.update(re.findall(r'qevd_\d{8}|subject_\d{3}_[a-z_]+',path.read_text(errors='replace')))
    used_people={r['person'] for r in old}
    exposed_people=used_people|{r['person'] for r in pool if r['id'] in known}
    roles={r['person']:r['split'] for r in old}
    final=[dict(id=r['id'],exercise=r['exercise'],person=r['person'],path=r['path'],
        recording_group=r.get('recording_group',r['person']),split='final_test_reserved')
        for r in pool if r['split']=='test' and r['person'] not in exposed_people and pathlib.Path(r['path']).exists()]
    reserved_people={r['person'] for r in final}
    reservation=dict(schema='held-final-reservation-v1',records=final,
        counts=dict(collections.Counter(r['exercise'] for r in final)),
        reserved_people=sorted(reserved_people),historical_exposed_people=sorted(exposed_people),
        source_hashes_pending=True,person_identity_verified=False,final_test_executed=False,
        pixels_or_score_targets_opened=False,policy='Never use reserved people, original hashes or recordings for training or model selection; require source hash and identity audit before independent-test claim.')
    target=ROOT/'final_test_reservation.json'
    if target.exists():
        previous=audit.read_json(target)
        if previous['records']!=reservation['records']:raise ValueError('Reservation is immutable')
    else:audit.save_json(target,reservation)
    terms={
      'pushups':['early stopping','butt sinking','legs on the floor','deep','average','not deep','very wide','early stopping'],
      'squats':['shallow','insufficient','back not straight','no obvious issue','shallow','insufficient','back not straight','average'],
      'plank':['on knees','legs and hips on the floor','butt sinking','high hips','on knees','legs and hips on the floor','butt sinking','average','high hips','average','on knees','butt sinking'],
      'ytw':['arms too low','torso parallel','not low enough','arms too low','torso parallel','not low enough','average','no obvious issue'],
      'shoulder':['bad','bad','bad','good']}
    selected=[]
    for e,queries in terms.items():
        choices=[r for r in pool if r['exercise']==e and r['split']=='train' and r['id'] not in known
            and roles.get(r['person'],'train')=='train' and r['person'] not in reserved_people
            and (r['id'].startswith('qevd_') if e!='shoulder' else r['id'].endswith('_front'))]
        choices.sort(key=lambda r:hashlib.sha256(('held-round3-'+r['id']).encode()).hexdigest())
        picked=set()
        for term in queries:
            r=next((r for r in choices if r['person'] not in picked and any(term in x for x in r['source_labels'])),None)
            if r is None:continue
            picked.add(r['person']);selected.append(dict(r,selection_term=term))
    (ROOT/'screen').mkdir(exist_ok=True)
    records=[]
    for r in selected:
        check_resources();c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);count=int(c.get(7));fps=c.get(5)
        frames=np.linspace(0,count-1,8).round().astype(int);canvas=Image.new('RGB',(1280,155),'#14212a');draw=ImageDraw.Draw(canvas)
        for i,frame in enumerate(frames):
            c.set(1,int(frame));ok,bgr=c.read()
            if not ok:raise ValueError('Held video decode failed')
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((160,112));canvas.paste(im,(i*160+(160-im.width)//2,22));draw.text((i*160+2,136),f'f{frame}',fill='white')
        c.release();draw.text((3,2),r['id']+' | '+r['person'],fill='white');path=ROOT/'screen'/(r['id']+'.jpg');canvas.save(path,quality=95)
        records.append(dict(r,frame_count=count,fps=fps,evidence=str(path),sha256=audit.digest(pathlib.Path(r['path'])),sample_frames=frames.tolist()))
        audit.save_json(ROOT/'screening_manifest.json',dict(records=records,automatic_preparation_not_direct_review=True,completed=False))
    pages=[]
    for e in terms:
        rows=[r for r in records if r['exercise']==e]
        for start in range(0,len(rows),6):
            subset=rows[start:start+6];page=Image.new('RGB',(1280,155*len(subset)))
            for i,r in enumerate(subset):page.paste(Image.open(r['evidence']),(0,i*155))
            path=ROOT/f'{e}_{start//6+1}.jpg';page.save(path,quality=95);pages.append(str(path))
    result=dict(records=records,pages=pages,automatic_preparation_not_direct_review=True,completed=True)
    audit.save_json(ROOT/'screening_manifest.json',result)
    return dict(prepared=len(records),by_exercise=dict(collections.Counter(r['exercise'] for r in records)),reserved=reservation['counts'],labels_created=0)


if __name__=='__main__':print(json.dumps(prepare()))
