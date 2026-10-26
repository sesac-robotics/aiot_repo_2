"""학습의 중간 등급과 새로운 검증 참여자를 우선 선택한다. 채점은 직접 검토로 수행한다."""
import json
from pathlib import Path
import prepare
OUT=Path(__file__).resolve().parent


def prepare_more():
    """기존 후보를 유지하며 학습 6개·검증 12개의 장면 표본을 추가한다.

    Returns:
        None: 개별 프레임 모음과 manifest.json을 저장한다.
    Raises:
        RuntimeError: 후보가 부족한 경우.
    """
    original=json.loads((OUT/'manifest.json').read_text())
    if len(original)>=46:
        print('추가 후보 이미 준비됨',len(original));return
    allrows=json.loads((prepare.BASE/'output/overnight-20261020/score_candidates.json').read_text());used={r['id'] for r in original};people={(r['exercise'],r['person']) for r in original};selected=list(original)
    for ex,split,token,quota in [('pushups','train','shallow',6),('pushups','validation','legs on the floor',3),('pushups','validation','90 degrees',3),('squats','validation','insufficient',2),('squats','validation','shallow',2),('squats','validation','over 90 degrees',2)]:
        pool=[r for r in allrows if r['exercise']==ex and r['split']==split and r['label_source']=='QEVD_provider' and any(token in t for t in r['source_labels'])];count=0
        for row in sorted(pool,key=lambda r:r['id']):
            if row['id'] in used or (ex,row['person']) in people:continue
            selected.append(dict(row,selection_reason='calibration_'+token));used.add(row['id']);people.add((ex,row['person']));count+=1
            if count==quota:break
        if count<quota:raise RuntimeError(ex+token)
    prepare.save_selected(selected)


if __name__=='__main__':prepare_more()
