"""예약 시험 후보의 원본 지문만 계산한다. 영상 디코딩·등급은 읽지 않는다."""
import collections
import hashlib
import json
import pathlib

BASE=pathlib.Path('/home/jayden/SeSACPAI-Mid')
ROOT=BASE/'output/five-exercise-recovery-20261023'


def main() -> None:
    """과거 원본과 중복된 후보를 제외하고 불변 지문 예약을 저장한다."""
    reservation=json.loads((ROOT/'final_test_reservation.json').read_text())
    old=json.loads((BASE/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json').read_text())
    history={s for r in old for s in r['sources']};records=[];rejected=[]
    for r in reservation['records']:
        digest=hashlib.sha256()
        with pathlib.Path(r['path']).open('rb') as stream:
            while block:=stream.read(1024*1024):digest.update(block)
        sha=digest.hexdigest()
        if sha in history:rejected.append(r['id']);continue
        records.append(dict(id=r['id'],exercise=r['exercise'],person=r['person'],sources=[sha],recording_groups=[r['recording_group']],split='final_test'))
    result=dict(schema='held-final-hashed-reservation-v1',records=records,counts=dict(collections.Counter(r['exercise'] for r in records)),rejected_historical_sources=rejected,historical_people=reservation['historical_exposed_people'],historical_sources=sorted(history),source_hashes_verified=True,person_identity_verified=False,pixels_or_score_targets_opened=False,final_test_executed=False)
    p=ROOT/'final_test_hashed_manifest.json'
    if p.exists() and json.loads(p.read_text())!=result:raise ValueError('Frozen reservation changed')
    p.write_text(json.dumps(result,indent=2)+'\n');print(result['counts'])


if __name__=='__main__':main()
