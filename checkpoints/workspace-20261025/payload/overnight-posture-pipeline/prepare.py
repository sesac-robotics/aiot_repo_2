"""원본 오류 라벨과 AI 점수 초안 대상을 분리하고 참여자별 분할을 고정한다."""
import collections
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from common import BASE, OUT, save, digest

sys.path.insert(0, str(BASE / 'mediapipe-pilot/qevd_coach'))
import catalog


def prepare():
    """5종 오류 명세와 운동별 유효 70개를 확보하기 위한 대체 점수 후보를 저장해 반환한다.

    Returns:
        원본 오류 주석 행 목록과 점수 초안 후보 목록. 점수는 생성하지 않는다.
    Raises:
        AssertionError: 참여자 분할 중복 또는 영상 누락.
        OSError: 원본 데이터 접근 실패.
    """
    profiles = copy.deepcopy(catalog.PROFILES)
    profiles.pop('bird_dog')
    profiles['ytw'] = dict(name='Y-T-W 레이즈', source='standing ytw',
        reference=['torso parallel to the ground, thumbs pointed up'], errors={
        'arms too low':['팔 높이 부족','팔이 지정한 Y·T·W 위치에 도달하는지 확인하세요.'],
        'thumbs pointed down':['엄지가 아래를 향함','지정한 손 방향을 확인하세요.'],
        'thumbs pointed sideway':['엄지가 옆을 향함','지정한 손 방향을 확인하세요.'],
        'hands flat':['손 방향 차이','기준 동작의 손 방향을 확인하세요.'],
        'too fast':['동작 속도 빠름','기준 동작과 비교해 속도를 확인하세요.'],
        'too slow':['동작 속도 느림','기준 동작과 비교해 속도를 확인하세요.']})
    catalog.PROFILES = profiles
    qrows = catalog.build_catalog()['items']
    items = []
    for row in qrows:
        items.append(dict(id='qevd_'+row['id'], original_id=row['id'], path=row['path'],
            person='qevd_'+row['worker'], exercise=row['exercise'], split=row['split'],
            errors=row['errors'], reference=row['reference'], source_labels=row['labels'],
            label_source='QEVD_provider', label_semantics='지정 기준 속성과 명시된 오류의 비교. 오류 부재를 정상으로 간주하지 않음.'))
    qroot = BASE/'data/qevd-selected'
    test_workers=set(json.loads((qroot/'manifests/official_test_workers.json').read_text()))
    for row in csv.DictReader((qroot/'manifests/short_clips.csv').open()):
        names={s.partition(' - ')[0] for s in json.loads(row['labels'])}
        if not names & {'standing twy','standing tyw','standing wty','standing wyt','standing ywt'}:
            continue
        worker=row['worker_id']
        if not worker or (row['split']!='test' and worker in test_workers):continue
        split='test' if row['split']=='test' else ('validation' if int(hashlib.sha256(worker.encode()).hexdigest()[:8],16)%5==0 else 'train')
        items.append(dict(id='qevd_'+row['video_id'],original_id=row['video_id'],path=str(qroot/row['video_path']),person='qevd_'+worker,exercise='ytw',split=split,errors=['sequence_order_mismatch'],reference=False,source_labels=json.loads(row['labels']),label_source='QEVD_action_name_derived',label_semantics='YTW를 목표로 지정했을 때 순서 이름이 다른 과제. 제공자가 자세 오류로 판정한 라벨은 아님.'))
    shoulder=BASE/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise-wise/shoulder'
    persons=sorted({p.stem.split('_')[1] for p in shoulder.rglob('*.mp4')})
    random.Random(37).shuffle(persons)
    assignments={p:('train' if i<15 else 'validation' if i<20 else 'test') for i,p in enumerate(persons)}
    for p in sorted(shoulder.rglob('*.mp4')):
        quality,view=p.relative_to(shoulder).parts[:2]
        items.append(dict(id=p.stem,original_id=p.stem,path=str(p),person='shoulder_'+p.stem.split('_')[1],exercise='shoulder',split=assignments[p.stem.split('_')[1]],errors=['form_bad'] if quality=='bad' else [],reference=quality=='good',source_labels=[quality],label_source='MultiView_provider',label_semantics='정상·오류 이진 구분만 제공. 세부 오류는 AI 초안과 별도 관리.'))
    ids=[r['id'] for r in items]
    assert len(ids)==len(set(ids))
    for r in items:assert Path(r['path']).is_file(),r['path']
    people={s:{r['person'] for r in items if r['split']==s} for s in ['train','validation','test']}
    assert not people['train'] & (people['validation']|people['test'])
    assert not people['validation'] & people['test']
    # 오류 분류용 좁은 하위집합에 한정하지 않고 원본 4종 전체를 점수 후보로 사용한다.
    score_pool=[]
    aliases={'pushups':'pushups','squats':'squats','elbow plank':'plank','standing ytw':'ytw'}
    for row in csv.DictReader((qroot/'manifests/short_clips.csv').open()):
        labels=json.loads(row['labels']);names={v.partition(' - ')[0] for v in labels}
        matched=names & aliases.keys()
        if len(matched)!=1 or len(names)!=1:continue
        worker=row['worker_id']
        if not worker or (row['split']!='test' and worker in test_workers):continue
        ex=aliases[next(iter(matched))]
        split='test' if row['split']=='test' else ('validation' if int(hashlib.sha256(worker.encode()).hexdigest()[:8],16)%5==0 else 'train')
        reference=any(v.endswith(' - no obvious issue') or v.endswith(' - average') for v in labels)
        score_pool.append(dict(id='qevd_'+row['video_id'],original_id=row['video_id'],path=str(qroot/row['video_path']),person='qevd_'+worker,exercise=ex,split=split,source_labels=labels,reference=reference,label_source='QEVD_provider',priority=2))
    multiview=shoulder.parent
    for exercise,folder in [('pushups','push_up'),('shoulder','shoulder')]:
        for p in sorted((multiview/folder).rglob('*.mp4')):
            quality,view=p.relative_to(multiview/folder).parts[:2]
            score_pool.append(dict(id=p.stem,original_id=p.stem,path=str(p),person='shoulder_'+p.stem.split('_')[1],exercise=exercise,split=assignments[p.stem.split('_')[1]],source_labels=[quality],reference=quality=='good',label_source='MultiView_provider',priority=0,recording_group='_'.join(p.stem.split('_')[:-1])))
    rehab=BASE/'data/rehab26-squat'
    videos={v['video_id']:v for v in csv.DictReader((rehab/'manifests/videos.csv').open())}
    seen_reps=set()
    for rep in csv.DictReader((rehab/'manifests/repetitions.csv').open()):
        key=(rep['person'],rep['repetition_number'],rep['person_orientation'],rep['first_frame'],rep['last_frame'])
        if key in seen_reps:continue
        seen_reps.add(key)
        orientation=float(rep['person_orientation'])
        cam=max([1,2,3],key=lambda c:abs((c-1)*45-orientation))
        video=videos[f"ex6_{rep['person']}_cam{cam}"]
        first,last=int(rep['first_frame']),int(rep['last_frame'])
        score_pool.append(dict(id=f"rehab_{rep['person']}_{first}_{last}",original_id=video['video_id'],path=str(rehab/video['video_path']),person='rehab_'+rep['person'],exercise='squats',split='test' if rep['person']=='Honza' else 'validation' if rep['person']=='Mima' else 'train',source_labels=[rep['correctness_score'],rep['error_description']],reference=rep['correctness_score']=='1',label_source='REHAB26_provider',priority=0,start_frame=first,end_frame=last,recording_group=f"ex6_{rep['person']}_{first}_{last}"))
    rng=random.Random(1701);candidates=[]
    for ex in ['pushups','squats','plank','ytw','shoulder']:
        for split in ['train','validation','test']:
            pool=[r for r in score_pool if r['exercise']==ex and r['split']==split]
            rng.shuffle(pool)
            pool.sort(key=lambda r:r['priority'])
            candidates.extend(pool)
    assert len(candidates)==len({r['id'] for r in candidates})
    people={sp:{r['person'] for r in candidates if r['split']==sp} for sp in ['train','validation','test']}
    assert not people['train'] & (people['validation']|people['test'])
    assert not people['validation'] & people['test']
    save(OUT/'error_manifest.json',items);save(OUT/'score_candidates.json',candidates)
    save(OUT/'label_policy.json',dict(version='2026-10-20-pilot-v3',score_unit='clip_quality_0_to_100',
        score_source='local_visual_language_model_draft',review_status='unreviewed',
        scope='클립 단위 시각적 수행 품질 초안. 반복별 정밀 점수·전문가 채점·의학 평가 아님.',
        anchors={'90-100':'명확한 자세 문제 없이 기준 동작을 안정적으로 수행하는 모습', '70-89':'동작을 수행하지만 작은 정렬·범위·제어 문제가 관측됨', '40-69':'명확한 오류 또는 불충분한 수행이 관측됨', '0-39':'목표 동작을 크게 벗어나거나 수행을 유지하지 못함'},
        exclusions=['관절 또는 동작이 충분히 보이지 않음','목표 운동인지 판단 불가','시각적 근거 부족','신뢰도 0.65 미만'],
        separation='원본 오류 라벨은 점수 교사 모델에 전달하지 않음. 모델 입력에도 라벨·참여자·파일명·촬영 방향을 넣지 않음.',
        limitations='AI 교사 라벨과의 일치도는 사람 평가 정확도가 아님. 점수 기준과 라벨은 추후 팀 검토 필요.'))
    save(OUT/'preparation_report.json',dict(error_clips=len(items),score_candidates=len(candidates),
        counts={ex:dict(collections.Counter(r['split'] for r in items if r['exercise']==ex)) for ex in profiles.keys()|{'shoulder'}},
        score_candidate_counts={ex:dict(collections.Counter(r['label_source'] for r in candidates if r['exercise']==ex)) for ex in ['pushups','squats','plank','ytw','shoulder']},
        target_eligible_per_exercise=70,
        split_participants={s:len(v) for s,v in people.items()},disjoint_participants=True,
        manifest_sha256=digest(OUT/'error_manifest.json')))
    return items,candidates


if __name__=='__main__':
    a,b=prepare();print('prepared',len(a),len(b))
