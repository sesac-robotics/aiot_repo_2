"""별도 환경에서 5종 원본→Heavy→점수→MP4 실행 경로를 검증한다."""

import json
import os
import pathlib
import subprocess

import audit
from resources import check_resources

ROOT = audit.BASE / "output/five-exercise-recovery-20261022"
OUT = ROOT / "team_video_smoke"


def run() -> dict:
    """기존 검토 구간만 사용해 배포 코드의 실제 영상 처리를 검사한다.

    Returns:
        dict: 운동별 결과 JSON·관절 MP4 연결과 실행 종료 코드.
        같은 자료의 재실행이며 새로운 직접 라벨·독립 정확도 검증이 아니다.

    Raises:
        OSError: 원본·별도 Python·출력 접근 실패.
        subprocess.SubprocessError: 배포 CLI 실패 또는 180초 제한 초과.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    cases=audit.read_json(ROOT/'team_portability/cases.json')['records']
    selections=[]
    for exercise in ('pushups','squats','shoulder','plank','ytw'):
        subset=[r for r in cases if r['exercise']==exercise]
        chosen=max(subset,key=lambda r:(r['expected']['ai_score'] is not None and r['expected']['rule_score'] is not None,
            sum(g is not None for g in r['expected']['ai_levels'])))
        selections.append(chosen)
    results=[]
    for case in selections:
        check_resources()
        import numpy as np
        with np.load(case['landmarks']) as data:
            first,last=int(data['frame_numbers'][0]),int(data['frame_numbers'][-1])
        options=dict(case['options'],exercise=case['exercise'],start_frame=first,end_frame=last)
        if case['exercise']=='squats':
            labels=audit.read_json(ROOT/'multiview_squat/batch2/score_annotations.json')
            row=next(r for r in labels if r['id']=='subject_012_squat_good_batch2_rep1')
            case=dict(case,video_path=row['video_path'],id=row['id'])
            options=dict(exercise='squats',view='side',
                start_frame=row['segment']['start_frame'],end_frame=row['segment']['end_frame'],
                front_video_path=row['front_path'],front_start_frame=row['front_segment']['start_frame'],
                front_end_frame=row['front_segment']['end_frame'],paired_view_reviewed=True,
                lag_frames=row['alignment']['lag_frames'],complete_attempt=row['attempt_complete'])
        request=dict(video_path=case['video_path'],labels=[],labels_descriptive=[],split='train',assessment_input=options)
        path=OUT/(case['exercise']+'_request.json');audit.save_json(path,request)
        output=OUT/case['exercise']
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        command=[str(audit.BASE/'.posture-score-verify/bin/python'),'-m','app.scoring',
            '--request',str(path),'--output',str(output),'--heavy-model',str(audit.BASE/'mediapipe-pilot/models/heavy.task'),
            '--delegate','CPU','--overlay']
        process=subprocess.run(command,cwd=audit.BASE/'team-posture-scoring/Server/AIService',env=env,
            text=True,capture_output=True,timeout=180,check=True)
        result=audit.read_json(output/'result.json')
        results.append(dict(exercise=case['exercise'],source_case=case['id'],result=str(output/'result.json'),
            html=str(output/'index.html'),video=str(output/'review.mp4'),
            ai_score=result['ai_score'],rule_score=result['rule_score'],
            extraction=result['extraction'],stdout=process.stdout,output_complete=result['output_complete']))
        audit.save_json(OUT/'verification.json',dict(records=results,completed=False))
    result=dict(records=results,completed=True,independent_final_test=False,
        actual_media_heavy_score_overlay_verified=True,browser_playback_verified=False)
    audit.save_json(OUT/'verification.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False))
