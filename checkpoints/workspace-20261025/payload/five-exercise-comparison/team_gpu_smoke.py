"""보유 검토 구간으로 배포 NVIDIA 어댑터 경로만 확인한다."""
import json
import os
import pathlib
import subprocess

from resources import check_resources

BASE = pathlib.Path('/home/jayden/SeSACPAI-Mid')
ROOT = BASE / 'output/five-exercise-recovery-20261022'
WORK = BASE / 'team-posture-scoring/Server/AIService'


def main() -> None:
    """기존 구간을 재추출한다. 새 라벨·독립 시험으로 사용하지 않는다."""
    output = ROOT / 'team_gpu_smoke'
    output.mkdir(exist_ok=True)
    records = []
    for exercise in ('pushups','shoulder','plank','ytw'):
        check_resources()
        request=json.loads((ROOT/'team_video_smoke'/f'{exercise}_request.json').read_text())
        if exercise=='pushups':
            labels=json.loads((BASE/'output/label-review-20261021/all-five-v2/labels.json').read_text())
            row=next(r for r in labels if r['id']=='subject_001_push_up_good_side')
            request={k:row[k] for k in ('video_path','labels','labels_descriptive','split')}
            request['assessment_input']=dict(exercise='pushups',view='side',start_frame=row['segment']['start_frame'],end_frame=row['segment']['end_frame'],complete_attempt=True)
        path=output/f'{exercise}_request.json'
        path.write_text(json.dumps(request,ensure_ascii=False,indent=2))
        env=dict(os.environ)
        for key in ('PYTHONPATH','LD_PRELOAD','__EGL_VENDOR_LIBRARY_FILENAMES'):
            env.pop(key,None)
        completed=subprocess.run([str(WORK/'tools/posture_score/run_nvidia.sh'),str(BASE/'.posture-score-verify/bin/python'),
          '--request',str(path),'--output',str(output/exercise),'--heavy-model',str(BASE/'mediapipe-pilot/models/heavy.task'),'--overlay'],
          cwd=WORK,env=env,capture_output=True,text=True,timeout=180)
        (output/f'{exercise}.log').write_text(completed.stdout+completed.stderr)
        completed.check_returncode()
        r=json.loads((output/exercise/'result.json').read_text())
        records.append(dict(exercise=exercise,ai_score=r['ai_score'],rule_score=r['rule_score'],extraction=r['extraction'],output_complete=r['output_complete']))
        (output/'verification.json').write_text(json.dumps(dict(records=records,independent_final_test=False,completed=False),indent=2))
    (output/'verification.json').write_text(json.dumps(dict(records=records,independent_final_test=False,completed=True),indent=2))


if __name__=='__main__':
    main()
