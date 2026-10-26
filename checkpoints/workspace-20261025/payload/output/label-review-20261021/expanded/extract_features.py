"""직접 라벨 구간의 MediaPipe Heavy 특징을 RTX에서 순차 추출한다.

점수를 계산하지 않는다. 자세 점수는 직접 검토한 라벨에만 있다.
"""
from pathlib import Path
import sys
import json
import datetime
import time
import numpy as np

BASE=Path('/home/jayden/SeSACPAI-Mid')
OUT=BASE/'output/label-review-20261021/expanded'
sys.path.insert(0,str(BASE/'overnight-posture-pipeline'))
import features
import safety_guard


def extract(exercise="pushups"):
    """라벨 구간만 추출하고 낮은 관측률·고온·센서 오류에서는 중단한다.

    Args:
        exercise: 라벨 파일을 선택하는 영어 운동 코드.
    Returns:
        None: features/*.npz와 특징 추출 보고서를 저장한다.
    Raises:
        RuntimeError: 센서 누락 또는 CPU 95℃·GPU 85℃ 이상일 경우.
    """
    labels=json.loads((OUT/f'{exercise}_labels.json').read_text());dest=OUT/'features';dest.mkdir(exist_ok=True);report=[]
    for label in labels:
        if not label['eligible_for_experimental_training']:continue
        reading=safety_guard.snapshot()
        if reading['cpu_c'] is None or reading['gpu_c'] is None or reading['cpu_c']>=95 or reading['gpu_c']>=85:
            raise RuntimeError('특징 추출 온도 점검 중단: '+str(reading))
        target=dest/(label['id']+'.npz')
        row=dict(path=label['video_path'],id=label['id'],start_frame=label['segment']['start_frame'],end_frame=label['segment']['end_frame'])
        t=time.monotonic()
        if target.exists():
            z=np.load(target);valid=bool(z['X'].size);samples=int(z['samples']);coverage=float(z['coverage'])
        else:
            x,samples,coverage=features.dense_clip(row);valid=x is not None
            np.savez_compressed(target,X=[] if x is None else x,samples=samples,coverage=coverage)
        report.append(dict(id=label['id'],valid=valid,samples=samples,coverage=coverage,seconds=time.monotonic()-t,temperature_before=reading))
        (OUT/f'{exercise}_feature_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(label['id'],valid,samples,round(coverage,2),flush=True)
    print('완료',sum(r['valid'] for r in report),'/',len(report),flush=True)


if __name__=='__main__':extract(sys.argv[1] if len(sys.argv)>1 else 'pushups')
