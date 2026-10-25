"""직접 라벨링한 구간의 관절 특징을 순차 추출한다. 점수를 자동 생성하지 않는다."""
import json
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/jayden/SeSACPAI-Mid')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'overnight-posture-pipeline'))
import features
import safety_guard


def extract():
    """온도와 관측률을 점검하면서 MediaPipe Heavy 특징을 RTX에서 추출한다.

    Returns:
        None: NPZ와 feature_report.json을 저장한다.
    Raises:
        RuntimeError: 센서 오류 또는 CPU 95℃·GPU 85℃ 이상일 경우.
    """
    dest=OUT/'features';dest.mkdir(exist_ok=True);report=[]
    for row in json.loads((OUT/'labels.json').read_text()):
        if not row['eligible_for_experimental_training']:continue
        temperature=safety_guard.snapshot()
        if temperature['cpu_c'] is None or temperature['gpu_c'] is None or temperature['cpu_c']>=95 or temperature['gpu_c']>=85:raise RuntimeError(str(temperature))
        target=dest/(row['id']+'.npz');key=dict(sha=row['video_sha256'],segment=row['segment'],schema='Heavy_10fps_BODY17_world_normalized_32timepoints_stats')
        if target.exists():
            with np.load(target) as z:
                assert str(z['cache_key'])==json.dumps(key,sort_keys=True)
                valid=bool(z['X'].size);samples=int(z['samples']);coverage=float(z['coverage'])
        else:
            source=dict(path=row['video_path'],id=row['id'],start_frame=row['segment']['start_frame'],end_frame=row['segment']['end_frame'])
            x,samples,coverage=features.dense_clip(source);valid=x is not None
            np.savez_compressed(target,X=[] if x is None else x,samples=samples,coverage=coverage,cache_key=json.dumps(key,sort_keys=True))
        report.append(dict(id=row['id'],valid=valid,samples=samples,coverage=coverage,temperature_before=temperature));(OUT/'feature_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(row['id'],valid,samples,round(coverage,2),flush=True)
    print('완료',sum(r['valid'] for r in report),'/',len(report),flush=True)


if __name__=='__main__':extract()
