"""Only the two improved checklist components; original totals remain a separate rubric."""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.util
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent/'squat-relabel-expanded-20261023'
@lru_cache(maxsize=1)
def load():
    report=__import__('json').loads((ROOT/'report.json').read_text())
    assert hashlib.sha256((ROOT/'candidate.joblib').read_bytes()).hexdigest()==report['model_sha256']
    bundle=joblib.load(ROOT/'candidate.joblib')
    assert hashlib.sha256((ROOT/'labels.frozen.json').read_bytes()).hexdigest()==bundle['label_sha256']
    spec=importlib.util.spec_from_file_location('accepted_criteria_features',ROOT/'criteria_features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return bundle['heads'],module.observed_features

def evaluate(xy):
    heads,features=load();f=features(xy);out=[]
    for key,method,title in [('trunk_balance','ai','몸통 좌우 기울기'),('heel_support','rule','뒤꿈치·발 지지')]:
        obs=f.get(key)
        item=dict(id=key,method=method,title=title,grade=None,maximum=2,status='미관측',coverage=0,experimental=True)
        if obs:
            h=heads[key]
            if method=='ai':grade=float(np.clip(h['model'].predict([obs['x']])[0],0,2))
            else:
                good,partial=h['boundaries'];v=obs['scalar'];grade=2 if v<=good else 1 if v<=partial else 0
            item.update(grade=round(grade,3),status='검증 중인 항목 평가',coverage=obs['coverage'])
        out.append(item)
    return dict(version='checklist-selective-v1',items=out,total=None,scope='몸통 AI·발 지지 규칙만 반영; 기존 총점과 다른 항목 기준',validation='1명 10회 개발 검증, 전문가 미확정')
