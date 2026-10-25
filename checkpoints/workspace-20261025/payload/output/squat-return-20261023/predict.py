"""Separate return candidate inference; never substitutes a rule for an AI output."""
from pathlib import Path
import hashlib,json
import joblib,numpy as np
from features import extract,VERSION,NAMES
O=Path(__file__).resolve().parent

def predict(xy,baseline=None,width=None,height=None,allow_unapproved=False):
 b=joblib.load(O/'candidate.joblib');sha=hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest()
 if b['labels_sha256']!=sha or b['feature_version']!=VERSION:raise ValueError('모델·라벨·특징 버전 불일치')
 if not b['runtime_approved'] and not allow_unapproved:return dict(status='validation_pending',ai_grade=None,ai_points=None,reason='사람별 분리 검증 기준 미달',runtime_approved=False)
 f=extract(xy,baseline,width,height)
 if f is None:return dict(status='unobserved',ai_grade=None,ai_points=None,reason='복귀 항목의 관측·기준·동작 구간 부족',runtime_approved=b['runtime_approved'])
 raw=float(b['model'].predict([f['x']])[0]);grade=float(np.clip(raw,0,2))
 return dict(status='experimental_candidate',ai_grade=grade,ai_points=round(grade*10,1),maximum=20,features=f,model_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),labels_sha256=sha,feature_version=VERSION,training_support=b['train_support'],supported_grades=b['supported_grades'],runtime_approved=b['runtime_approved'],limitations=b['limitations'],rule_used_in_ai_score=False,gpt_used_in_inference=False)
