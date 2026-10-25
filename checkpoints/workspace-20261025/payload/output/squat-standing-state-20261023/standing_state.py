"""Learn standing similarity relative to the observed initial upright template."""
from pathlib import Path
from functools import lru_cache
import numpy as np,joblib,hashlib
ROOT=Path(__file__).resolve().parent
VERSION='initial-standing-state-v1'
CORE_NAMES=['기준 대비 골반 높이 차','기준 대비 어깨 높이 차','몸통 길이 변화','몸통 기울기 변화']
KNEE_NAMES=['왼쪽 골반-무릎 높이 차 변화','오른쪽 골반-무릎 높이 차 변화']

def make_reference(xy):
 x=np.asarray(xy,float);mask=np.isfinite(x[:,[11,12,23,24]]).all((1,2));x=x[mask]
 if len(x)<2:return None
 p=np.full((33,2),np.nan)
 for j in range(33):
  if np.isfinite(x[:,j]).all(1).any():p[j]=np.nanmedian(x[:,j],axis=0)
 s=p[[11,12]].mean(0);h=p[[23,24]].mean(0);scale=float(np.linalg.norm(h-s))
 if scale<1:return None
 return dict(points=p,scale=scale)

def features(points,reference,width=None,height=None):
 x=np.asarray(points,float).copy()
 if x.shape!=(33,2) or reference is None:return None
 if width is not None and height is not None:x[(x[:,0]<0)|(x[:,0]>=width)|(x[:,1]<0)|(x[:,1]>=height)]=np.nan
 if not np.isfinite(x[[11,12,23,24]]).all():return None
 ref=np.asarray(reference['points']);scale=reference['scale'];s=x[[11,12]].mean(0);h=x[[23,24]].mean(0);rs=ref[[11,12]].mean(0);rh=ref[[23,24]].mean(0)
 if not np.isfinite(scale) or scale<1:return None
 if abs(h[0]-rh[0])/scale>.45:return None
 ratio=float(np.linalg.norm(h-s))/scale
 if not .4<ratio<2.5:return None
 core=[abs(float(h[1]-rh[1]))/scale,abs(float(s[1]-rs[1]))/scale,abs(ratio-1),abs(float((s-h)[0]-(rs-rh)[0]))/scale]
 knee=None
 if np.isfinite(x[[25,26]]).all() and np.isfinite(ref[[25,26]]).all():knee=core+[abs(float((x[25+j,1]-x[23+j,1])-(ref[25+j,1]-ref[23+j,1])))/scale for j in [0,1]]
 return dict(core=core,knee=knee)

@lru_cache(maxsize=1)
def load():
 p=ROOT/'model.joblib';return joblib.load(p),hashlib.sha256(p.read_bytes()).hexdigest()

def predict_frame(points,reference,width=None,height=None,bundle=None):
 f=features(points,reference,width,height)
 if f is None:return None
 b,sha=load() if bundle is None else (bundle,None);kind='core';m=b['models'][kind];prob=float(m.predict_proba([f[kind]])[0,list(m.classes_).index(1)])
 return dict(standing_probability=prob,standing=prob>=b['threshold'],feature_bank=kind,features=f[kind],model_sha256=sha,feature_version=VERSION)

def evaluate_return(xy,reference=None,width=None,height=None,phase='complete',bundle=None):
 """A standing state only means return after a separately observed descent."""
 x=np.asarray(xy,float)
 if x.ndim!=3 or x.shape[1:]!=(33,2) or len(x)<10:return dict(status='unobserved',ai_grade=None)
 ref=reference or make_reference(x[:6])
 if ref is None:return dict(status='reference_missing',ai_grade=None)
 valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
 if valid.mean()<.8 or valid[-3:].sum()<2:return dict(status='unobserved',ai_grade=None)
 hips=x[:,[23,24]].mean(1);drop=(hips[:,1]-ref['points'][[23,24]].mean(0)[1])/ref['scale'];bottom=int(np.nanargmax(np.where(valid,drop,np.nan)))
 if np.nanmax(drop)<.08:return dict(status='no_descent',ai_grade=None)
 if phase!='complete':return dict(status='return_pending',ai_grade=None)
 if bottom>=len(x)-3:return dict(status='return_pending',ai_grade=None)
 # Latest top may occur just before a new descent. Use only the final quarter,
 # following the bottom; no earlier repetition may lend its standing result.
 begin=max(bottom+1,len(x)-max(3,len(x)//4));windows=[]
 predictions=[predict_frame(p,ref,width,height,bundle) for p in x[begin:]]
 for i in range(begin,len(x)-2):
  ps=predictions[i-begin:i-begin+3]
  if all(p is not None for p in ps):windows.append((float(np.mean([p['standing_probability'] for p in ps])),i,ps))
 if not windows:return dict(status='unobserved',ai_grade=None)
 probability,index,ps=max(windows,key=lambda a:a[0]);b,sha=load() if bundle is None else (bundle,None);confirmed=all(p['standing'] for p in ps)
 # A learned standing confidence is an experimental /20 proxy, not probability
 # that a whole squat is correct. No Rule Base grade is copied here.
 grade=2*probability
 return dict(status='close' if confirmed else 'different',ai_grade=grade,ai_points=round(grade*10,1),standing_confirmed=confirmed,standing_probability=probability,reference_joints={str(j):ref['points'][j].tolist() for j in [11,12,23,24]},reference_scale=float(ref['scale']),frame_evidence=ps,return_window=[index,index+2],bottom_index=bottom,feature_bank=ps[0]['feature_bank'],model_sha256=sha,version=VERSION,method='학습한 초기 기립 상태 유사도 · 하강 후 연속3표본 확인',score_meaning='학습한 기립 상태 신뢰도×20의 실험 점수',gpt_used=False,rule_used_in_ai_score=False)
