"""Perspective-aware observational features. No labels, rule outputs, or identities."""
import numpy as np

def q(x):return [float(v) for v in np.percentile(x,[10,50,90])]
def features(xy,width,height,old_extract,three_items):
 x=np.asarray(xy,float).copy();outside=(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height);x[outside]=np.nan
 old=old_extract(x);out={}
 torso=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
 if torso.mean()>=.8 and torso.sum()>=10 and 'trunk_balance' in old:
  a=x[torso];s=a[:,[11,12]].mean(1);h=a[:,[23,24]].mean(1);scale=np.median(np.linalg.norm(s-h,axis=1));hw=np.maximum(1,abs(a[:,24,0]-a[:,23,0]));sw=abs(a[:,12,0]-a[:,11,0])
  v=old['trunk_balance']['x']+q(abs(s[:,1]-h[:,1])/scale)+q(sw/hw)+q(abs(h[:,0]-np.median(h[:3,0]))/scale)+q(abs(s[:,0]-h[:,0])/scale)
  out['trunk_balance']=dict(x=v,scalar=v[20],higher=False,coverage=float(torso.mean()),meaning='몸통 길이 대비 어깨·골반 좌우 편위; 투영 길이·폭을 AI 문맥 특징으로 분리')
 if 'knee_inward' in old:
  note=three_items.describe('squats',None,x)[2];core=np.asarray(note['feature'][:8],float)
  if np.isfinite(core).all() and core[7]>=.8:
   out['knee_inward']=dict(x=old['knee_inward']['x']+core.tolist(),scalar=float(core[5]),higher=False,coverage=float(core[7]),meaning='몸통 길이 대비 무릎의 골반–발목선 안쪽 편위·동적 안쪽 이동. 발끝 방향이나 뒤꿈치 지지의 합성 점수는 아님')
 return old,out
