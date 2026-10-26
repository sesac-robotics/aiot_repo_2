"""새 항목의 관측 특징. 라벨·사람·카메라 번호·시간을 입력하지 않는다."""
import numpy as np

def quant(v):
 v=np.asarray(v,float);return [float(t) for t in np.percentile(v,[10,50,90])]

def observed_features(xy):
 x=np.asarray(xy,float);out={}
 torso=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
 if torso.mean()<.8:return out
 t=x[torso];s=t[:,[11,12]].mean(1);h=t[:,[23,24]].mean(1);scale=np.median(np.linalg.norm(h-s,axis=1))
 if scale<1:return out
 tilt=np.degrees(np.arctan2(h[:,0]-s[:,0],h[:,1]-s[:,1]));shoulder=np.degrees(np.arctan2(t[:,12,1]-t[:,11,1],abs(t[:,12,0]-t[:,11,0])))
 # 몸통: 절대 기울기와 시작 기준 변화, 어깨선 변화.
 vals=quant(abs(tilt-np.median(tilt[:3])))+quant(abs(shoulder-np.median(shoulder[:3])))+quant(abs(tilt))
 out['trunk_balance']=dict(x=vals,scalar=vals[2],higher=False,coverage=float(torso.mean()))
 hy=h[:,1];n=min(10,max(3,len(hy)//4));delta=abs(np.median(np.sort(hy[:n])[:3])-np.median(np.sort(hy[-n:])[:3]))/scale
 out['standing_return']=dict(x=[float(delta)],scalar=float(delta),higher=False,coverage=float(torso.mean()))
 knees=torso & np.isfinite(x[:,[25,26]]).all((1,2))
 if knees.mean()>=.8:
  a=x[knees];hipwidth=abs(a[:,24,0]-a[:,23,0]);ratio=abs(a[:,26,0]-a[:,25,0])/np.maximum(hipwidth,1);base=np.median(ratio[:3]);relative=ratio/max(.01,base)
  q=quant(relative);out['knee_inward']=dict(x=q+quant(ratio),scalar=q[0],higher=True,coverage=float(knees.mean()))
  hips=a[:,[23,24]].mean(1);depth=(hips[:,1]-np.median(hips[:3,1]))/scale;hk=(a[:,[25,26],1].mean(1)-hips[:,1])/scale
  out['descent_range']=dict(x=quant(depth)+quant(hk),scalar=float(np.percentile(depth,90)-np.percentile(depth,10)),higher=True,coverage=float(knees.mean()))
 feet=torso & np.isfinite(x[:,[29,30,31,32]]).all((1,2))
 if feet.mean()>=.8:
  a=x[feet];signals=[]
  for heel,toe in [(29,31),(30,32)]:
   length=np.linalg.norm(a[:,heel]-a[:,toe],axis=1);v=(a[:,toe,1]-a[:,heel,1])/np.maximum(1,length);signals.append(v-np.median(v[:3]))
  v=np.maximum(*signals);q=quant(v);out['heel_support']=dict(x=q,scalar=q[2],higher=False,coverage=float(feet.mean()))
 # 허리 전후 정렬·반동은 정면 2D에서 신뢰할 특징을 정의하지 않는다.
 return out
