"""Optional stance context. No ankle observation ever gates the core heads."""
import numpy as np

def stance_context(xy):
 x=np.asarray(xy,float);required=[23,24,25,26,27,28];valid=np.isfinite(x[:,required]).all((1,2))
 if valid.sum()<10 or valid.mean()<.8:return None
 a=x[valid];hip=a[:,[23,24],1].mean(1);bottom=hip>=np.percentile(hip,75)
 if bottom.sum()<3:return None
 width=abs(a[:,28,0]-a[:,27,0]);ratios=abs(a[:,26,0]-a[:,25,0])/np.maximum(1,width);base=float(np.median(ratios[:3]));b=ratios[bottom]
 if np.median(width)<5:return None
 q=lambda v:[float(n) for n in np.percentile(v,[10,50,90])]
 return dict(x=q(b)+q(b-base),scalar=float(np.median(b)),higher=True,coverage=float(valid.mean()),scope='무릎 간격 / 발목 지지 폭 · 하강 하단 정면 투영; 발끝 방향 판정 아님')
