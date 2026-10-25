"""준비 기준 및 관측된 복귀 시점 특징. 등급/점수를 만들지 않는 독립 후보."""
import numpy as np

VERSION='shoulder-ready-return-relative-v1'
JOINTS=[11,12,13,14,15,16,23,24]
FEATURE_NAMES=['left_elbow_return_deficit_deg','right_elbow_return_deficit_deg',
 'left_elevation_return_deficit_deg','right_elevation_return_deficit_deg',
 'left_wrist_return_height_delta_torso','right_wrist_return_height_delta_torso']
TRUNK_NAMES=['absolute_tilt_p50_deg','absolute_tilt_p90_deg','absolute_tilt_p95_deg',
 'ready_relative_tilt_p50_deg','ready_relative_tilt_p90_deg','ready_relative_tilt_p95_deg']

def angle(x,a,b,c):
 u=x[:,a]-x[:,b];v=x[:,c]-x[:,b];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
 out=np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/np.maximum(den,1e-12),-1,1)));out[den<1]=np.nan
 return out

def extract(samples,width=640,height=480):
 """실제 슬롯을 유지. 정상하단 또는 후속재상승이 확인될 때만 복귀창 반환.

 누락/잘린하강/사용자종료는 복귀실패 정답이 아니다. 고정판정기 아닌 특징후보다.
 """
 if not samples:return {'status':'held','reason':'empty_input','version':VERSION}
 ticks=np.array([t for t,_ in samples],int)
 if np.any(np.diff(ticks)<=0):raise ValueError('strictly increasing slots required')
 x=np.full((ticks[-1]-ticks[0]+1,33,2),np.nan)
 x[ticks-ticks[0]]=np.asarray([p for _,p in samples],float)
 lo=np.array([width,height])*.02;hi=np.array([width,height])*.98
 bad=~np.isfinite(x).all(2)|((x<lo)|(x>hi)).any(2);x[bad]=np.nan
 valid=np.isfinite(x[:,JOINTS]).all((1,2));coverage=float(valid.mean())
 def hold(reason):return dict(status='held',reason=reason,version=VERSION,coverage=coverage)
 if len(x)<10 or coverage<.8:return hold('insufficient_actual_slot_observation')
 sh=x[:,[11,12]].mean(1);hip=x[:,[23,24]].mean(1);torso=np.maximum(np.linalg.norm(sh-hip,axis=1),30.)
 elbows=np.stack([angle(x,11,13,15),angle(x,12,14,16)],1)
 elevations=np.stack([angle(x,23,11,13),angle(x,24,12,14)],1)
 wrist=(x[:,[11,12],1]-x[:,[15,16],1])/torso[:,None]
 elbow_height=(x[:,[11,12],1]-x[:,[13,14],1])/torso[:,None]
 low=valid&(elbows.max(1)<=110)&(elbow_height.max(1)<.2)
 high=valid&(elbows.min(1)>=150)&(wrist.min(1)>=.35)&(elbow_height.min(1)>=.05)
 ready=next((i for i in range(5,len(x)) if low[i-5:i+1].all()),None)
 if ready is None:return hold('no_observed_ready_reference')
 top=next((i for i in range(ready+2,len(x)) if high[i-1:i+1].all()),None)
 if top is None:return hold('no_observed_bilateral_top')
 end=next((i for i in range(top+3,len(x)) if low[i-2:i+1].all()),None);kind='rack_observed'
 if end is None:
  # A partial return can be bracketed by the NEXT ascent. An unclosed end alone
  # never proves failure. Require 3+3 observed frames and an actual lower valley.
  signal=elevations.mean(1)
  for i in range(top+3,len(x)-3):
   if not valid[i-2:i+4].all():continue
   valley=float(np.median(signal[i-2:i+1]));next_lift=float(np.median(signal[i+1:i+4]))
   if float(np.max(signal[top-1:i+1]))-valley>=15 and next_lift-valley>=8 and np.all(signal[i+1:i+4]>valley+4):
    end=i;kind='lower_turnaround_with_observed_next_ascent';break
 if end is None:return hold('return_not_closed_or_observable')
 # Never impute across an in-attempt tracking interruption.
 if not valid[ready-5:end+1].all():return hold('missing_joint_inside_phase_context')
 reference=np.arange(ready-5,ready+1);ret=np.arange(end-2,end+1)
 ref_el=np.median(elbows[reference],0);ref_elev=np.median(elevations[reference],0)
 f=np.r_[np.median(elbows[ret],0)-ref_el,np.median(elevations[ret],0)-ref_elev,np.median(wrist[ret],0)-np.median(wrist[reference],0)]
 tilt=np.degrees(np.arctan2(sh[:,0]-hip[:,0],hip[:,1]-sh[:,1]));baseline=float(np.median(tilt[reference]));active=tilt[ready+1:end+1]
 trunk=np.r_[np.percentile(abs(active),[50,90,95]),np.percentile(abs(active-baseline),[50,90,95])]
 return dict(status='features_ready',version=VERSION,coverage=coverage,reference_ticks=(reference+ticks[0]).tolist(),top_tick=int(top+ticks[0]),return_ticks=(ret+ticks[0]).tolist(),closure=kind,feature_names=FEATURE_NAMES,return_features=f.tolist(),trunk_feature_names=TRUNK_NAMES,trunk_features=trunk.tolist(),grade=None,score=None,label_inferred=False)
