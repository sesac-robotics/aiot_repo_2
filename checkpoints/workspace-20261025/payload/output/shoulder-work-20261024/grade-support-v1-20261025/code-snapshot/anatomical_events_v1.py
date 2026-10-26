"""Causal upper-to-lower-turn events, including partial turns without initial ready.

No grade/score inference. Requires a following ascent; recording end alone is
not a closed event. This is a development event detector, not an approved model.
"""
import numpy as np
from collections import deque
from phase_features_v3 import angle,JOINTS
from observation_features_v4 import dense_samples,quality
from anatomical_turn_features_v1 import extract
class AnatomicalTurnDetector:
    def __init__(self,width=640,height=480,view='front',posture='standing',source_fps=10):
        self.width,self.height,self.view,self.posture,self.fps=width,height,view,posture,source_fps
        self.last_tick=None;self.last_frame=None;self._reset()
    def _reset(self):
        self.history=deque(maxlen=151);self.has_upper=False;self.peak_index=None;self.peak_value=None
    def push(self,tick,xy,native_frame):
        if not isinstance(tick,(int,np.integer)) or not isinstance(native_frame,(int,np.integer)):raise ValueError('integer clocks required')
        if self.last_tick is not None and (tick<=self.last_tick or native_frame<=self.last_frame):raise ValueError('strictly increasing actual clocks required')
        if self.last_tick is not None and tick!=self.last_tick+1:self._reset()
        self.last_tick,self.last_frame=int(tick),int(native_frame)
        ts,x=dense_samples([(int(tick),xy)],self.width,self.height)
        if self.view!='front' or not np.isfinite(x[:,JOINTS]).all() or not quality(ts,x)['passed']:
            self._reset();return None
        elev=np.array([angle(x,23,11,13)[0],angle(x,24,12,14)[0]])
        el=np.array([angle(x,11,13,15)[0],angle(x,12,14,16)[0]])
        signal=float(elev.mean());upper=bool(np.any((elev>=145)&(el>=150)))
        self.history.append(dict(tick=int(tick),frame=int(native_frame),xy=x[0],signal=signal,upper=upper))
        if len(self.history)==151:self._reset();return None
        h=list(self.history)
        if not self.has_upper:
            if len(h)>=3 and all(r['upper'] for r in h[-3:]):
                self.history=deque(h[-3:],maxlen=151);self.has_upper=True
                h=list(self.history);self.peak_index=int(np.argmax([r['signal'] for r in h]));self.peak_value=h[self.peak_index]['signal']
            return None
        if signal>self.peak_value:self.peak_value=signal;self.peak_index=len(h)-1
        if len(h)<9:return None
        valley=h[-6:-3];following=h[-3:];v=float(np.median([r['signal'] for r in valley]))
        first=len(h)-6
        if first<=self.peak_index or max(r['signal'] for r in valley)>self.peak_value-15:return None
        if np.median([r['signal'] for r in following])-v<8 or not all(r['signal']>v+4 for r in following):return None
        # Three already observed source frames around the prior upper maximum.
        ps=max(0,min(self.peak_index-1,first-3));peak=h[ps:ps+3]
        event=dict(protocol='shoulder_press',laterality='bilateral',posture=self.posture,
                   closure='observed_next_ascent',peak_frames=[peak[0]['frame'],peak[-1]['frame']],
                   turn_frames=[valley[0]['frame'],valley[-1]['frame']],
                   next_ascent_frames=[following[0]['frame'],following[-1]['frame']])
        samples=[(r['tick'],r['xy']) for r in h[ps:]]
        f=extract(samples,event,self.fps,self.width,self.height,self.view,native_frame_numbers=[r['frame'] for r in h[ps:]])
        result=dict(event=event,features=f,confirmed_at_tick=int(tick),turn_ticks=[r['tick'] for r in valley],
                    label_created=False,grade=None,score=None,production_approved=False)
        # The confirming ascent belongs to the next attempt; retain only its actual frames.
        self._reset();self.history=deque(following,maxlen=151)
        if all(r['upper'] for r in following):
            self.has_upper=True;self.peak_index=int(np.argmax([r['signal'] for r in following]));self.peak_value=following[self.peak_index]['signal']
        return result
