"""Causal bounded post-completion context for offline integration candidates.

Tracker completion and ready-relative return closure are different events.
This queue changes neither the repetition count nor any score/model/label.
"""
from numbers import Integral
from observation_features_v4 import extract_return

class ReturnContextQueue:
    def __init__(self,max_context_slots=3):
        if type(max_context_slots) is not int or not 0<=max_context_slots<=3:
            raise ValueError('engineering candidate supports only 0..3 actual 10Hz slots')
        self.max_context_slots=max_context_slots
        self.pending=[]
        self.last_tick=None

    def _result(self,p,reason=None):
        f=p['features']
        return dict(attempt_id=p['id'],original_window=p['window'],
                    evaluated_at_tick=p['samples'][-1][0],
                    added_actual_ticks=[t for t,_ in p['samples'] if t>p['window'][1]],
                    features=f,context_reason=reason,
                    feature_input_samples=p['samples'],grade=None,score=None,
                    production_replacement=False)

    def _evaluate(self,p):
        p['features']=extract_return(p['samples'],p['width'],p['height'],p['view'])
        f=p['features']
        if f['status']=='features_ready' and f['top_tick']>p['window'][1]:
            p['features']=dict(status='held',reason='later_attempt_peak_not_original_attempt',
                               version=f['version'])
        return p['features']

    def submit(self,attempt_id,segment,width=640,height=480,view='front'):
        if not segment:raise ValueError('completed original segment required')
        if any(p['id']==attempt_id for p in self.pending):raise ValueError('duplicate pending attempt')
        if any(not isinstance(t,Integral) or isinstance(t,bool) for t,_ in segment):
            raise ValueError('original integer slots required; no clock rounding')
        samples=[(int(t),xy.copy()) for t,xy in segment]
        if self.last_tick is not None and samples[-1][0]<self.last_tick:
            raise ValueError('cannot submit a completion in the past')
        self.last_tick=samples[-1][0]
        p=dict(id=attempt_id,samples=samples,window=[samples[0][0],samples[-1][0]],
               width=width,height=height,view=view)
        f=self._evaluate(p)
        if f['status']=='features_ready' or f.get('reason')!='return_not_closed_or_observable' or self.max_context_slots==0:
            return self._result(p)
        self.pending.append(p)
        return None

    def push(self,tick,xy):
        if type(tick) is not int or (self.last_tick is not None and tick<=self.last_tick):
            raise ValueError('strictly increasing actual integer slots required')
        self.last_tick=tick;out=[];keep=[]
        for p in self.pending:
            deadline=p['window'][1]+self.max_context_slots
            if tick>deadline:
                out.append(self._result(p,'post_context_timeout_no_future_frame_borrowed'));continue
            p['samples'].append((tick,xy.copy()));f=self._evaluate(p)
            if f['status']=='features_ready':out.append(self._result(p,'observed_causal_post_context'))
            elif f.get('reason')!='return_not_closed_or_observable' or tick>=deadline:
                out.append(self._result(p,'post_context_exhausted_or_unobservable'))
            else:keep.append(p)
        self.pending=keep
        return out

    def finish(self):
        out=[self._result(p,'recording_ended_before_return_context_closed') for p in self.pending]
        self.pending=[]
        return out
