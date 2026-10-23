"""Presentation only: immutable classification, provisional temporal feedback gate."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import json,math

class FeedbackPolicy:
    """One policy per user stream. Overlapping windows are not independent evidence."""
    def __init__(self,config=None,registry=None,root=None):
        root=Path(root) if root else Path(__file__).resolve().parents[2]
        self.config=json.loads((root/'config/feedback_policy.json').read_text())
        if config:self.config.update(config)
        c=self.config
        if not (0<=c['min_confidence']<=1 and 1<=c['min_votes']<=c['window_size']<=120 and 1<=c['min_consecutive']<=c['window_size'] and math.isfinite(c['min_duration_seconds']) and c['min_duration_seconds']>=0 and math.isfinite(c['max_gap_seconds']) and c['max_gap_seconds']>0):
            raise ValueError('Invalid feedback policy settings')
        if c['neutral_text']!='Keep going.':raise ValueError('Neutral text must remain non-corrective: Keep going.')
        self.registry=registry if registry is not None else json.loads((root/'config/model_registry.json').read_text())
        self.reset()

    def reset(self):
        self.history=deque(maxlen=self.config['window_size']);self.last_time=None;self.context=None

    def apply(self,classification,timestamp):
        """Returns nested raw classification + UI feedback. Never changes raw status/scores."""
        raw=deepcopy(classification)
        display=dict(text=self.config['neutral_text'],corrective=False,error_type=None,reason='not_persistent_or_reliable',provisional=True)
        if not isinstance(raw,dict):
            return dict(classification=raw,display_feedback={**display,'reason':'invalid_feedback_input'},updated=True)
        out=dict(classification=raw,display_feedback=display,updated=bool(raw.get('updated',True)))
        if not out['updated']:
            display['reason']='no_new_inference';return out
        try:
            t=float(timestamp)
            if not math.isfinite(t):raise ValueError('timestamp')
            context=(raw.get('exercise'),raw.get('model_name'))
            if self.context!=context or (self.last_time is not None and (t<=self.last_time or t-self.last_time>self.config['max_gap_seconds'])):self.reset()
            self.context=context;self.last_time=t
            exercise=self.registry.get(raw.get('exercise'),{})
            if raw.get('status')!='ERROR' or not exercise.get('error_feedback_enabled',False):
                self.history.clear();display['reason']='neutral_for_'+str(raw.get('status','UNKNOWN'));return out
            mapping=exercise.get('feedback',{});scores=(raw.get('model_outputs') or {}).get('error_types',{});reliable={}
            for name in raw.get('error_types',[]):
                value=scores.get(name,raw.get('confidence') if name==raw.get('error_type') else None)
                if name in mapping and isinstance(value,(int,float)) and math.isfinite(value) and self.config['min_confidence']<=value<=1:reliable[name]=float(value)
            if not reliable:
                self.history.clear();return out
            self.history.append((t,set(reliable)))
            persistent=[]
            for name in reliable:
                votes=sum(name in names for _,names in self.history);consecutive=0;since=t
                for when,names in reversed(self.history):
                    if name not in names:break
                    consecutive+=1;since=when
                if votes>=self.config['min_votes'] and consecutive>=self.config['min_consecutive'] and t-since+1e-9>=self.config['min_duration_seconds']:persistent.append(name)
            if persistent:
                name=max(persistent,key=lambda v:reliable[v]);display.update(text=mapping[name],corrective=True,error_type=name,reason='reliable_persistent_error')
            return out
        except (TypeError,ValueError,KeyError,AttributeError):
            self.reset();display['reason']='invalid_feedback_input';return out
