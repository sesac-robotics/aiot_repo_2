"""Classification only. Does not receive network packets or change repetition counts."""
from pathlib import Path
from collections import deque
import json,math
import numpy as np,torch
from .networks import MODELS,Hybrid
from .squat_features import features,resample
from .suite_features import prepare_landmarks

ALIASES={'squat':'squats','push-up':'pushups','pushup':'pushups','plank':'elbow_plank','dumbbell_shoulder_press':'shoulder_press'}
def summary(x):
 d=np.diff(x,axis=1);q=np.quantile(x,[.1,.5,.9],axis=1).transpose(1,0,2).reshape(len(x),-1)
 return np.concatenate([q,x.mean(1),x.std(1),np.mean(abs(d),1),np.std(d,1),*[v.mean(1) for v in np.array_split(x,4,axis=1)]],axis=1).astype(np.float32)

class PostureClassifier:
 def __init__(self,exercise='squats',architecture='cnn',root=None,device='cpu'):
  self.device=torch.device(device)
  self.root=Path(root) if root else Path(__file__).resolve().parents[2];self.exercise=ALIASES.get(exercise,exercise);self.architecture=architecture;self.load_error=None;self.loaded=False;self.length=32;self.fps=6
  try:
   if architecture not in {'cnn','gru'}:raise ValueError('architecture must be cnn or gru')
   self.config=json.loads((self.root/'config/model_registry.json').read_text())[self.exercise];self.meta=self.config['models'][architecture];self.required=self.config['required_joints'];self.length=self.meta['sequence_length'];self.fps=self.meta['sample_fps'];self.targets=self.meta['output_classes'];self.subtype_count=len(self.targets)-(self.exercise!='squats')
   self.ck=torch.load(self.root/self.meta['path'],map_location='cpu',weights_only=True);c=self.ck
   # 인계 메타데이터와 실제 체크포인트가 다르면 추측하지 않고 사용을 거부합니다.
   if (c['architecture']!=architecture or c['feature_count']!=self.meta['feature_count'] or c['sequence_length']!=self.length or c['sample_fps']!=self.fps or c['targets']+([] if self.exercise=='squats' else ['any_selected_error'])!=self.targets):raise ValueError('checkpoint_metadata_mismatch')
   if (np.asarray(c['mean']).shape!=(c['feature_count'],) or np.asarray(c['std']).shape!=(c['feature_count'],) or not np.isfinite(c['mean']).all() or not np.isfinite(c['std']).all() or np.any(np.asarray(c['std'])<=0)):raise ValueError('checkpoint_normalization_invalid')
   self.model=MODELS[c['architecture']](c['feature_count'],len(self.targets));self.model.load_state_dict(c['state_dict']);self.model.to(self.device).eval();self.quality=None
   if self.config['normal_enabled']:
    q=self.meta['quality'];qc=torch.load(self.root/q['path'],map_location='cpu',weights_only=True)
    if qc['kind']!=architecture or qc['d']!=c['feature_count'] or qc['n']!=1 or qc['s']!=759:raise ValueError('quality_metadata_mismatch')
    self.quality=Hybrid(qc['d'],qc['s'],qc['n'],qc['kind']);self.quality.load_state_dict(qc['state_dict']);self.quality.to(self.device).eval();self.qnorm=dict(np.load(self.root/q['normalization'],allow_pickle=False))
   self.loaded=True
  except Exception as e:self.load_error=type(e).__name__
 def unknown(self,reason,**extra):
  return dict(exercise=self.exercise,status='UNKNOWN',error_type=None,error_types=[],confidence=None,class_index=None,class_index_namespace=None,status_index=-1,model_name=f'{self.exercise}_{self.architecture}',feedback=None,feedbacks=[],reason=reason,model_outputs=None,confidence_kind='uncalibrated_sigmoid_not_accuracy',normal_certified=False,**extra)
 def preprocess(self,landmarks,aspect):
  p=np.asarray(landmarks,np.float32)
  if p.ndim!=3 or p.shape[1:]!=(33,4) or len(p)<8 or not np.isfinite(aspect) or aspect<=0:return None,0.
  if self.exercise!='squats':return prepare_landmarks(p,aspect,self.required)
  p=p.copy();valid=np.isfinite(p).all((1,2))&(np.nan_to_num(p[:,self.required,3],nan=0).min(1)>=.3);ratio=float(valid.mean())
  if valid.sum()<8 or ratio<.6:return None,ratio
  flat=p.reshape(len(p),-1)
  for j in range(flat.shape[1]):flat[:,j]=np.interp(np.arange(len(p)),np.flatnonzero(valid),flat[valid,j])
  return resample(features(p,aspect),self.length),ratio
 def predict_sequence(self,landmarks,aspect=1.,supported_variant=True,correlation_id=None):
  if not self.loaded:return self.unknown('model_unavailable',load_error=self.load_error)
  if not supported_variant:return self.unknown('unsupported_variant')
  try:
   x,ratio=self.preprocess(landmarks,float(aspect))
   if x is None:return self.unknown('insufficient_pose_or_invalid_shape',valid_frame_ratio=ratio)
   r=self.predict_features(x);r['valid_frame_ratio']=ratio
   if correlation_id is not None:r['correlation_id']=str(correlation_id)
   return r
  except Exception as e:return self.unknown('preprocessing_failed',error_kind=type(e).__name__)
 def predict_features(self,features_raw):
  if not self.loaded:return self.unknown('model_unavailable',load_error=self.load_error)
  try:
   x=np.asarray(features_raw,np.float32)
   if x.shape!=(self.length,self.meta['feature_count']) or not np.isfinite(x).all():return self.unknown('invalid_feature_shape_or_values')
   c=self.ck;z=(x-np.asarray(c['mean'],np.float32))/np.asarray(c['std'],np.float32)
   with torch.inference_mode():scores=torch.sigmoid(self.model(torch.tensor(z[None],device=self.device))).cpu().numpy()[0]
   if not np.isfinite(scores).all():return self.unknown('nonfinite_model_output')
   thresholds=np.asarray(self.meta['thresholds']);active=np.asarray(self.meta['active_targets'],bool);margin=float(self.config['decision_margin']);hits=active[:self.subtype_count]&(scores[:self.subtype_count]>=thresholds[:self.subtype_count]);strong=active[:self.subtype_count]&(scores[:self.subtype_count]>=np.minimum(1.,thresholds[:self.subtype_count]+margin));quality=None;normal_confident=False;error_confident=True
   if self.quality is not None:
    n=self.qnorm;s=summary(x[None]);a=torch.tensor(np.clip((x[None]-n['mean'])/n['std'],-8,8));b=torch.tensor(np.clip((s-n['summary_mean'])/n['summary_std'],-8,8))
    with torch.inference_mode():quality=float(torch.sigmoid(self.quality(a.to(self.device),b.to(self.device))).cpu().numpy().ravel()[0])
    if not math.isfinite(quality):return self.unknown('nonfinite_quality_output')
    t=self.meta['quality']['threshold'];normal_confident=quality<max(0.,t-margin);error_confident=quality>=min(1.,t+margin)
   output=self.unknown('insufficient_score_margin');output['model_outputs']=dict(error_types={name:float(scores[j]) for j,name in enumerate(self.targets)},error_type_thresholds={name:float(thresholds[j]) for j,name in enumerate(self.targets)},quality_error_score=quality,quality_error_threshold=self.meta.get('quality',{}).get('threshold'),normal_assessment_supported=bool(self.config['normal_enabled']),decision_policy_validated=bool(self.config['policy_validated']))
   if not self.config['error_feedback_enabled']:output['reason']='unverified_exercise_label_scope';return output
   if np.any(strong) and error_confident:
    indices=np.flatnonzero(strong);indices=sorted(indices,key=lambda j:float(scores[j]),reverse=True);names=[self.targets[j] for j in indices];feedbacks=[self.config['feedback'][name] for name in names];output.update(status='ERROR',error_type=names[0],error_types=names,confidence=float(scores[indices[0]]),class_index=int(indices[0]),class_indices=[int(j) for j in indices],class_index_namespace='error_type_head',status_index=1,feedback=feedbacks[0],feedbacks=feedbacks,reason=None)
   elif normal_confident and not np.any(hits):output.update(status='NORMAL',confidence=float(1-quality),class_index=0,class_index_namespace='binary_quality_class',status_index=0,reason=None,normal_scope='QEVD supported squat reference only; not whole-form certification')
   elif normal_confident and np.any(hits):output['reason']='quality_and_error_type_evidence_conflict'
   elif not self.config['normal_enabled']:output['reason']='normal_not_supported_and_no_confident_selected_error'
   return output
  except Exception as e:return self.unknown('inference_failed',error_kind=type(e).__name__)

class SequenceSession:
 """One instance per smartphone stream/user/exercise. Buffering, not repetition counting."""
 def __init__(self,classifier):self.classifier=classifier;self.frames=deque(maxlen=classifier.length);self.times=deque(maxlen=classifier.length);self.last_seen=None;self.last_sample=None;self.aspect=None;self.subject_token=None
 def reset(self):self.frames.clear();self.times.clear();self.last_seen=None;self.last_sample=None;self.aspect=None;self.subject_token=None
 def push(self,landmarks,timestamp,aspect,supported_variant=True,subject_token=None):
  c=self.classifier
  if not c.loaded:return c.unknown('model_unavailable')
  try:
   t=float(timestamp);a=float(aspect);p=np.asarray(landmarks,np.float32)
   if not math.isfinite(t) or not math.isfinite(a) or a<=0 or p.shape!=(33,4) or not np.isfinite(p).all() or p[c.required,3].min()<.3:
    self.reset();return c.unknown('missing_or_invalid_pose',updated=True,buffer_frames=0)
   if self.last_seen is not None and t<=self.last_seen:self.reset();return c.unknown('nonmonotonic_timestamp',updated=True,buffer_frames=0)
   reset_reason=None
   if self.last_seen is not None and t-self.last_seen>1.:reset_reason='stream_gap'
   if self.aspect is not None and abs(a-self.aspect)>.01:reset_reason='aspect_or_orientation_changed'
   if self.subject_token is not None and subject_token!=self.subject_token:reset_reason='subject_changed'
   if reset_reason:self.reset()
   self.last_seen=t;self.aspect=a;self.subject_token=subject_token
   if not supported_variant:self.reset();return c.unknown('unsupported_variant',updated=True,buffer_frames=0)
   if self.last_sample is not None and t-self.last_sample+1e-6<1/c.fps:return c.unknown('not_sampled_this_frame',updated=False,buffer_frames=len(self.frames))
   self.last_sample=t;self.frames.append(p.copy());self.times.append(t)
   if len(self.frames)<c.length:return c.unknown(reset_reason or 'warming_up',updated=True,buffer_frames=len(self.frames),required_frames=c.length)
   dt=np.diff(self.times)
   if dt.max()>.35 or not .12<=float(np.median(dt))<=.22:self.reset();return c.unknown('sampling_interval_mismatch',updated=True,buffer_frames=0)
   result=c.predict_sequence(np.asarray(self.frames),a);result.update(updated=True,buffer_frames=len(self.frames),required_frames=c.length);return result
  except Exception as e:self.reset();return c.unknown('stream_input_failed',updated=True,buffer_frames=0,error_kind=type(e).__name__)
