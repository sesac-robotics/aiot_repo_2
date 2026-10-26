"""No candidate passes on missing or normal-only development evidence."""
import numpy as np

def counts(ps):
 return dict(n=len(ps),normal_n=sum(p['target']==2 for p in ps),error_n=sum(p['target']<2 for p in ps),normal_groups=len({p['group'] for p in ps if p['target']==2}),error_groups=len({p['group'] for p in ps if p['target']<2}))
def assess(cv,validation,baseline,old=None):
 reasons=[];c=counts(validation)
 if min(c['normal_n'],c['error_n'])<3:reasons.append('development_requires_at_least_three_normal_and_error_segments')
 if min(c['normal_groups'],c['error_groups'])<2:reasons.append('development_requires_at_least_two_normal_and_error_person_source_groups')
 def metric(ps):
  err=[p for p in ps if p['target']<2];normal=[p for p in ps if p['target']==2]
  groups=sorted({p['group'] for p in ps});grades=sorted({p['target'] for p in ps});balanced=np.mean([np.mean([np.mean([abs(p['target']-p['prediction']) for p in ps if p['group']==g and p['target']==y]) for g in {p['group'] for p in ps if p['target']==y}]) for y in grades]) if ps else None
  return dict(error_recall=np.mean([p['prediction']<1.5 for p in err]) if err else None,normal_false_alarm=np.mean([p['prediction']<1.5 for p in normal]) if normal else None,MAE=np.mean([abs(p['target']-p['prediction']) for p in ps]) if ps else None,balanced_group_MAE=balanced)
 for name,ps in [('training_group_cv',cv),('fixed_development',validation)]:
  m=metric(ps)
  if m['error_recall'] is None or m['error_recall']<.65:reasons.append(name+'_error_recall_below_65_percent_or_unknown')
  if m['normal_false_alarm'] is None or m['normal_false_alarm']>.15:reasons.append(name+'_false_alarm_above_15_percent_or_unknown')
 cvm=metric(cv);vm=metric(validation)
 if cvm['balanced_group_MAE'] is None or cvm['balanced_group_MAE']>=baseline['balanced_group_MAE']:reasons.append('does_not_beat_training_constant_baseline')
 if old and vm['MAE'] is not None:
  if vm['MAE']>old['MAE']:reasons.append('development_MAE_worse_than_saved_model')
  if old['error_recall'] is not None and (vm['error_recall'] is None or vm['error_recall']<old['error_recall']):reasons.append('development_error_recall_worse_than_saved_model')
  if old['normal_false_alarm'] is not None and (vm['normal_false_alarm'] is None or vm['normal_false_alarm']>old['normal_false_alarm']):reasons.append('development_false_alarm_worse_than_saved_model')
 return dict(passed=not reasons,reasons=reasons,validation_support=c,independent_final_test_required=True,production_replacement=False)
