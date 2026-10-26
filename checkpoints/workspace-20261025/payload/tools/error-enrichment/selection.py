"""Select on TRAINING group CV only, matching the actual fault-detection task."""
def choose(candidates):
 def decision(c):
  m=c['metrics'];recall=m['error_recall'];false=m['normal_false_alarm']
  if recall is None or false is None:return (2,2,m['balanced_group_MAE'])
  passes=recall>=.65 and false<=.15
  return (0 if passes else 1,(1-recall+false)/2,m['balanced_group_MAE'])
 passing=[c for c in candidates if decision(c)[0]==0]
 return min(passing,key=lambda c:c['metrics']['balanced_group_MAE']) if passing else min(candidates,key=decision)
