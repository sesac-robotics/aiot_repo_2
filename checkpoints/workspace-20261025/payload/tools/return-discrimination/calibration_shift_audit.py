"""Diagnostic control only; frozen ensemble and Test predictions are immutable."""
import json,sys
from pathlib import Path
import numpy as np,joblib
B=Path(__file__).resolve().parents[2];O=B/'output/return-discrimination-ensemble-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from models import PositiveDeficitClassifier
from train_scores import weights,metrics,predictions
from split_guard import assert_training_allowed
read=lambda p:json.loads(p.read_text());before=sha(O/'score-candidates.joblib');rows=read(O/'score-inputs.frozen.json');rs=[]
for r in rows:
 for j,(i,g) in enumerate(zip(r['items'],r['levels'])):
  if i['bank']=='standing_support_extension_v1' and i['eligible'] and g is not None:rs.append(dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],exercise=r['exercise'],split=r['split'],sources=r['sources'],x=i['feature'],target=g))
a=[r for r in rs if r['split']=='train'];b=[r for r in rs if r['split']=='validation'];assert_training_allowed(a);rep=read(O/'new-head-training.json');C=float(rep['selected'].split('_C')[-1]);w=weights(a);normal=np.array([r['target']==2 for r in a])
for mask in [normal,~normal]:
 if mask.any():w[mask]/=w[mask].sum()
w*=len(w)/w.sum();m=PositiveDeficitClassifier(C).fit([r['x'] for r in a],[r['target'] for r in a],sample_weight=w);m.normal_threshold=rep['normal_probability_cutoff'];p=predictions(m,b);fp=metrics(p);saved=rep['fixed_development_validation'];d=dict(diagnostic_control_not_a_selection=True,same_Train_labels_features_C_sample_weights_cutoff=True,only_change='single full-data refit versus frozen nine person-held components',cutoff=m.normal_threshold,C=C,full_refit_development=fp,frozen_ensemble_development=saved,full_refit_predictions=p,frozen_model_unchanged=before==sha(O/'score-candidates.joblib'),Test_not_used_for_control_fit=True);assert d['frozen_model_unchanged'];(O/'calibration-shift-control.json').write_text(json.dumps(d,indent=2));print('CONTROL_FULL',fp,'FROZEN_ENSEMBLE',saved)
