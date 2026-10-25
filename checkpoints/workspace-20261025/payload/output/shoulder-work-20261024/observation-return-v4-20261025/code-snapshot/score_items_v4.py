"""Isolated scoring adapter. All shipped model artifacts remain unsupported here."""
import numpy as np
from observation_features_v4 import extract_return,extract_body,RETURN_SCHEMA,BODY_SCHEMA
WEIGHTS={'lower_return':25,'frontal_trunk_hold':20}
SCHEMAS={'lower_return':RETURN_SCHEMA,'frontal_trunk_hold':BODY_SCHEMA}
TARGETS={'lower_return':'observed_ready_relative_lower_return',
         'frontal_trunk_hold':'observed_frontal_window_trunk_stability_only'}

def score(samples,models=None,width=640,height=480,view='front',pose_context=None,
          trunk_window=None,exercise_confirmed=False):
    lower=extract_return(samples,width,height,view)
    if trunk_window is None and lower['status']=='features_ready':
        trunk_window=[lower['reference_ticks'][0],lower['return_ticks'][-1]]
        exercise_confirmed=True
    body=extract_body(samples,trunk_window,exercise_confirmed,width,height,view)
    features={'lower_return':lower,'frontal_trunk_hold':body};rows=[]
    for item,weight in WEIGHTS.items():
        f=features[item];artifact=(models or {}).get(item)
        row=dict(id=item,weight=weight,grade=None,points=None,reasons=[])
        if f['status']!='features_ready':row['reasons'].append(f['reason'])
        elif artifact is None:row['reasons'].append('no_validated_multi_grade_model')
        else:
            counts=artifact.get('training_grade_counts',{})
            support=sorted(int(g) for g,n in counts.items() if n>0)
            if 2 not in support or not any(g in (0,1) for g in support):row['reasons'].append('normal_only_or_missing_normal_training')
            if any(g not in (0,1,2) for g in support):row['reasons'].append('invalid_grade_support')
            if artifact.get('feature_schema')!=SCHEMAS[item]:row['reasons'].append('feature_schema_mismatch')
            if artifact.get('criterion')!=TARGETS[item]:row['reasons'].append('criterion_mismatch')
            if artifact.get('view')!=view or view!='front':row['reasons'].append('view_mismatch')
            if pose_context not in ('standing','seated') or artifact.get('pose_context')!=pose_context:row['reasons'].append('pose_context_mismatch')
            if not artifact.get('development_gate',{}).get('passed'):row['reasons'].append('development_gate_not_passed')
            if not artifact.get('training_ids') or not artifact.get('training_source_hashes'):row['reasons'].append('missing_training_provenance')
            if artifact.get('score_prediction_enabled') is not True:row['reasons'].append('score_prediction_not_enabled')
            if not row['reasons']:
                key='return_features' if item=='lower_return' else 'features'
                value=float(artifact['model'].predict(np.asarray([f[key]],float))[0])
                if not np.isfinite(value) or value<min(support) or value>max(support):row['reasons'].append('prediction_outside_observed_grade_support')
                else:row.update(grade=value,points=value/2*weight)
        rows.append(row)
    return dict(features=features,items=rows,all_items_supported=all(r['points'] is not None for r in rows),
                production_replacement=False,independent_final_test=False)
