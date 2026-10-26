"""Read-only fixed-model diagnostic; never fit or select thresholds on development."""
import datetime, hashlib, json, sys
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parents[2]
P = B / 'output/squat-fixed-reference-combined-v3-20261024'
O = B / 'output/squat-remaining-diagnostic-v4-20261024'
sys.path.insert(0, str(B / 'tools/error-enrichment'))
from common import sha

def read(p): return json.loads(p.read_text())
def save(name, value):
    p = O / name
    assert not p.exists(), p
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def sheet(path, frames, dest, title):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    cols = 8; tilew = 210; tileh = 260
    out = Image.new('RGB', (cols*tilew, ((len(frames)+cols-1)//cols)*tileh+30), 'white')
    draw = ImageDraw.Draw(out); draw.text((5, 5), title, fill='black')
    for j, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, a = cap.read()
        assert ok, (path, f)
        im = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)); im.thumbnail((tilew, tileh-28))
        x=j%cols*tilew; y=j//cols*tileh+30
        out.paste(im, (x+(tilew-im.width)//2, y+25)); draw.text((x+3,y+3), f'f{f}', fill='black')
    cap.release(); out.save(dest, quality=95)

def main():
    O.mkdir(exist_ok=True)
    save('protocol.frozen.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_model_sha256=sha(P/'new-head.joblib'), no_training=True, no_threshold_selection=True,
        development_previously_exposed=True, final_Test_opened=False,
        diagnostic='Compare fixed-model probabilities, coefficients and coordinate overlap. Perturbation/counterfactuals are diagnostic only, never real labels or measured accuracy.'))
    head=next(iter(joblib.load(P/'new-head.joblib')['heads'].values())); model=head['model']
    rows=read(P/'score-inputs.frozen.json'); samples=[]
    for r in rows:
        for it,g in zip(r['items'],r['levels']):
            if it['bank']=='standing_fixed_initial_context_v3' and it['eligible'] and r['split']=='validation':
                samples.append(dict(id=r['id'],person=r['person'],target=g,x=it['feature']))
    names=['hip_height_mean','hip_height_max','shoulder_height_mean','shoulder_height_max',
           'knee_deficit_mean','knee_deficit_max','hip_angle_deficit_mean','hip_angle_deficit_max',
           'trunk_deficit_mean','trunk_deficit_max']
    for s in samples:
        s['normal_probability']=float(model.normal_probability([s['x']])[0])
        s['prediction']=float(model.predict([s['x']])[0])
        s['normal_margin']=s['normal_probability']-model.normal_threshold
    missed=next(s for s in samples if s['id'].endswith('diag_w2_5_7_r09'))
    normals=[s for s in samples if s['target']==2]
    scale=np.median([m.scaler.scale_ for m in model.members],axis=0)
    distances=sorted((float(np.linalg.norm((np.array(s['x'])-missed['x'])/scale)),s['id']) for s in normals)
    nearest=next(s for s in normals if s['id']==distances[0][1])
    weights=np.array([m.coef_/m.scaler.scale_ for m in model.members])
    pairs=[]
    for s in normals:
        # Additive ERROR-logit difference is exact within each component;
        # averaging nonlinear probabilities is not an additive attribution.
        contributions=weights*(np.array(missed['x'])-s['x'])
        pairs.append(dict(normal_id=s['id'],mean_error_logit_difference_by_feature=contributions.mean(0).tolist(),
                          normal_probability=s['normal_probability'],normal_x=s['x']))
    # All cutoffs are enumerated only to document overlap, never saved as candidate thresholds.
    changes=[]
    for s in sorted(samples,key=lambda a:a['normal_probability']):
        if s['target']==1:changes.append(dict(error_id=s['id'],normal_probability=s['normal_probability'],
            normals_at_or_below_error_probability=[n['id'] for n in normals if n['normal_probability']<=s['normal_probability']]))
    save('fixed-model-diagnostic.json', dict(feature_names=names, cutoff=model.normal_threshold,
        samples=samples, missed=missed, nearest_normal_standardized_distance=distances[0][0],nearest_normal=nearest,
        mean_raw_coordinate_error_coefficients=weights.mean(0).tolist(),pairwise_component_error_logit_differences=pairs,
        probability_order_overlap=changes, no_refit=True, no_threshold_or_labels_changed=True,
        notes='Coefficient magnitudes use different feature scales. Differences in component logits are exact; mean logits do not equal ensemble probability.'))
    labels={r['id']:r for r in read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels']}
    d=labels['squat_diag_w2_5_7_r09']
    frames=list(range(0,7,3))+list(range(384,409))
    sheet(d['path'],frames,O/'r09-upper-every-frame.jpg','Previously exposed development r09; initial f0/3/6 then complete upper turn f384..408')
    sheet(d['path'],list(range(237,262)),O/'r05-normal-upper-every-frame.jpg','Previously exposed normal r05; upper turn f237..261')
    save('visual-review-intake.frozen.json',dict(source_path=d['path'],source_sha256=sha(d['path']),
        reference_frames=[0,3,6],r09_frames=list(range(384,409)),r05_frames=list(range(237,262)),
        original_grades_unchanged=True,source_role='validation',not_for_training=True))
    print(json.dumps(dict(missed=missed,nearest_normal=nearest,ordering=changes),indent=2),flush=True)

if __name__=='__main__': main()
