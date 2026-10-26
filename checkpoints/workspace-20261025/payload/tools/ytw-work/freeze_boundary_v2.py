import json,hashlib
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/boundary-v2';inv={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
entry={'V37':[[[3.5,3.7],[4.8,4.9],[5.9,6.0]],[[7.9,8.1],[9.3,9.4],[10.6,10.8]],[[12.7,12.9],[14.1,14.3],[15.2,15.4]]],'V39':[[[4.3,4.5],[6.3,6.5],[7.7,7.8]],[[9.3,9.5],[10.9,11.0],[12,12.1]],[[13.3,13.5],[15.3,15.4],[16.4,16.6]]],'V41':[[[3.5,3.7],[5.1,5.3],[6,6.2]],[[7.8,8],[8.8,9],[9.7,9.9]]],'V43':[[[5.7,5.9],[7.2,7.4],[8.2,8.4]],[[10,10.2],[11.2,11.4],[12.2,12.4]]]}
positive=[]
for sid,cycles in entry.items():
 for i,windows in enumerate(cycles,1):
  positive.append(dict(id=f'{sid}_review_cycle{i}',source_id=sid,source_sha256=inv[sid]['sha256'],phase_entry_seconds=dict(zip('YTW',windows)),phase_entry_original_frames={p:[round(t*float(Fraction(inv[sid]['fps']))) for t in w] for p,w in zip('YTW',windows)},confirmation_seconds=[round(t+.4,1) for t in windows[2]],quality_grade=None,reviewed_completed_order=True,review_kind='pixel_reviewed_10Hz_entry_uncertainty_brackets_not_expert_truth'))
negative=[]
def neg(sid,a,b,kind,detail):
 negative.append(dict(id=f'{sid}_negative_{len(negative)+1:02d}',source_id=sid,source_sha256=inv[sid]['sha256'],interval_seconds=[a,b],kind=kind,evidence=detail,complete_order_observed=False,quality_grade=None,natural_episode=True))
for a,b in [[0,2.7],[2.8,6.6],[6.7,10.2],[10.3,14.6],[14.7,26.2]]:neg('V38',a,b,'T_hold_omitted','Y raised hold changes directly into bent W-like hold; no bilateral extended horizontal T dwell in reviewed transition')
for a,b in [[5.3,10.6],[10.7,14.5],[19.3,25.0]]:neg('V40',a,b,'W_hold_omitted','Y/T changes into arms-down pause; no bilateral bent raised W hold in reviewed transition; pause stays in same episode')
for a,b in [[0,3.4],[3.5,8.6]]:neg('V44',a,b,'T_hold_omitted','Y to bent raised W-like posture, no sustained bilateral extended T')
for a,b in [[8.7,11.4],[11.5,14.5]]:neg('V44',a,b,'W_hold_omitted','Y/T to arms down, observed pause belongs to this episode')
neg('V43',26.5,28.75,'arms_down_wait','Single contiguous natural wait; kept once, not sliced to increase denominator')
waits=[dict(source_id=sid,interval_seconds=[a,b],count_as_extra_negative=False,reason='nested_in_one_noncomplete_episode') for sid,a,b in [('V40',9.3,10.5),('V40',13.2,14.5),('V40',23.7,25.0),('V44',10.9,11.4),('V44',14.0,14.5)]]+[dict(source_id='V43',interval_seconds=[26.5,28.75],count_as_extra_negative=True)]
amb=[dict(source_id='V40',interval_seconds=w,reason='asymmetric arm phase may be malformed W attempt; quality vs absent phase not safely separated; exclude from binary negative denominator',quality_grade=None) for w in [[0,5.2],[14.6,19.2],[25.1,31.0]]]+[dict(source_id='V44',interval_seconds=[14.6,25.2],reason='long Y/asymmetric transition/brief T-like ending; completion semantics uncertain; no forced negative or grade0')]
out=dict(schema='ytw-video-boundary-negative-reference-v2-draft',created_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(O/'plan.frozen.json'),reviewer='AI_assisted_direct_pixels_not_expert_or_blind',prior_exposure='full_model_and_previous_LOPO_counts_known; new_negative_predictions_not_read_before_freeze',frozen_before_v2_replay=True,source_roles='existing_Train_unchanged',positive_attempts=positive,natural_negative_episodes=negative,wait_spans=waits,ambiguous_excluded=amb,phase_boundary_uncertainty_max_seconds=.2,terminal_meaning='W_entry_plus_0.4s_confirmation_not_arms_down_or_full_quality',no_severity_labels_added=True,source_video_sha256={s:inv[s]['sha256'] for s in sorted({p['source_id'] for p in positive+negative})},review_sheets_sha256={p.name:sha(p) for p in sorted(O.glob('*.jpg'))})
p=O/'references-v2.frozen.json'
if p.exists():raise RuntimeError('Frozen references already exist; use audited new version')
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(len(positive),len(negative),len(waits),len(amb))
