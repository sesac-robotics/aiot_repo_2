from pathlib import Path
import json,datetime
O=Path('/home/jayden/SeSACPAI-Mid/output/shoulder-work-20261024/full-review-v1')
def add(id,note,lower=None,body=None,events=None,protocol='shoulder_press'):
 p=O/'dense-visual-labels.json'; rows=json.loads(p.read_text()) if p.exists() else []
 source=next(r for r in json.loads((O/'shoulder-review-journal.json').read_text()) if r['id']==id)
 receipt=next(r for r in json.loads((O/'dense-receipts.json').read_text()) if r['id']==id)
 r=dict(id=id,source_sha256=source['source_sha256'],person=source['person'],recording_group=source['recording_group'],split=source['split'],view=source['view'],reviewer='AI_assisted_visual_not_expert_adjudicated',method='full_duration_2fps_screen_and_selected_original_5Hz_windows',reviewed_pages=receipt['pages'],reviewed_windows=source['dense_followup_windows'],note_ko=note,lower_return_grade=lower,frontal_trunk_grade=body,movement_protocol=protocol,observed_events=events or [],scope='selected_observed_returns_only_not_whole_clip',label_inferred_from_provider=False,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 rows=[x for x in rows if x['id']!=id]+[r];p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
