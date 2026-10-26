from pathlib import Path
import json,sys,datetime
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/full-review-v1';P=O/'shoulder-review-journal.json'
# Explicit human/agent visual observations only; no copying provider tags into item grades.
def add(id,lower,body,note,intervals=None,dense_needed=None):
 rows=json.loads(P.read_text()) if P.exists() else [];inv=next(r for r in json.loads((O/'inventory.frozen.json').read_text())['sources'] if r['id']==id);receipt=next(r for r in json.loads((O/'render-receipts.json').read_text()) if r['id']==id)
 row=dict(id=id,source_sha256=inv['source_sha256'],split=inv['split'],view=inv['view'],person=inv['person'],recording_group=inv['recording_group'],reviewer='AI_assisted_visual_draft_not_expert',review_method='full_duration_chronological_2fps_contact_sheets_plus_explicit_dense_followup_when_listed',review_pages=receipt['pages'],timeline_sample_count=len(receipt['review_times']),duration=receipt['duration'],lower_return_visual_draft=lower,frontal_trunk_visual_draft=body,evidence_note_ko=note,complete_cycle_intervals=intervals or [],dense_followup_windows=dense_needed or [],label_status='screened_pending_dense_cycle_check' if dense_needed else 'visual_draft',training_status='not_fitted',written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 rows=[r for r in rows if r['id']!=id]+[row];P.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':add(**json.loads(sys.argv[1]))
