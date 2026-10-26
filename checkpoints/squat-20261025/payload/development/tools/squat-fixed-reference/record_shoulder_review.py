"""Record manually inspected evidence only; never infer labels or fit a model."""
import datetime
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[2]
R = B / 'output/shoulder-common-support-audit-20261024/full-review-v1'

def record(subject, quality, view, lower, trunk, protocol, events, note, extra_pages=()):
    sid = f'subject_{subject:03d}_shoulder_{quality}_{view}'
    src = next(x for x in json.loads((R/'decoded-manifest.frozen.json').read_text())['sources'] if x['id'] == sid)
    panel = next(x for x in json.loads((R/'dense-panel-manifest.frozen.json').read_text())['records'] if x['id'] == sid)
    journal = R/'dense-visual-labels.journal.json'
    rows = json.loads(journal.read_text())
    assert sid not in {x['id'] for x in rows}, sid
    pages = [R/x for x in panel['sheets']] + [R/x for x in extra_pages]
    screen = R/(sid+'-screen.jpg')
    def sha(p):
        with p.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    rows.append(dict(id=sid, source_sha256=src['source_sha256'], person=src['person'],
        recording_group=src['recording_group'], split=src['split'], view=view,
        reviewer='AI_assisted_visual_not_expert_adjudicated',
        method='full_duration_1Hz_screen_and_selected_original_5Hz_windows',
        reviewed_pages=[str(p) for p in pages], screen=str(screen),
        evidence_sha256={p.name:sha(p) for p in pages+[screen]},
        reviewed_windows=panel['intervals'], note_ko=note,
        lower_return_grade=lower, frontal_trunk_grade=trunk,
        sagittal_trunk_grade=None, movement_protocol=protocol, observed_events=events,
        scope='selected_observed_returns_and_trunk_only_not_whole_clip',
        label_inferred_from_provider=False, fitted=False,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    tmp = journal.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(journal)

if __name__ == '__main__':
    raise SystemExit('Import record and supply actual manual review decisions.')
