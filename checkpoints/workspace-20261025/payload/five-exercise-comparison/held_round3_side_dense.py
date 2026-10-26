"""보유 측면 후보의 선택 구간을 확대 준비한다. 정답은 생성하지 않는다."""
import pathlib

import audit
import external_review
from held_round3 import ROOT


def main() -> None:
    """푸시업 반복과 플랭크 지지 대비·짧은 유지 구간을 표본화한다."""
    previous={r['id']:r for r in audit.read_json(ROOT/'side_preparation.json')['records']}
    windows={'subject_007_push_up_bad_side':(184,215),'subject_009_push_up_bad_side':(187,217),'subject_014_push_up_bad_side':(186,205),'subject_010_push_up_good_side':(180,206)}
    plan=[]
    for key,(a,b) in windows.items():plan.append(dict(previous[key],frames=[a,b],view='side'))
    lineage=audit.read_json(audit.OUT/'sources/dvids_640237_side_lineage.json')
    plan.append(dict(id='dvids_gates640237_floor_contrast_held3',exercise='plank',person=lineage['person'],path=lineage['derived_video_path'],sha256=lineage['derived_sha256'],original_source_sha256=lineage['original_sha256'],recording_group=lineage['performance_group'],split='train',source_labels=[],fps=29.97,frames=[270,362],view='side'))
    r=previous['dvids_745041_held3_screen'];plan.append(dict(r,id='dvids745041_partial_hold_held3',person='dvids745041_female_performer',frames=[660,720],view='side'))
    records=[]
    for r in plan:
        r['sha256']=r.get('sha256') or r['preparation']['video_sha256'];a,b=r['frames'];receipt=external_review.contact_sheet(pathlib.Path(r['path']),ROOT/'dense'/(r['id']+'.jpg'),a/r['fps'],b/r['fps'],24)
        records.append(dict(r,preparation=receipt))
        audit.save_json(ROOT/'side_dense_preparation.json',dict(records=records,completed=False))
    audit.save_json(ROOT/'side_dense_preparation.json',dict(records=records,completed=True,automatic_preparation_not_direct_review=True))


if __name__=='__main__':main()
