"""정면 후보의 관측 한계를 보완할 보유 측면 원본만 추가 검토한다."""
import collections
import hashlib
import pathlib

import cv2

import audit
import external_review
from held_round3 import ROOT


def main() -> None:
    """기존 학습 역할의 측면 푸시업과 보유 공공 영상 표본판을 준비한다."""
    pool=audit.read_json(audit.BASE/'output/overnight-20261020/score_candidates.json');old=audit.read_json(audit.BASE/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json')
    used={s:r['split'] for r in old for s in r['sources']};rows=[]
    for key in ['subject_007_push_up_bad_side','subject_009_push_up_bad_side','subject_014_push_up_bad_side','subject_010_push_up_good_side']:
        r=next(r for r in pool if r['id']==key)
        if r['split']!='train':raise ValueError('Not a training source')
        rows.append(dict(r,review_start_seconds=12,review_end_seconds=20))
    source=audit.OUT/'sources'
    for code in ['640237','1000181','873358','745041']:
        path=source/f'dvids_{code}.mp4'
        if not path.exists():continue
        sha=audit.digest(path)
        if used.get(sha) in ('validation','test'):continue
        rows.append(dict(id='dvids_'+code+'_held3_screen',path=str(path),exercise='plank',person='held_dvids_'+code,split='train',source_labels=[],sha256=sha,review_start_seconds=0,review_end_seconds=None))
    out=[];(ROOT/'side_review').mkdir(exist_ok=True)
    for r in rows:
        cap=cv2.VideoCapture(r['path']);fps=cap.get(5);count=int(cap.get(7));cap.release()
        end=min(r['review_end_seconds'],(count-1)/fps) if r['review_end_seconds'] is not None else (count-1)/fps
        receipt=external_review.contact_sheet(pathlib.Path(r['path']),ROOT/'side_review'/(r['id']+'.jpg'),r['review_start_seconds'],end,24)
        out.append(dict(r,preparation=receipt,fps=fps,frame_count=count))
        audit.save_json(ROOT/'side_preparation.json',dict(records=out,completed=False))
    audit.save_json(ROOT/'side_preparation.json',dict(records=out,completed=True,automatic_preparation_not_direct_review=True))


if __name__=='__main__':main()
