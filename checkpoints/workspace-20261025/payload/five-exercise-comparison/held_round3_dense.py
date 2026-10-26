"""직접 1차 검토에서 선택한 학습 후보만 시간순 확대 준비한다."""
import json
import pathlib

import audit
import external_review
from held_round3 import ROOT


def main() -> None:
    """점수를 생성하지 않고 선택 구간의 24장 검토판만 만든다."""
    selections={
      'qevd_00281594':(0,145),'qevd_00199942':(0,64),'qevd_00182807':(0,84),
      'qevd_00144112':(0,66),'qevd_00265669':(0,85),'qevd_00264509':(0,154),
      'qevd_00257470':(0,147),
      'qevd_00258184':(0,177),'qevd_00079260':(0,177),'qevd_00085675':(0,177),'qevd_00229683':(0,175),
      'subject_023_shoulder_bad_front':(270,459),'subject_016_shoulder_bad_front':(270,465),
      'subject_001_shoulder_bad_front':(270,464),'subject_005_shoulder_good_front':(270,460)}
    rows={r['id']:r for r in audit.read_json(ROOT/'screening_manifest.json')['records']}
    (ROOT/'dense').mkdir(exist_ok=True);records=[]
    for key,(a,b) in selections.items():
        r=rows[key];receipt=external_review.contact_sheet(pathlib.Path(r['path']),ROOT/'dense'/f'{key}.jpg',a/r['fps'],b/r['fps'],24)
        records.append(dict(r,frames=[a,b],preparation=receipt))
        audit.save_json(ROOT/'dense_preparation.json',dict(records=records,completed=False,automatic_preparation_not_direct_review=True))
    audit.save_json(ROOT/'dense_preparation.json',dict(records=records,completed=True,automatic_preparation_not_direct_review=True))


if __name__=='__main__':main()
