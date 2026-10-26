"""잔여 학습 후보의 확대판만 추가한다. 시험 예약 자료는 읽지 않는다."""
import pathlib
import audit
import external_review
from held_round3 import ROOT


def main() -> None:
    """미달 깊이와 누락 W 후보의 시간순 장면을 준비한다."""
    rows={r['id']:r for r in audit.read_json(ROOT/'screening_manifest.json')['records']}
    records=[]
    for key in ['qevd_00146521','qevd_00165807','qevd_00133899','qevd_00151214','qevd_00124682']:
        r=rows[key];a,b=0,r['frame_count']-1
        receipt=external_review.contact_sheet(pathlib.Path(r['path']),ROOT/'dense'/f'{key}.jpg',a/r['fps'],b/r['fps'],24)
        records.append(dict(r,frames=[a,b],preparation=receipt))
    audit.save_json(ROOT/'extra_preparation.json',dict(records=records,automatic_preparation_not_direct_review=True))


if __name__=='__main__':main()
