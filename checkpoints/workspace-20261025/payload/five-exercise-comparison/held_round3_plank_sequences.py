"""기존 승인 마스크를 보존해 지지 특징 진단용 관절 사본만 모은다."""
import pathlib

import cv2
import numpy as np

import audit
import three_items
from held_round3 import ROOT
from plank_partial_item_trial import load_rows


def main() -> None:
    """학습·개발 원본만 읽고 기존 항목 특징과 관절의 대응을 보존한다."""
    old=audit.BASE/'output/five-exercise-recovery-20261022'
    sequences={r['id']:r['seq'] for r in load_rows(True)}
    folder=old/'allfive_round1';sources={r['id']:r for r in audit.read_json(folder/'score_annotations.json')}
    for r in audit.read_json(folder/'extraction.json')['records']:
        if not r['id'].startswith('held_navy'):continue
        with np.load(r['extraction']['cache']) as data:seq=data['image_xy'].copy()
        cap=cv2.VideoCapture(sources[r['id']]['video_path']);seq=three_items.mask_image_border(seq,cap.get(3),cap.get(4));cap.release();sequences[r['id']]=seq
    folder=old/'held_round2';sources={r['id']:r for r in audit.read_json(folder/'dense_preparation.json')['records']};sources.update({r['id']:r for r in audit.read_json(folder/'held_plank_preparation.json')['records']});reviews={r['id']:r for r in audit.read_json(folder/'subject_tracking_review.json')['records']}
    for r in audit.read_json(folder/'extraction.json')['records']:
        key=r['id']
        if sources[key]['exercise']!='plank':continue
        with np.load(r['extraction']['cache']) as data:seq=data['image_xy'].copy();frames=data['frame_numbers']
        for a,b in reviews[key].get('unreliable_frame_intervals',[]):seq[(frames>=a)&(frames<=b)]=np.nan
        cap=cv2.VideoCapture(sources[key]['path']);seq=three_items.mask_image_border(seq,cap.get(3),cap.get(4));cap.release();sequences[key+'_held2_score']=seq
    for r in audit.read_json(ROOT/'new_feature_rows.json')['records']:
        if r['exercise']!='plank':continue
        key=r['id'].removesuffix('_held3_score')
        with np.load(ROOT/'observations'/key/'primary.npz') as data:seq=three_items.mask_image_border(data['image_xy'],float(data['width']),float(data['height']))
        sequences[r['id']]=seq
    (ROOT/'plank_sequence_cache').mkdir(exist_ok=True);records=[]
    for key,seq in sequences.items():
        path=ROOT/'plank_sequence_cache'/(key+'.npy');np.save(path,seq,allow_pickle=False);records.append(dict(id=key,coordinates=str(path)))
    audit.save_json(ROOT/'plank_sequences.json',dict(records=records,final_test_opened=False,automatic_feature_preparation_not_direct_review=True))


if __name__=='__main__':main()
