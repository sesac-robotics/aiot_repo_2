"""고정 직접 초안의 학습 구간만 배포 Heavy GPU로 추출한다."""
import json
import os
import pathlib
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from held_round3 import ROOT
from resources import check_resources


def main() -> None:
    """구간별 수치 관절·기존 모델 진단·시간순 관절 검토판을 저장한다."""
    sources={r['id']:r for r in audit.read_json(ROOT/'screening_manifest.json')['records']}
    decisions=audit.read_json(ROOT/'manual_decisions.json')['records'];work=audit.BASE/'team-posture-scoring/Server/AIService'
    (ROOT/'requests').mkdir(exist_ok=True);(ROOT/'observations').mkdir(exist_ok=True);(ROOT/'joints').mkdir(exist_ok=True);records=[]
    for d in decisions:
        check_resources();r=sources[d['id']];options=dict(exercise=r['exercise'],view=d['view'],start_frame=d['frames'][0],end_frame=d['frames'][1],complete_attempt=d['attempt_complete'],permitted_items=[g is not None for g in d['levels']],phase_windows=d['phase_windows'],projection_reviewed=d['projection_reviewed'])
        request=dict(video_path=r['path'],labels=r['source_labels'],labels_descriptive=[],split='train',assessment_input=options)
        path=ROOT/'requests'/(r['id']+'.json');audit.save_json(path,request);out=ROOT/'observations'/r['id']
        if not out.exists():
            env=dict(os.environ)
            for key in ('PYTHONPATH','LD_PRELOAD','__EGL_VENDOR_LIBRARY_FILENAMES'):env.pop(key,None)
            command=[str(work/'tools/posture_score/run_nvidia.sh'),str(audit.BASE/'.posture-score-verify/bin/python'),'--request',str(path),'--output',str(out),'--heavy-model',str(audit.BASE/'mediapipe-pilot/models/heavy.task')]
            p=subprocess.run(command,cwd=work,env=env,text=True,capture_output=True,timeout=180)
            (ROOT/'observations'/(r['id']+'.log')).write_text(p.stdout+p.stderr);p.check_returncode()
        evidence=ROOT/'joints'/(r['id']+'.jpg')
        if evidence.exists():
            records.append(dict(id=r['id'],source_sha256=r['sha256'],landmarks=str(out/'primary.npz'),joint_review_sheet=str(evidence)))
            audit.save_json(ROOT/'extraction.json',dict(records=records,completed=False))
            continue
        with np.load(out/'primary.npz') as data:seq=data['image_xy'];frames=data['frame_numbers']
        indices=np.unique(np.rint(np.linspace(0,len(seq)-1,12)).astype(int));canvas=Image.new('RGB',(1200,235*2),'#14212a');draw=ImageDraw.Draw(canvas);cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        edges=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]
        for k,i in enumerate(indices):
            cap.set(1,int(frames[i]));ok,frame=cap.read()
            if not ok:raise ValueError('Joint review decode failed')
            for a,b in edges:
                if np.isfinite(seq[i,[a,b]]).all():cv2.line(frame,tuple(seq[i,a].round().astype(int)),tuple(seq[i,b].round().astype(int)),(40,240,60),2)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((200,198));x,y=(k%6)*200,(k//6)*235;canvas.paste(im,(x+(200-im.width)//2,y+22));draw.text((x+2,y+220),f'f{frames[i]}',fill='white')
        cap.release();draw.text((3,2),r['id']+' | GPU Heavy; tracking review only',fill='white');evidence=ROOT/'joints'/(r['id']+'.jpg');canvas.save(evidence,quality=95)
        records.append(dict(id=r['id'],source_sha256=r['sha256'],landmarks=str(out/'primary.npz'),joint_review_sheet=str(evidence)))
        audit.save_json(ROOT/'extraction.json',dict(records=records,completed=False))
    audit.save_json(ROOT/'extraction.json',dict(records=records,completed=True,direct_tracking_review=False))


if __name__=='__main__':main()
