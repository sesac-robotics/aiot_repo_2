import argparse, csv, hashlib, importlib.metadata, json, math, os, platform, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import psutil
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
# 내장 SSD에 복사한 선택 자료를 사용한다. T9 연결 여부와 무관하다.
DATA = ROOT.parent / 'data'
JOINTS = {'left_elbow':(11,13,15),'right_elbow':(12,14,16),'left_knee':(23,25,27),'right_knee':(24,26,28),'left_hip':(11,23,25),'right_hip':(12,24,26),'left_shoulder':(13,11,23),'right_shoulder':(14,12,24)}
EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def angle(a,b,c):
    u,v=np.asarray(a)-np.asarray(b),np.asarray(c)-np.asarray(b)
    norm=np.linalg.norm(u)*np.linalg.norm(v)
    if norm<1e-12:return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/norm,-1,1))))

def prepare():
    assert abs(angle([1,0],[0,0],[0,1])-90)<1e-8
    assert abs(angle([-1,0],[0,0],[1,0])-180)<1e-8
    assert angle([0,0],[0,0],[1,0]) is None
    samples=[];inventory={}
    r=DATA/'rehab26-squat'
    rows=list(csv.DictReader((r/'manifests/videos.csv').open()))
    row=next(x for x in rows if x['exercise']=='ex6' and x['fold_0']=='train' and x['camera']=='1')
    labels=list(csv.DictReader((r/'raw/annotations/Segmentation.csv').open(),delimiter=';'))
    labels=[x for x in labels if x['video_id']==row['recording_id']]
    samples.append({'dataset':'REHAB26','video':str(r/row['video_path']),'split':'train (proposed fold_0; not official)','video_id':row['video_id'],'labels':labels,'label_kind':'repetition frame ranges, correctness_score, error_description'})
    inventory['REHAB26']={'videos':len(rows),'exercise_counts':dict(Counter(x['exercise'] for x in rows)),'split_counts_fold_0':dict(Counter(x['fold_0'] for x in rows)),'split_policy':json.loads((r/'manifests/split_policy.json').read_text()),'exercise_names':{'ex1':'Arm abduction','ex6':'Squat','ex7':'Shoulder extension'}}
    r=DATA/'qevd-selected';counts=Counter();total=0;chosen=None
    with (r/'manifests/short_clips.csv').open() as f:
        for x in csv.DictReader(f):
            total+=1;counts[x['split']]+=1
            if chosen is None and x['split']=='train' and any(label.split(' - ')[0].lower() == 'squats' for label in json.loads(x['labels'])) and (r/x['video_path']).is_file():chosen=x
    assert chosen
    samples.append({'dataset':'QEVD','video':str(r/chosen['video_path']),'split':'train (manifest original split)','video_id':chosen['video_id'],'labels':json.loads(chosen['labels']),'worker_id':chosen['worker_id'],'label_kind':'clip-level labels; not frame-level joint ground truth'})
    inventory['QEVD']={'short_clip_manifest_rows':total,'official_split_counts':dict(counts),'split_policy':json.loads((r/'manifests/split_policy.json').read_text())}
    r=DATA/'pose-comparison-selected'
    samples.append({'dataset':'Pose-Estimation-Model-Comparison','video':str(r/'raw/videos/squat.mp4'),'split':'unassigned; preview only, not train','video_id':'squat','labels':[],'label_kind':'provided model predictions are not verified human ground truth'})
    inventory['Pose-Estimation-Model-Comparison']={'videos':[p.name for p in sorted((r/'raw/videos').glob('*.mp4'))],'split':'unassigned'}
    for s in samples:
        p=Path(s['video']);cap=cv2.VideoCapture(str(p));ok,frame=cap.read()
        assert ok,s['video']
        s['video_metadata']={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'fps':cap.get(cv2.CAP_PROP_FPS),'frame_count':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        cap.release()
    save(ROOT/'configs/samples.json',samples);save(ROOT/'outputs/dataset-inventory.json',inventory)
    sources=[]
    for model in ['lite','full','heavy']:
        url=f'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model}/float16/1/pose_landmarker_{model}.task'
        p=ROOT/'models'/f'{model}.task';p.parent.mkdir(exist_ok=True)
        if not p.exists():urllib.request.urlretrieve(url,p)
        sources.append({'model':model,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save(ROOT/'models/sources.json',sources)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    save(ROOT/'outputs/environment.json',{'date':time.strftime('%Y-%m-%d %H:%M:%S %z'),'platform':platform.platform(),'python':sys.version,'cpu':subprocess.check_output(['lscpu'],text=True),'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'ram_bytes':psutil.virtual_memory().total,'available_ram_before':psutil.virtual_memory().available,'swap_before':dict(psutil.swap_memory()._asdict()),'gpu':gpu,'packages':{p:importlib.metadata.version(p) for p in ['mediapipe','opencv-contrib-python','numpy','psutil']},'delegate':'CPU explicitly selected; NVIDIA readings are whole-device background activity, not MediaPipe GPU acceleration'})
    print('Prepared 3 samples and Lite/Full/Heavy models',flush=True)

def run(model,delegate='cpu',max_frames=180,run_name=None,samples_file=None):
    folder=ROOT/'outputs';folder=folder/run_name if run_name else folder;folder=folder/delegate/model;folder.mkdir(parents=True,exist_ok=True)
    proc=psutil.Process();stop=threading.Event();resources=[]
    def monitor():
        proc.cpu_percent();psutil.cpu_percent()
        while not stop.wait(.5):
            g=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            resources.append({'time':time.time(),'process_cpu_pct_one_core':proc.cpu_percent(),'process_rss_mb':proc.memory_info().rss/2**20,'system_cpu_pct':psutil.cpu_percent(),'system_available_mb':psutil.virtual_memory().available/2**20,'swap_used_mb':psutil.swap_memory().used/2**20,'whole_gpu_util_memory':g.stdout.strip()})
    thread=threading.Thread(target=monitor,daemon=True);thread.start();summaries=[]
    try:
        for sample in json.loads((Path(samples_file) if samples_file else ROOT/'configs/samples.json').read_text()):
            name=sample['dataset'];out=folder/name;out.mkdir(exist_ok=True)
            opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models'/f'{model}.task'),delegate=mp.tasks.BaseOptions.Delegate.GPU if delegate=='gpu' else mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
            t=time.perf_counter();detector=mp.tasks.vision.PoseLandmarker.create_from_options(opts);init_ms=(time.perf_counter()-t)*1000
            cap=cv2.VideoCapture(sample['video']);fps=sample['video_metadata']['fps'];writer=None;rows=[];lat=[];detected=0;valid_angles=Counter();pipeline=[];draw_write=[]
            cpu0=proc.cpu_times();start=time.perf_counter()
            fields=['frame','timestamp_ms','detected','inference_ms','pipeline_ms']+[f'{j}_{dim}_deg' for j in JOINTS for dim in ['2d','3d']]
            with (out/'landmarks.jsonl').open('w') as points:
                for frame_id in range(min(max_frames,sample['video_metadata']['frame_count'])):
                    tick=time.perf_counter();ok,frame=cap.read()
                    if not ok:raise RuntimeError(f'Decode failure at frame {frame_id}: {sample["video"]}')
                    h,w=frame.shape[:2];scale=min(1,640/max(h,w));frame=cv2.resize(frame,(int(w*scale)//2*2,int(h*scale)//2*2));h,w=frame.shape[:2]
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb));ts=round(frame_id*1000/fps)
                    t=time.perf_counter();result=detector.detect_for_video(image,ts);ms=(time.perf_counter()-t)*1000
                    row={'frame':frame_id,'timestamp_ms':ts,'detected':bool(result.pose_landmarks),'inference_ms':ms};a2={};a3={};landmarks=[];world=[]
                    if result.pose_landmarks:
                        detected+=1;landmarks=result.pose_landmarks[0];world=result.pose_world_landmarks[0]
                        for j,ids in JOINTS.items():
                            valid=all(landmarks[k].visibility>=.5 and landmarks[k].presence>=.5 and 0<=landmarks[k].x<=1 and 0<=landmarks[k].y<=1 for k in ids)
                            a2[j]=angle(*[[landmarks[k].x*w,landmarks[k].y*h] for k in ids]) if valid else None
                            a3[j]=angle(*[[world[k].x,world[k].y,world[k].z] for k in ids]) if valid else None
                            if a3[j] is not None:valid_angles[j]+=1
                    for j in JOINTS:
                        row[f'{j}_2d_deg']=a2.get(j);row[f'{j}_3d_deg']=a3.get(j)
                    row['pipeline_ms']=(time.perf_counter()-tick)*1000
                    rows.append(row);lat.append(ms);pipeline.append(row['pipeline_ms'])
                    points.write(json.dumps({'frame':frame_id,'timestamp_ms':ts,'normalized_landmarks':[[k.x,k.y,k.z,k.visibility,k.presence] for k in landmarks],'world_landmarks':[[k.x,k.y,k.z,k.visibility,k.presence] for k in world]},allow_nan=False)+'\n')
                    t=time.perf_counter()
                    if landmarks:
                        for a,b in EDGES:
                            if landmarks[a].visibility>=.5 and landmarks[b].visibility>=.5:
                                cv2.line(frame,(int(landmarks[a].x*w),int(landmarks[a].y*h)),(int(landmarks[b].x*w),int(landmarks[b].y*h)),(0,255,100),2)
                    panel=np.zeros((h,370,3),np.uint8)
                    lines=[f'{model.upper()} | {name}',f'frame {frame_id} | {ts} ms',f'pose {ms:.1f} ms | {delegate.upper()}', '3D included angles (estimated)']
                    lines += [f'{j}: {a3[j]:.1f}' if a3.get(j) is not None else f'{j}: unavailable' for j in JOINTS]
                    lines += ['visibility + presence >= 0.5',sample['split'][:43]]
                    for k,line in enumerate(lines):cv2.putText(panel,line,(8,20+k*22),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1,cv2.LINE_AA)
                    canvas=np.hstack([frame,panel])
                    if writer is None:
                        writer=cv2.VideoWriter(str(out/'preview-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(canvas.shape[1],canvas.shape[0]));assert writer.isOpened()
                    writer.write(canvas)
                    if frame_id==min(120,min(max_frames,sample['video_metadata']['frame_count'])//2):cv2.imwrite(str(out/'preview.jpg'),canvas)
                    draw_write.append((time.perf_counter()-t)*1000)
            elapsed=time.perf_counter()-start;cpu1=proc.cpu_times();cap.release();writer.release();detector.close()
            with (out/'angles.csv').open('w') as f:
                cw=csv.DictWriter(f,fields);cw.writeheader();cw.writerows(rows)
            save(out/'sample-and-labels.json',sample)
            subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(out/'preview-raw.mp4'),'-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'preview.mp4')],check=True)
            check=cv2.VideoCapture(str(out/'preview.mp4'));decoded=0
            while check.read()[0]:decoded+=1
            check.release();assert decoded==len(rows),(decoded,len(rows));(out/'preview-raw.mp4').unlink()
            timed=lat[5:];pipe=pipeline[5:]
            s={'model':model,'dataset':name,'input_split':sample['split'],'frames':len(rows),'warmup_frames_excluded_from_latency':5,'detected_frames':detected,'detected_rate':detected/len(rows),'valid_angle_frames':dict(valid_angles),'init_ms':init_ms,'pose_mean_ms':float(np.mean(timed)),'pose_p95_ms':float(np.percentile(timed,95)),'pose_fps_equivalent':1000/float(np.mean(timed)),'pipeline_mean_ms_decode_resize_pose_angles':float(np.mean(pipe)),'preview_throughput_fps_including_csv_json_drawing_encoding':len(rows)/elapsed,'process_cpu_pct_one_core_mean':100*((cpu1.user+cpu1.system)-(cpu0.user+cpu0.system))/elapsed,'inference_delegate':delegate.upper(),'video_verified_frames':decoded}
            save(out/'metrics.json',s);summaries.append(s);print(json.dumps(s),flush=True)
    finally:
        stop.set();thread.join();save(folder/'resources.json',resources)
    save(folder/'summary.json',summaries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--model',choices=['lite','full','heavy']);p.add_argument('--delegate',choices=['cpu','gpu'],default='cpu');p.add_argument('--max-frames',type=int,default=180);p.add_argument('--run-name');p.add_argument('--samples-json');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.model,a.delegate,a.max_frames,a.run_name,a.samples_json)
