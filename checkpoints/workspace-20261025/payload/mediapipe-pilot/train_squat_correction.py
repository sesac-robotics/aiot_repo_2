"""Train actual label-supervised squat classifiers, split strictly by person."""
from pathlib import Path
import csv,json,time,hashlib,subprocess
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,balanced_accuracy_score
from pilot import ROOT,DATA,JOINTS

C=ROOT/'correction';C.mkdir(exist_ok=True)
TRAIN={'Petra','Terka','Vasil'};VAL={'Mima'};TEST={'Honza'}
BODY=[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
CLASSES=['NORMAL','ERR_KNEES_INSIDE','ERR_TORSO_TO_SIDE','ERR_SHALLOW_SQUAT','ERR_KNEES_FORWARD','ERR_LIFTED_HEELS']
KOREAN={'NORMAL':'정상','ERR_KNEES_INSIDE':'무릎이 안쪽으로 모임','ERR_TORSO_TO_SIDE':'몸통이 옆으로 기울어짐','ERR_SHALLOW_SQUAT':'스쿼트 깊이 부족','ERR_KNEES_FORWARD':'무릎이 앞으로 많이 이동','ERR_LIFTED_HEELS':'뒤꿈치 들림'}
ADVICE={'NORMAL':'이 반복에서 분류기가 정상 패턴을 예측했습니다. 전체 운동의 안전성을 보증하지 않습니다.','ERR_KNEES_INSIDE':'무릎이 안쪽으로 모이는지 확인하고, 발 방향과 무릎 움직임을 다시 확인하세요.','ERR_TORSO_TO_SIDE':'몸통이 한쪽으로 기울어지는지 확인하고 좌우 균형을 점검하세요.','ERR_SHALLOW_SQUAT':'반복의 내려가는 범위를 확인하세요. 개인별 적정 깊이는 이 모델이 정하지 않습니다.','ERR_KNEES_FORWARD':'무릎 전방 이동 패턴을 점검하세요. 무릎이 발끝을 넘었다는 사실만으로 잘못된 자세라고 단정하지 않습니다.','ERR_LIFTED_HEELS':'동작 중 뒤꿈치가 바닥에서 들리는지 확인하세요.'}

def feature_sequence(world,visibility):
    """No labels, identities, view angles, repetition numbers or absolute time inputs."""
    world=np.asarray(world,float);visibility=np.asarray(visibility,float)
    hip=(world[:,23]+world[:,24])/2
    scale=(np.linalg.norm(world[:,11]-world[:,23],axis=1)+np.linalg.norm(world[:,12]-world[:,24],axis=1))/2
    valid_scale=np.isfinite(scale)&(scale>1e-4)
    coords=(world[:,BODY]-hip[:,None,:])/np.where(valid_scale,scale,np.nan)[:,None,None]
    good=np.isfinite(world).all(axis=2)&(visibility>=.5)
    coords[~good[:,BODY]]=np.nan
    columns=[coords.reshape(len(world),-1)]
    for a,b,c in JOINTS.values():
        u=world[:,a]-world[:,b];v=world[:,c]-world[:,b]
        den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cos=np.sum(u*v,axis=1)/np.maximum(den,1e-12)
        deg=np.degrees(np.arccos(np.clip(cos,-1,1)))/180
        deg[~(good[:,a]&good[:,b]&good[:,c])|(den<1e-10)]=np.nan
        columns.append(deg[:,None])
    for a,b in [(25,26),(27,28),(29,31),(30,32)]:
        z=np.linalg.norm(world[:,a]-world[:,b],axis=1)/np.where(valid_scale,scale,np.nan)
        z[~(good[:,a]&good[:,b])]=np.nan;columns.append(z[:,None])
    seq=np.concatenate(columns,axis=1)
    coverage=float(np.mean(np.isfinite(seq)))
    if coverage<.6 or len(seq)<10:return None,coverage
    # Missing original coordinates stay missing; interpolation below is only a
    # disclosed classifier feature transform within an annotated repetition.
    result=[]
    grid=np.linspace(0,1,16);time_axis=np.linspace(0,1,len(seq))
    for col in seq.T:
        ok=np.isfinite(col)
        if ok.sum()<3:result.extend([np.nan]*21);continue
        q=np.quantile(col[ok],[.05,.25,.5,.75,.95])
        result.extend(q);result.extend(np.interp(grid,time_axis[ok],col[ok]))
    return np.asarray(result,np.float32),coverage

def metrics(model,X,y):
    pred=model.predict(X)
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'report':classification_report(y,pred,output_dict=True,zero_division=0),'labels':list(model.classes_),'confusion_matrix':confusion_matrix(y,pred,labels=model.classes_).tolist()}

def main():
    started=time.perf_counter();r=DATA/'rehab26-squat';annotations=[x for x in csv.DictReader((r/'raw/annotations/Segmentation.csv').open(),delimiter=';') if x['exercise_id']=='6'];videos=list(csv.DictReader((r/'manifests/videos.csv').open()))
    items=[];features=[];errors=[]
    for video in videos:
        if video['exercise']!='ex6':continue
        data=np.load(r/video['coordinates_3d'],allow_pickle=False)
        for a in annotations:
            if a['video_id']!=video['recording_id']:continue
            first,last=int(a['first_frame']),int(a['last_frame']);assert 0<=first<=last<len(data)
            assert (a['correctness_score']=='1' and not a['error_description']) or (a['correctness_score']=='5' and a['error_description'] in CLASSES)
            label='NORMAL' if a['correctness_score']=='1' else a['error_description'];clip=data[first:last+1]
            feat,coverage=feature_sequence(clip[:,:,:3],clip[:,:,3])
            split='train' if a['person'] in TRAIN else ('validation' if a['person'] in VAL else 'test')
            item={'video_id':video['video_id'],'recording':a['video_id'],'person':a['person'],'camera':video['camera'],'first_frame':first,'last_frame':last,'orientation':a['person_orientation'],'repetition':a['repetition_number'],'split':split,'label':label,'coverage':coverage}
            if feat is None:errors.append(item);continue
            items.append(item);features.append(feat)
    X=np.asarray(features);y=np.array([x['label'] for x in items]);splits=np.array([x['split'] for x in items]);tr=splits=='train';va=splits=='validation';te=splits=='test'
    assert set(x['person'] for x in items if x['split']=='train').isdisjoint(TEST|VAL)
    np.savez_compressed(C/'training-features.npz',X=X,y=y,splits=splits)
    (C/'sample-manifest.json').write_text(json.dumps(items,ensure_ascii=False,indent=2));(C/'excluded-low-quality.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    results={};selected={}
    for task in ['normal_error','error_type']:
        target=np.where(y=='NORMAL','NORMAL','ERROR') if task=='normal_error' else y
        candidates=[]
        for leaf in [1,3,7]:
            tick=time.perf_counter();model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('classifier',ExtraTreesClassifier(n_estimators=300,min_samples_leaf=leaf,max_features='sqrt',class_weight='balanced',random_state=37,n_jobs=4))])
            model.fit(X[tr],target[tr]);score=metrics(model,X[va],target[va]);candidates.append((score['macro_f1'],model,leaf,time.perf_counter()-tick))
        best=max(candidates,key=lambda x:x[0]);model=best[1]
        # Final test is read only after model selection on validation.
        results[task]={'selected_min_samples_leaf':best[2],'training_seconds_selected':best[3],'candidate_validation_macro_f1':{str(x[2]):x[0] for x in candidates},'validation':metrics(model,X[va],target[va]),'test':metrics(model,X[te],target[te])}
        joblib.dump(model,C/(task+'.joblib'));selected[task]=model
    summary={'exercise':'squat','source':'REHAB26 provided MediaPipe 3D joints, not new Lite/Full/Heavy features','train_people':sorted(TRAIN),'validation_people':sorted(VAL),'test_people':sorted(TEST),'unique_repetitions':len(annotations),'samples_after_quality_filter':len(items),'excluded':len(errors),'counts':{s:{'samples':int(sum(splits==s)),'class_counts':{c:int(sum((splits==s)&(y==c))) for c in sorted(set(y))}} for s in ['train','validation','test']},'feature_dim':X.shape[1],'feature_interpolation':'Within annotated repetition, 16 relative-time samples per geometric channel; no future repetition or identity inputs. Predictions available after complete repetition only.','results':results,'total_preparation_and_training_seconds':time.perf_counter()-started,'source_annotations_sha256':hashlib.sha256((r/'raw/annotations/Segmentation.csv').read_bytes()).hexdigest(),'source_frame_semantics':'0-based inclusive, synchronized cams, 30 fps per provider Segmentation.txt','estimator_training_device':'CPU 4 threads, compact tree classifiers; MediaPipe video extraction uses RTX GPU','no_medical_validation':True}
    (C/'training-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    # Display only a completely held-out subject; no training/validation videos.
    video=next(x for x in videos if x['video_id']=='ex6_Honza_cam1');source=r/video['video_path']
    import cv2
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    demo=[{'dataset':'REHAB26','video':str(source),'split':'held-out TEST: Honza, never used in fitting or model selection','video_id':video['video_id'],'labels':[x for x in annotations if x['person']=='Honza'],'label_kind':'0-based inclusive repetition segments; correctness 1 normal / 5 error','video_metadata':{'width':w,'height':h,'fps':fps,'frame_count':frames,'bytes':source.stat().st_size}}]
    (C/'demo-samples.json').write_text(json.dumps(demo,ensure_ascii=False,indent=2))
    print(json.dumps({'counts':summary['counts'],'seconds':summary['total_preparation_and_training_seconds'],'results':{k:{'validation_macro_f1':v['validation']['macro_f1'],'test_accuracy':v['test']['accuracy'],'test_macro_f1':v['test']['macro_f1']} for k,v in results.items()}},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
