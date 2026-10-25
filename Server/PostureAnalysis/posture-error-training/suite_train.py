"""Train per-exercise CNN/GRU with source-test preservation and honest support masks."""
from collections import defaultdict
from pathlib import Path
import argparse,copy,csv,hashlib,json,random,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score,confusion_matrix
from models import MODELS
from suite_config import BASE,SUITE,TASKS,CONFIG_VERSION,SEQUENCE_LENGTH,SAMPLE_FPS
from suite_features import FEATURE_NAMES

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def duplicate_audit(exercise):
    directory=SUITE/exercise;records=json.loads((directory/'selected.json').read_text());groups=defaultdict(list)
    for r in records:
        with open(r['video_path'],'rb') as f:key=hashlib.file_digest(f,'sha256').hexdigest()
        groups[key].append(r)
    excluded=set();duplicates=[]
    for rows in groups.values():
        if len(rows)<2:continue
        conflict=len({tuple(r['target']) for r in rows})>1
        ordered=sorted(rows,key=lambda r:(r['source_split']!='test',r['video_id']))
        removed=ordered if conflict else ordered[1:]
        excluded.update(r['video_id'] for r in removed)
        duplicates.append(dict(video_ids=[r['video_id'] for r in rows],conflicting_targets=conflict,excluded=[r['video_id'] for r in removed]))
    dump(directory/'duplicate_audit.json',dict(groups=duplicates,excluded_ids=sorted(excluded),policy='Exclude all conflicting duplicates; otherwise keep source-test first, then lowest ID.'))
    return excluded

def scores(y,p,targets,train_counts=None,val_counts=None):
    result={};per={};eligible=[]
    for j,t in enumerate(targets):
        npos=int(y[:,j].sum());nneg=len(y)-npos
        train_ok=train_counts is None or 0<train_counts[j]<train_counts[-1]
        valid=npos>0 and nneg>0 and train_ok
        precision,recall,f1,_=precision_recall_fscore_support(y[:,j],p[:,j],average='binary',zero_division=0)
        per[t]=dict(positive_support=npos,negative_support=nneg,train_positive_support=int(train_counts[j]) if train_counts is not None else None,validation_positive_support=int(val_counts[j]) if val_counts is not None else None,evaluable=bool(valid),precision=float(precision) if valid else None,recall=float(recall) if valid else None,f1=float(f1) if valid else None,reason=None if valid else 'Missing positive/negative test support or unseen training target')
        if valid:eligible.append(j)
    for metric,key in [('precision','precision_macro_supported'),('recall','recall_macro_supported'),('f1','macro_f1_supported')]:
        result[key]=float(np.mean([per[targets[j]][metric] for j in eligible])) if eligible else None
    result.update(evaluated_targets=len(eligible),total_targets=len(targets),macro_f1_all_targets=result['macro_f1_supported'] if len(eligible)==len(targets) else None,exact_match_accuracy=float(accuracy_score(y,p)),per_target=per)
    return result

def binary_metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    precision,recall,_,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    enough=len(np.unique(y))==2
    return dict(accuracy=float(accuracy_score(y,p)),precision=float(precision) if enough else None,recall=float(recall) if enough else None,macro_f1=float(f1_score(y,p,average='macro',zero_division=0)) if enough else None,positive_support=int(y.sum()),negative_support=int(len(y)-y.sum()),confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist(),evaluable=enough)

def choose_thresholds(y,probs,active):
    result=np.full(y.shape[1],.5)
    for j in range(y.shape[1]):
        if active[j] and 0<y[:,j].sum()<len(y):
            candidates=np.arange(.1,.91,.05)
            result[j]=max(candidates,key=lambda t:f1_score(y[:,j],probs[:,j]>=t,zero_division=0))
    return result

def fit(exercise,epochs):
    config=TASKS[exercise];directory=SUITE/exercise;out=directory/'outputs';out.mkdir(exist_ok=True)
    excluded=duplicate_audit(exercise)
    with np.load(directory/'data.npz') as raw:
        keep=np.array([v not in excluded for v in raw['ids']]);X=raw['X'][keep];y=raw['y'][keep].astype(int);g=raw['groups'][keep];ids=raw['ids'][keep];source_split=raw['source_split'][keep]
    assert len(X)>20 and np.isfinite(X).all()
    source_train=np.flatnonzero(source_split=='train');test=np.flatnonzero(source_split=='test')
    source_counts=y[source_train].sum(axis=0)
    splitter=GroupShuffleSplit(n_splits=100,test_size=.2,random_state=42)
    chosen_split=None
    for candidate,(a,b) in enumerate(splitter.split(X[source_train],groups=g[source_train])):
        candidate_train,candidate_val=source_train[a],source_train[b]
        counts=y[candidate_train].sum(axis=0)
        covered=np.all(counts[source_counts>0]>0) and np.all(counts[source_counts<len(source_train)]<len(candidate_train))
        validation_binary=y[candidate_val].any(axis=1)
        if covered and len(np.unique(validation_binary))==2:
            train,val=candidate_train,candidate_val;chosen_split=candidate;break
    if chosen_split is None:raise ValueError('Cannot make a group-disjoint split covering source-train classes; do not use test to fix it')
    assert len(test)>0 and len(train)>0 and len(val)>0
    assert not(set(g[train])&set(g[val]) or set(g[train])&set(g[test]) or set(g[val])&set(g[test]))
    splits=dict(train=train,validation=val,test=test)
    split_report={name:dict(videos=len(idx),workers=sorted(map(int,set(g[idx]))),video_ids=ids[idx].tolist(),positive_counts=dict(zip(config['targets'],map(int,y[idx].sum(axis=0)))),any_selected_error_count=int(y[idx].any(axis=1).sum())) for name,idx in splits.items()}
    dump(out/'splits.json',split_report)
    Y=np.concatenate([y,y.any(axis=1)[:,None]],axis=1);targets=config['targets']+['any_selected_error']
    active=(Y[train].sum(axis=0)>0)&(Y[train].sum(axis=0)<len(train))
    if not active.any():raise ValueError('No learnable targets')
    mean=X[train].mean(axis=(0,1));std=np.maximum(X[train].std(axis=(0,1)),1e-5);Z=(X-mean)/std
    tensors={name:torch.tensor(Z[idx],dtype=torch.float32) for name,idx in splits.items()}
    loader=DataLoader(TensorDataset(tensors['train'],torch.tensor(Y[train],dtype=torch.float32)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42))
    positives=Y[train].sum(axis=0);weights=np.clip((len(train)-positives)/np.maximum(positives,1),.1,20)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32),reduction='none');mask=torch.tensor(active,dtype=torch.float32)
    validation_mask=active&(Y[val].sum(axis=0)>0)&(Y[val].sum(axis=0)<len(val))
    # Sparse shoulder error classes cannot be selected independently across 3 disjoint sets.
    selection_mask=validation_mask.copy()
    if exercise=='shoulder_press':selection_mask[:-1]=False
    if not selection_mask.any():raise ValueError('No evaluable validation target; do not use test to repair split')
    model_metrics={};history={};checkpoints={};start=time.time()
    for architecture,constructor in MODELS.items():
        torch.manual_seed(42);model=constructor(X.shape[2],len(targets));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.005)
        best=-1.;stale=0;curve=[]
        for epoch in range(1,epochs+1):
            model.train();total=0.
            for xb,yb in loader:
                optimizer.zero_grad();loss=(loss_fn(model(xb),yb)*mask).sum()/(len(xb)*mask.sum());loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=loss.item()*len(xb)
            model.eval()
            with torch.inference_mode():probs=torch.sigmoid(model(tensors['validation'])).numpy()
            thresholds=choose_thresholds(Y[val],probs,active)
            current=float(f1_score(Y[val][:,selection_mask],probs[:,selection_mask]>=thresholds[selection_mask],average='macro',zero_division=0))
            curve.append(dict(epoch=epoch,train_loss=total/len(train),validation_macro_f1=current))
            if current>best+1e-5:
                best=current;best_state=copy.deepcopy(model.state_dict());best_thresholds=thresholds.copy();best_epoch=epoch;stale=0
            else:stale+=1
            if epoch%5==0:print(f'{exercise}/{architecture} epoch={epoch} val={current:.4f} best={best:.4f}',flush=True)
            if stale>=12:break
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():probs=torch.sigmoid(model(tensors['test'])).numpy()
        pred=probs>=best_thresholds;pred[:,~active]=False
        m=scores(y[test],pred[:,:-1],config['targets'],[*y[train].sum(axis=0),len(train)],y[val].sum(axis=0))
        # Binary auxiliary is learned separately and is not proof of universally normal form.
        m['selected_error_vs_other']=binary_metrics(Y[test,-1],pred[:,-1]);m.update(validation_selection_score=best,best_epoch=best_epoch,thresholds=dict(zip(targets,map(float,best_thresholds))))
        model_metrics[architecture]=m;history[architecture]=curve
        checkpoint=dict(state_dict=best_state,architecture=architecture,exercise=exercise,targets=config['targets'],feature_count=X.shape[2],feature_names=FEATURE_NAMES,mean=mean.tolist(),std=std.tolist(),thresholds=best_thresholds.tolist(),active_targets=active.tolist(),required_joints=config['required'],sequence_length=SEQUENCE_LENGTH,sample_fps=SAMPLE_FPS,config_version=CONFIG_VERSION,normal_supported=config['normal_supported'],scope=config['scope'],label_status='draft' if exercise=='shoulder_press' else 'source_dataset_annotation_unassessed')
        torch.save(checkpoint,out/f'{architecture}.pt');checkpoints[architecture]=checkpoint
        with (out/f'{architecture}_predictions.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['video_id','worker_id']+['true_'+t for t in targets]+['prob_'+t for t in targets]);writer.writerows([[ids[i],int(g[i]),*Y[i].tolist(),*p.tolist()] for i,p in zip(test,probs)])
    winner=max(model_metrics,key=lambda n:model_metrics[n]['validation_selection_score']);torch.save(checkpoints[winner],out/'best.pt')
    majority=np.tile(y[train].mean(axis=0)>=.5,(len(test),1));baseline=scores(y[test],majority,config['targets'],[*y[train].sum(axis=0),len(train)],y[val].sum(axis=0))
    baseline['selected_error_vs_other']=binary_metrics(Y[test,-1],np.full(len(test),Y[train,-1].mean()>=.5))
    limitations=[config['scope'],'Source clip labels; not frame-level or expert-verified error truth.','Unlisted selected labels treated as negative (closed-world assumption).','All models use same source test and group-disjoint validation; no test tuning.','Invisible joints are masked, but occlusion, camera view and annotation bias remain.','Selected-error auxiliary negative is not a comprehensive normal class.','No validated rule baseline for these four tasks; squat proxy comparison remains separate.']
    if exercise=='shoulder_press':limitations+=['Only 25 source subjects and 30 draft error videos; multiple views are correlated.','Source test contains only one positive error clip, so headline accuracy can be misleading.','Some errors occur in only one or two subjects; independent train/validation/test error evaluation is impossible.']
    if exercise=='ytw':limitations+=['No verified upright YTW subset. This model is exploratory and must not be used as a validated upright-form assessor.']
    report=dict(validation_split_candidate=chosen_split,validation_split_policy='First seed42 candidate retaining every available source-train target in train; validation binary has both classes; test is never inspected.',exercise=exercise,targets=config['targets'],selected_model=winner,selection_basis='Validation score only; shoulder uses selected-error auxiliary validation F1 due to sparse class support.',models=model_metrics,majority_baseline=baseline,trainable_targets=dict(zip(targets,map(bool,active))),split_sizes={k:len(v) for k,v in splits.items()},worker_overlap=0,elapsed_seconds=time.time()-start,limitations=limitations,dataset_sha256=hashlib.sha256((directory/'data.npz').read_bytes()).hexdigest(),pipeline_version=CONFIG_VERSION)
    dump(out/'metrics.json',report);dump(out/'history.json',history)
    text=['# '+config['name']+' 학습 결과','',config['scope'],'',f'선택 모델: {winner}. 피험자 분리: '+str(report['split_sizes']),'','| 모델 | 선택 오류/기타 정확도 | 평가 가능한 오류 Macro F1 | 평가 오류 수 |','|---|---:|---:|---:|']
    for n,m in model_metrics.items():text.append(f'| {n} | {m["selected_error_vs_other"]["accuracy"]:.4f} | {m["macro_f1_supported"] if m["macro_f1_supported"] is not None else "평가 불가"} | {m["evaluated_targets"]}/{m["total_targets"]} |')
    text+=['','Macro F1 전체값은 모든 대상에 양성·음성 테스트 지지가 있을 때만 제공합니다. 지지가 없는 클래스는 null이며 0점이나 100점으로 채우지 않습니다.','',*['- '+s for s in limitations]]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n');print(f'Finished {exercise}: {winner}; sizes={report["split_sizes"]}',flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--exercise',choices=list(TASKS));p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
    torch.set_num_threads(4);np.random.seed(42);random.seed(42);torch.manual_seed(42)
    for exercise in ([a.exercise] if a.exercise else TASKS):fit(exercise,a.epochs)

if __name__=='__main__':main()
