"""Freeze a source-grouped, within-person and quality-stratified partition.
This supersedes the count-only partition. No predictions enter optimization.
"""
import collections,hashlib,itertools,json
from pathlib import Path
import numpy as np

B=Path('/home/jayden/SeSACPAI-Mid')
OLD=B/'output/error-enrichment-20261024'
NEW=B/'output/per-person-stratified-80-20-20261024'
ANN=B/'data/r2d2-team-20261023/annotations'
POLICY=ANN/'per-person-stratified-80-20.frozen.json'

def main():
    NEW.mkdir(exist_ok=True)
    labels=json.loads((ANN/'labels.frozen.json').read_text())['labels']
    sources={r['id']:r for r in json.loads((ANN/'sources.json').read_text())}
    inputs=json.loads((OLD/'score-inputs.frozen.json').read_text())
    team={r['id'][5:]:r for r in inputs if r['family']=='R2D2'}
    classes={}
    for row in labels:
        grades=team.get(row['id'],{}).get('levels',[])[:4]
        classes[row['id']]='error' if any(g is not None and g<2 for g in grades) else 'normal' if len(grades)==4 and all(g==2 for g in grades) else 'unknown'
    by_source=collections.defaultdict(list)
    for row in labels:by_source[row['source_id']].append(row)
    if POLICY.exists():policy=json.loads(POLICY.read_text())
    else:
        assignments={};summaries=[];exercise_use=collections.Counter();person_options=[]
        global_details=sorted({(r['exercise'],classes[r['id']]) for r in labels if classes[r['id']]!='unknown'})
        for person in ['P01','P02','P03','P04']:
            videos=sorted(s for s,rs in by_source.items() if rs[0]['person']==person)
            categories=['normal','error','unknown']
            counts=np.array([[sum(classes[r['id']]==c for r in by_source[s]) for c in categories] for s in videos])
            totals=counts.sum(0);n=int(totals.sum())
            details=sorted({(r['exercise'],classes[r['id']]) for s in videos for r in by_source[s] if classes[r['id']]!='unknown'})
            detail_counts=np.array([[sum((r['exercise'],classes[r['id']])==c for r in by_source[s]) for c in details] for s in videos])
            detail_totals=detail_counts.sum(0)
            candidates=[]
            for k in range(1,len(videos)):
                for inds in itertools.combinations(range(len(videos)),k):
                    test=counts[list(inds)].sum(0)
                    if not (test[:2]>0).all() or not (test[:2]<totals[:2]).all():continue
                    if abs(test.sum()/n-.2)>.065:continue
                    ratios=test/np.maximum(totals,1)
                    # Prioritize normal/error proportions, then total/unknown and exercise balance.
                    class_deviation=float(np.max(abs(ratios[:2]-.2)))
                    detail_test=detail_counts[list(inds)].sum(0)
                    detail_loss=float(np.average(abs(detail_test/np.maximum(detail_totals,1)-.2),weights=np.sqrt(detail_totals)))
                    exercises={by_source[videos[i]][0]['exercise'] for i in inds}
                    objective=(round(class_deviation,9),
                               round(float(np.sum(abs(ratios[:2]-.2))),9),
                               round(abs(float(test.sum()/n)-.2)+.15*abs(float(ratios[2])-.2)+.2*detail_loss,9),
                               sum(exercise_use[e] for e in exercises),
                               hashlib.sha256(('stratified80-v1:'+','.join(videos[i] for i in inds)).encode()).hexdigest())
                    candidates.append((objective,inds,test,detail_test))
            if not candidates:raise ValueError('No nonempty class-balanced grouped split for '+person)
            best_by_coverage={}
            for candidate in candidates:
                objective,inds,test,detail_test=candidate
                coverage=sum(1<<global_details.index(details[j]) for j,value in enumerate(detail_test) if value>0)
                if coverage not in best_by_coverage or objective<best_by_coverage[coverage][0]:best_by_coverage[coverage]=candidate
            person_options.append(dict(person=person,videos=videos,totals=totals,n=n,details=details,detail_totals=detail_totals,candidates=best_by_coverage))
        # Global constraints: every exercise has normal AND error examples in Test.
        # Dynamic programming keeps the best quality-ratio solution for each coverage mask.
        states={0:((0.,0.,0.),[])}
        for options in person_options:
            next_states={}
            for old_mask,(old_cost,path) in states.items():
                for mask,candidate in sorted(options['candidates'].items()):
                    cost=tuple(old_cost[i]+candidate[0][i] for i in range(3));combined=old_mask|mask
                    if combined not in next_states or cost<next_states[combined][0]:next_states[combined]=(cost,path+[candidate])
            states=next_states
        full_mask=(1<<len(global_details))-1
        if full_mask not in states:raise ValueError('Cannot cover normal and error for all exercises without relaxing constraints')
        for options,candidate in zip(person_options,states[full_mask][1]):
            person=options['person'];videos=options['videos'];totals=options['totals'];n=options['n'];details=options['details'];detail_totals=options['detail_totals']
            _,inds,test,detail_test=candidate
            chosen={videos[i] for i in inds}
            for s in videos:assignments[s]='test' if s in chosen else 'train'
            exercise_use.update(by_source[s][0]['exercise'] for s in chosen)
            summaries.append(dict(person=person,total_clips=n,train_clips=n-int(test.sum()),test_clips=int(test.sum()),
                classes={c:dict(total=int(totals[i]),train=int(totals[i]-test[i]),test=int(test[i]),test_fraction=float(test[i]/totals[i])) for i,c in enumerate(categories)},
                exercise_class_counts=[dict(exercise=e,quality=c,total=int(detail_totals[i]),train=int(detail_totals[i]-detail_test[i]),test=int(detail_test[i])) for i,(e,c) in enumerate(details)],
                test_source_ids=sorted(chosen),train_sources=len(videos)-len(chosen),test_sources=len(chosen)))
        protected=[s for s,split in assignments.items() if split=='test']
        clip_ids=[r['id'] for r in labels if assignments[r['source_id']]=='test']
        policy=dict(schema='within_person_quality_stratified_whole_source_80_20_v1',
                    requested_train_fraction=.8,quality_definition='Any observed primary item grade <2 => error; all four primary grades 2 => normal; otherwise unknown. Uses documented visual label corrections.',
                    assignments=assignments,people=summaries,protected_people=[],
                    protected_clip_ids=clip_ids+['r2d2_'+i for i in clip_ids],
                    protected_source_sha256=[sources[s]['sha256'] for s in sorted(protected)],
                    test_exercises=sorted(exercise_use),normal_error_nonempty_in_both_sides_for_each_person=True,
                    all_five_exercises_have_normal_and_error_in_test=True,
                    selection_used_predictions=False,selection_used_label_strata=True,
                    split_deviations_expected_because_source_videos_are_indivisible=True,
                    same_person_allowed_across_train_test=True,same_source_allowed_across_train_test=False,
                    historical_exposure='Previously reviewed data, some previously used to fit old models. New models must fit from scratch without these source videos.',
                    final_independent_test_claim_allowed=False,
                    supersedes=str(ANN/'per-person-80-20.frozen.json'))
        assert len(policy['test_exercises'])==5
        POLICY.write_text(json.dumps(policy,ensure_ascii=False,indent=2))
    train=[];test=[]
    source_for={r['id']:r['source_id'] for r in labels}
    for row in inputs:
        if row['family']=='R2D2':row['split']=policy['assignments'][source_for[row['id'][5:]]]
        (test if row['split']=='test' else train).append(row)
    for name,rows in [('score-inputs.frozen.json',train),('test-inputs.frozen.json',test)]:
        p=NEW/name
        if p.exists() and json.loads(p.read_text())!=rows:raise ValueError('Would overwrite frozen inputs')
        p.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
    (NEW/'clip-assignments.frozen.json').write_text(json.dumps([dict(id=r['id'],person=r['person'],exercise=r['exercise'],source_id=r['source_id'],quality_class=classes[r['id']],split=policy['assignments'][r['source_id']]) for r in labels],ensure_ascii=False,indent=2))
    print(json.dumps(dict(people=[{k:v for k,v in r.items() if k in ['person','train_clips','test_clips','classes','test_source_ids']} for r in policy['people']],test_input_rows=len(test),test_exercises=policy['test_exercises']),ensure_ascii=False))

if __name__=='__main__':main()
