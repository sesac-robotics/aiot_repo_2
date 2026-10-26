"""Audit original bytes, canonical roles and existing fixed Test without refitting."""
import collections,datetime,hashlib,json
from pathlib import Path
from source_alias_guard import index_sources,assert_alias_safe_training,assert_disjoint_content
B=Path(__file__).resolve().parents[2]
P=B/'output/squat-fixed-reference-combined-v3-20261024'
O=B/'output/squat-source-alias-audit-v6-20261025'
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):
    p=O/n;assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    O.mkdir(exist_ok=True)
    canonical={}
    for r in read(B/'output/overnight-20261020/score_candidates.json'):
        if r['exercise']=='shoulder':
            canonical.setdefault(r['person'],set()).add(r['split'])
    assert all(len(v)==1 for v in canonical.values())
    root=B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise/squat'
    records=[]
    for p in sorted(root.rglob('*.mp4')):
        person='shoulder_'+p.stem.split('_')[1]
        role=next(iter(canonical[person]))
        records.append(dict(id=p.stem,path=str(p),source_sha256=sha(p),person=person,split=role,role_origin='existing_MultiView_person_role_registry',provider_tag=p.parent.parent.name))
    assert len(records)==150
    groups=index_sources(records)
    duplicates=[dict(source_sha256=h,aliases=rs,roles=sorted({r['split'] for r in rs}),people=sorted({r['person']for r in rs}),provider_tags=sorted({r['provider_tag']for r in rs}))for h,rs in groups.items()if len(rs)>1]
    inputs=read(P/'score-inputs.frozen.json');test=read(P/'test-inputs.frozen.json')
    actual_train=[r for r in inputs if r['split']=='train' and any(i['eligible'] and g is not None for i,g in zip(r['items'],r['levels']))]
    assert_alias_safe_training(actual_train,records)
    assert_disjoint_content(actual_train,test)
    exposures=[]
    for d in duplicates:
        h=d['source_sha256'];matches=[dict(id=r['id'],person=r['person'],split=r['split'],eligible_items=[i['item']for i,g in zip(r['items'],r['levels']) if i['eligible'] and g is not None])for r in inputs+test if h in r['sources']]
        exposures.append(dict(source_sha256=h,input_rows=matches))
    head=read(P/'new-head-training.json')
    lookup={r['id']:r for r in inputs}
    for fold in head['nested_CV_folds']:
        assert_disjoint_content([lookup[i] for i in fold['fit_ids']],[lookup[i]for i in fold['outer_ids']])
        for f in fold['inner_members']:
            assert_disjoint_content([lookup[i]for i in f['fit_ids']],[lookup[i]for i in f['held_ids']])
    for f in head['component_training']:
        assert_disjoint_content([lookup[i]for i in f['fit_ids']],[lookup[i]for i in f['held_ids']])
    for name,h in read(P/'artifact-hashes.frozen.json').items():assert sha(P/name)==h
    paired=read(P/'comparison.frozen.json')['paired_predictions'];assert len(paired)==163
    first=read(B/'output/squat-fixed-reference-v1-20261024/multiview-review/intake.frozen.json')['records']
    later=read(B/'output/squat-pose-alignment-v5-20261024/remaining-Train/intake.frozen.json')
    assert len(first+later)==15
    unique_review=len({r['source_sha256']for r in first+later})
    assert unique_review==14
    policy=B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'
    labels=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
    assert sha(policy)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
    assert sha(labels)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
    save('sources.frozen.json',records);save('duplicate-groups.frozen.json',duplicates);save('existing-input-exposure.frozen.json',exposures)
    save('review-count-correction.frozen.json',dict(previous_scope='15 canonical Train bad/side filename records',corrected_distinct_byte_sources=14,independent_people_claim_allowed=False,reason='005 bad side and 006 bad side are byte-identical; filename person identity unresolved',draft_005_retained=True,draft_training_grade=None,new_actual_error_labels=0,new_independent_error_people=0,old_frozen_artifacts_preserved=True))
    cross=[d for d in duplicates if len(d['roles'])>1]
    quarantine=[dict(source_sha256=d['source_sha256'],ids=[r['id']for r in d['aliases']],roles=d['roles'],action='block_future_Train_fit_no_split_reassignment')for d in cross]
    save('future-Train-quarantine.frozen.json',quarantine)
    result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_files=150,distinct_byte_sources=len(groups),duplicate_groups=len(duplicates),mixed_provider_tag_groups=sum(len(d['provider_tags'])>1 for d in duplicates),cross_role_alias_groups=len(cross),actual_existing_eligible_Train_alias_conflicts=0,existing_Train_Test_hash_overlap=0,existing_squat_head_nested_folds_content_disjoint=True,parent_artifacts_hash_verified=True,prior_15_side_review_records_unique_sources=unique_review,source_split_changed=False,labels_changed=False,new_error_labels=0,new_fit_rows=0,paired_Test_items_unchanged=163,independent_final_Test_pixels_opened=False,production_changed=False,guard_tests=6)
    save('verification.frozen.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
