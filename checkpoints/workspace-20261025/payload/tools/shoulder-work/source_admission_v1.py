"""Canonical source-role/group admission, before any label or model admission.

A historical train_eligible flag describes a review and cannot override split.
This guard alone never approves a label, features, model fitting or deployment.
"""
from copy import deepcopy
VERSION='shoulder-source-role-and-alias-admission-v1'
def assess_source(row,canonical_sources,alias_groups,view_receipt=None):
    source_id=row.get('source_id',row.get('id'))
    source=canonical_sources.get(source_id)
    held=[]
    if source is None:
        return dict(version=VERSION,source_id=source_id,training_source_allowed=False,
                    holds=['unknown_canonical_source'],model_fit_approved=False)
    sha=source['source_sha256'];group=alias_groups[source_id]
    if row.get('source_sha256')!=sha:held.append('source_hash_mismatch')
    if row.get('split')!=source['split']:held.append('canonical_role_mismatch')
    if source['split']!='train':held.append('canonical_'+source['split']+'_not_training')
    if row.get('person')!=source['person']:held.append('canonical_person_mismatch')
    view=row.get('view')
    if view!=source['view']:held.append('canonical_view_mismatch')
    if group['metadata_view_conflict']:
        receipt=view_receipt or {}
        if not (receipt.get('source_sha256')==sha and receipt.get('actual_view')==view
                and receipt.get('review_status')=='direct_source_visual_review'):
            held.append('alias_view_conflict_requires_source_visual_receipt')
    return dict(version=VERSION,source_id=source_id,source_sha256=sha,
        canonical_split=source['split'],canonical_person=source['person'],
        canonical_source_group=group['source_group'],person_group=source['person'],
        original_recording_group=source['recording_group'],
        training_source_allowed=not held,holds=held,
        historical_train_eligible_ignored=row.get('train_eligible'),
        label_eligibility_assessed=False,feature_eligibility_assessed=False,
        model_fit_approved=False,source_or_role_modified=False)
