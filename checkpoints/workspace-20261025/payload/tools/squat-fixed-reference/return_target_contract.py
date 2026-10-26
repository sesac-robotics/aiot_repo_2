"""Offline annotation admission; never creates grades, features, scores or models.

The caller supplies the immutable source registry and an independently generated
observation receipt. A visual label may be admissible while fitting remains held.
"""
from copy import deepcopy

VERSION = 'return-target-admission-v1'
READY_RELATIVE = 'observed_ready_relative_lower_return'
ANATOMICAL = 'observed_anatomical_lower_turn'
SCHEMAS = {
    READY_RELATIVE: 'shoulder-ready-return-relative-v3',
    ANATOMICAL: 'shoulder-anatomical-lower-turn-v1',
}

def _window(value):
    return (isinstance(value, list) and len(value) == 2
            and all(type(x) is int and x >= 0 for x in value)
            and value[0] <= value[1])

def assess(annotation, source, *, observation=None, registered_schemas=()):
    """Assess one shoulder event; no mutation and no numeric threshold tuning.

    source must come from the frozen source registry, not from a proposed label.
    registered_schemas means reviewed extractors, not merely a version string.
    No extractor for the new anatomical bank is registered by default.
    """
    a = deepcopy(annotation)
    target = a.get('criterion')
    holds = []
    for key in ['id', 'source_sha256', 'person', 'split', 'view']:
        if a.get(key) != source.get(key) or source.get(key) is None:
            holds.append('canonical_source_mismatch:' + key)
    if target not in SCHEMAS:
        holds.append('unrecognized_target')
    if a.get('exercise') != 'shoulder': holds.append('exercise_target_mismatch')
    if a.get('protocol') != 'shoulder_press': holds.append('press_protocol_not_confirmed')
    if a.get('laterality') != 'bilateral': holds.append('bilateral_target_not_observed')
    if a.get('view') != 'front': holds.append('frontal_target_projection_not_supported')
    if a.get('posture') not in ['standing', 'seated']:
        holds.append('posture_bank_unknown')
    if a.get('label_origin') != 'direct_source_visual_review':
        holds.append('provider_prediction_or_rule_is_not_visual_target')
    if a.get('review_status') != 'reviewed_AI_assisted': holds.append('visual_review_pending')
    if type(a.get('grade')) is not int or a['grade'] not in [0, 1, 2]:
        holds.append('manual_grade_not_assigned')
    if not a.get('evidence_sha256'):
        holds.append('source_visual_evidence_missing')
    if a.get('scope') != 'selected_event_only': holds.append('event_scope_required')
    if a.get('event_source_sha256') != source.get('source_sha256'):
        holds.append('event_uses_other_source')
    peak, turn, following = (a.get(x) for x in ['peak_frames', 'turn_frames', 'next_ascent_frames'])
    event_ok = all(_window(w) for w in [peak, turn])
    rack_closure = (a.get('closure') == 'observed_rack'
                    and a.get('grade') == 2 and a.get('rack_visible_stable') is True)
    next_closure = a.get('closure') == 'observed_next_ascent' and _window(following)
    if not event_ok:
        holds.append('closed_turnaround_not_observed')
    elif not peak[1] < turn[0]:
        holds.append('peak_turn_next_ascent_order_invalid')
    elif not (rack_closure or next_closure):
        holds.append('closed_turnaround_not_observed')
    elif next_closure and turn[1] >= following[0]:
        holds.append('peak_turn_next_ascent_order_invalid')
    elif max(turn[1], following[1] if next_closure else turn[1]) >= source.get('frame_count', 0):
        holds.append('event_outside_original_source')
    if a.get('interrupted_or_clipped') is not False:
        holds.append('interruption_or_clip_end_not_a_failure_label')
    if a.get('actual_pixels_visible') is not True:
        holds.append('event_not_visually_observable')
    if target == READY_RELATIVE:
        ready = a.get('ready_frames')
        if (not _window(ready) or not event_ok or ready[1] >= peak[0]
                or a.get('ready_source_sha256') != source.get('source_sha256')):
            holds.append('causal_same_source_ready_reference_missing')
    # A bilateral anatomical event can be annotated without causal initial rack.
    # It is a DIFFERENT target and never silently supervises the relative head.
    if target == ANATOMICAL and a.get('rubric') != 'anatomical_rack_range_v1':
        holds.append('explicit_anatomical_rubric_required')
    label_ok = not holds
    feature_holds = []
    schema = SCHEMAS.get(target)
    if schema not in registered_schemas:
        feature_holds.append('reviewed_feature_extractor_not_registered')
    if not isinstance(observation, dict) or observation.get('status') != 'passed':
        feature_holds.append('actual_observation_QA_not_passed')
    else:
        if observation.get('source_sha256') != source.get('source_sha256'):
            feature_holds.append('observation_source_mismatch')
        if observation.get('feature_schema') != schema:
            feature_holds.append('feature_target_schema_mismatch')
        if observation.get('target_person_tracking_QA') is not True:
            feature_holds.append('target_person_tracking_unverified')
        if observation.get('actual_10Hz_slots_80pct_and_2pct_border') is not True:
            feature_holds.append('actual_time_and_border_contract_not_passed')
        if observation.get('no_interpolation_or_tracking_break') is not True:
            feature_holds.append('missing_or_tracking_break_not_usable')
        if observation.get('posture') != a.get('posture') or observation.get('view') != a.get('view'):
            feature_holds.append('observation_bank_mismatch')
        if observation.get('event_frames') != [peak, turn, following]:
            feature_holds.append('observation_event_alignment_unverified')
    role = source.get('split')
    training_holds = list(holds) + feature_holds
    if role != 'train': training_holds.append('canonical_' + str(role) + '_not_training')
    evaluation_holds = list(holds) + feature_holds
    if role not in ['validation', 'test']: evaluation_holds.append('not_held_out_evaluation_source')
    return dict(version=VERSION, criterion=target, feature_schema=schema,
        bank=f'{schema}/{a.get("view")}/{a.get("posture")}/bilateral',
        visual_label_eligible=label_ok, visual_label_holds=holds,
        training_eligible=not training_holds, training_holds=training_holds,
        evaluation_input_eligible=not evaluation_holds, evaluation_holds=evaluation_holds,
        manual_grade=a.get('grade') if label_ok else None,
        feature_holds=feature_holds, grade_created=False, score=None,
        production_approved=False, source_role_changed=False,
        expert_adjudicated=False)

def compatible_with_head(admission, head):
    """New anatomy labels/features cannot populate an old relative score head."""
    return (head.get('criterion') == admission['criterion']
            and head.get('feature_schema') == admission['feature_schema']
            and head.get('bank') == admission['bank'])
