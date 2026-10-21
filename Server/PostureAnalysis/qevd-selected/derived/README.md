# QEVD English feedback mapping

63 original label names are preserved. English feedback templates cover all 18,652 local clips, including all 3,597 YTW clips. Original raw annotations and the 7,107 original feedback records are unchanged.

Templates are authored examples, not expert-validated ground truth or newly observed video diagnoses. No model training or new pose inference was performed.

## Files

- `coach/feedbacks_short_clips.json`: same three-field schema as QEVD original feedback.
- `manifests/feedback_label_mapping.json`: exact original label -> category -> English feedback.
- `manifests/feedback_review.json`: per-clip label associations and selection rules.
- `checks/validation_report.json`: coverage and raw source hashes.

## Cue selection

Visibility cues take priority when the subject is not visible. Specific error/timing/sequence cues take priority over generic positive attributes. Every original label-to-feedback association is retained in the review sidecar, even when not selected for display.

Do not classify every label as an error: spacing, depth, phase and exercise variations can be context-specific. The source `squats - knees over toes` conflicts with its original description; it is flagged for review and does not produce a knees-behind-toes instruction.

## Label dictionary

| Original label | Category | English feedback |
|---|---|---|
|elbow plank - arms facing inward|error_candidate|Match your forearm direction to the reference plank position.|
|elbow plank - arms facing inward and legs and hips on the floor|error_candidate|Match your forearm direction to the reference and lift your hips and legs into the demonstrated plank position.|
|elbow plank - arms facing inward and on knees|variant_or_error|Check your forearm direction. If a full plank is intended, lift your knees to match the reference; otherwise keep the intended knee-supported variation.|
|elbow plank - arms too wide|error_candidate|Bring your forearms closer together to match the reference plank position.|
|elbow plank - average|ambiguous|Continue with a controlled plank and compare your alignment with the reference; this label does not specify a particular error.|
|elbow plank - butt sinking|error_candidate|Lift your hips toward the straight body line shown in the reference.|
|elbow plank - head down|error_candidate|Bring your head into line with your torso, following the reference neck position.|
|elbow plank - head up|error_candidate|Avoid lifting your head away from your body line; follow the reference neck position.|
|elbow plank - high hips|error_candidate|Lower your hips toward the straight body line shown in the reference.|
|elbow plank - legs and hips on the floor|error_candidate|Lift your hips and legs from the floor to match the demonstrated plank position.|
|elbow plank - on knees|variant_or_error|If a full plank is intended, lift your knees to match the reference; retain knee support if that is the intended variation.|
|elbow plank - shaking|error_candidate|Reset your position and use a controlled hold; pause if you cannot maintain the demonstrated alignment.|
|elbow plank - shoulders shrugged up toward the ears|error_candidate|Keep your shoulders from rising toward your ears and follow the reference shoulder position.|
|elbow plank - stopping early|timing_candidate|Complete the planned hold duration while you can maintain the reference position; pause if needed.|
|pushups - 90 degrees|reference_attribute|Maintain the demonstrated push-up depth and controlled arm movement.|
|pushups - butt sinking|error_candidate|Lift your hips to keep your torso aligned during the push-up.|
|pushups - child pose cheat|variant_or_error|Return to the demonstrated push-up position when ready; keep rest positions separate from active repetitions.|
|pushups - deep|reference_deviation|Match the demonstrated lower position instead of descending beyond the intended range.|
|pushups - early stopping|timing_candidate|Complete the planned repetitions while maintaining the reference form; pause if needed.|
|pushups - elevated hips|error_candidate|Lower your hips toward the straight body line shown in the reference push-up.|
|pushups - fast|timing_candidate|Slow the movement to match the controlled tempo in the reference.|
|pushups - head down|error_candidate|Keep your head aligned with your torso instead of dropping it during the push-up.|
|pushups - head moving in an out of frame|visibility|Adjust the camera so your head and full body remain visible throughout the push-up.|
|pushups - insufficient|ambiguous|Compare the full repetition with the reference; the source label does not identify a specific joint correction.|
|pushups - legs on the floor|variant_or_error|If a full push-up is intended, lift your legs into the reference position; check whether a supported variation is intended.|
|pushups - moving head|error_candidate|Keep your head aligned with your torso as your body moves through the push-up.|
|pushups - narrow|reference_deviation|For this reference variation, move your hands slightly farther apart to match the demonstrated spacing.|
|pushups - not visible|visibility|Adjust the camera so your full body is visible before assessing push-up form.|
|pushups - shallow|error_candidate|Lower your body to the demonstrated depth before pressing back up.|
|pushups - shoulder-width|reference_attribute|Maintain the hand spacing shown in the reference push-up.|
|pushups - side to side|error_candidate|Keep your body centered and reduce side-to-side movement during each repetition.|
|pushups - sloppy|ambiguous|Reset and follow the reference movement with control; this label does not identify a specific joint error.|
|pushups - very wide|reference_deviation|For this reference variation, bring your hands closer together to match the demonstrated spacing.|
|pushups - waiting position|phase_attribute|Begin the next repetition from the reference starting position when ready.|
|pushups - wide|variant_or_error|Check your hand spacing against the intended push-up variation before changing it.|
|squats - 90 degrees|reference_attribute|Maintain the squat depth demonstrated in the reference variation.|
|squats - back not straight|error_candidate|Keep your torso controlled and match the back alignment demonstrated in the reference squat.|
|squats - hold|phase_attribute|Maintain the demonstrated bottom hold if it is part of the planned repetition.|
|squats - insufficient|ambiguous|Compare the full squat with the reference; this label does not specify a particular joint correction.|
|squats - knees over toes|source_conflict|Knee-position annotation needs review before a corrective cue can be assigned.|
|squats - narrow|reference_deviation|For this reference squat variation, move your feet slightly farther apart to match the demonstrated stance.|
|squats - no obvious issue|normal_attribute|Maintain the reference squat form and controlled movement.|
|squats - not visible|visibility|Adjust the camera so your full body is visible before assessing squat form.|
|squats - over 90 degrees|reference_deviation|Match the intended reference squat depth; a deeper squat is not automatically an error in every variation.|
|squats - plie|variant_or_error|Confirm the intended squat variation before changing a wide stance or outward-pointing toes.|
|squats - shallow|error_candidate|Lower into the demonstrated squat depth before standing back up.|
|squats - shoulder-width|reference_attribute|Maintain the foot spacing demonstrated in the reference squat.|
|squats - starting late|timing_candidate|Begin the movement with the planned cue or reference timing when ready.|
|squats - wide|reference_deviation|For this reference squat variation, bring your feet closer together to match the demonstrated stance.|
|standing twy - wrong order|sequence_candidate|Perform the positions in the intended Y, then T, then W order.|
|standing tyw - wrong order|sequence_candidate|Perform the positions in the intended Y, then T, then W order.|
|standing wty - wrong order|sequence_candidate|Perform the positions in the intended Y, then T, then W order.|
|standing wyt - wrong order|sequence_candidate|Perform the positions in the intended Y, then T, then W order.|
|standing ytw - arms too low|error_candidate|Raise your arms to the height demonstrated for the current Y, T, or W position.|
|standing ytw - bent legs|reference_deviation|Match the leg position demonstrated in the intended Y-T-W variation.|
|standing ytw - hands flat|reference_deviation|Match your hand and thumb orientation to the reference Y-T-W position.|
|standing ytw - not low enough|ambiguous|Match the lower position demonstrated for the current Y-T-W phase; confirm which body segment the label refers to before assigning a joint-specific cue.|
|standing ytw - thumbs pointed down|reference_deviation|Rotate your thumbs to the direction demonstrated in the reference Y-T-W position.|
|standing ytw - thumbs pointed sideway|reference_deviation|Adjust your thumb direction to match the reference Y-T-W position.|
|standing ytw - too fast|timing_candidate|Slow down and complete each Y, T, and W position at the demonstrated tempo.|
|standing ytw - too slow|timing_candidate|Match the planned Y-T-W tempo while keeping each position controlled.|
|standing ytw - torso parallel to the ground, thumbs pointed up|variant_or_error|Confirm the intended Y-T-W variation and match its demonstrated torso and thumb positions.|
|standing ywt - wrong order|sequence_candidate|Perform the positions in the intended Y, then T, then W order.|
