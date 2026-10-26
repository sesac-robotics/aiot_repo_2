# Shoulder press — QEVD-style English labels

All training labels use `shoulder press - <posture name>`. The existing stable codes and label strings are preserved. Descriptions use English observational sentences. These are custom shoulder extensions, not official QEVD label classes.

| Stable code | Posture name | Descriptive label | Corrective feedback |
|---|---|---|---|
|NO_OBVIOUS_ISSUE|no obvious issue|shoulder press - No obvious form issue is indicated by the source label|Keep following the reference movement and arm path.|
|TOP_WIDE|arms too wide at the top|shoulder press - User's arms spread too far outward at the top compared with the reference press|Keep your arms from spreading outward at the top; follow the reference arm path.|
|TOP_INCOMPLETE|incomplete upward range|shoulder press - User lowers the weights before reaching the upward range shown in the reference press|Reach the upward position shown in the reference before lowering your arms.|
|HANDS_BEHIND_HEAD|hands move behind the head|shoulder press - User's hands move behind the head while the elbows remain high during lowering|Follow the reference press path from beside your shoulders rather than folding the weights behind your head.|
|FOREARM_INWARD|forearms tilt inward|shoulder press - User's wrists move inward relative to the elbows during lowering|Avoid moving only your hands inward as you lower; match wrist and elbow alignment to the reference.|
|ELBOW_PATH_NARROW|elbow path narrower than reference|shoulder press - User's elbows move closer to the torso at the bottom than in the reference press variation|If following this reference press variation, adjust your elbow spacing slightly wider to match the reference.|
|ARM_ASYMMETRY|uneven arm height and timing|shoulder press - User's arms reach different heights and move at different times during the simultaneous press|Raise both arms together at matching heights and speeds for this simultaneous press variation.|

## Files

- `raw/coach/fine_grained_labels.json`: `video_path`, `labels`, `labels_descriptive`, `split`; identical key structure to the local QEVD dataset.
- `raw/coach/feedbacks_short_clips.json`: English feedback templates, authored for this project.
- `manifests/label_definitions.json`: stable code, English posture name, full label, description and feedback.
- `manifests/annotation_review.json`: English fields `labels_en`, `labels_descriptive_en`, `feedbacks_en`; Korean translations and review provenance are retained.
- `manifests/short_clips.csv`: synchronized label arrays and descriptions.

## Review status

All 100 rows remain drafts. English descriptions do not establish biomechanical correctness. Normal candidates derive from source-good labels. Elbow spacing is reference-variation-specific; hands behind the head may indicate a different exercise. Existing review notes and exclusions are retained. Clip-level labels do not indicate exact onset/offset timing.

The demo shows posture names, descriptions and corrective feedback in English. The application controls may remain Korean.
