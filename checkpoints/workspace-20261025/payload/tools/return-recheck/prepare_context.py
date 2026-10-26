"""Generate contextual features without changing any existing labels/source roles."""
from pathlib import Path
B=Path(__file__).resolve().parents[2];s=(B/'tools/return-discrimination/prepare.py').read_text();s=s.replace("O=B/'output/return-discrimination-v1-20261024'","O=B/'output/return-context-20261024'").replace("P=B/'output/mild-return-calibrated-20261024'","P=B/'output/return-discrimination-ensemble-20261024'")
s=s.replace("new_bank='standing_support_extension_v1'","new_bank='standing_support_context_v1',context_seconds=.4,context_policy='same-source bounded post-interval context; first completed lower/upper reversal only; bottom must belong to original interval; next descent stops search',original_grades_unchanged=True")
s=s.replace("z['frame_numbers']<=d['last_frame'])","z['frame_numbers']<=d['last_frame']+round(.4*d['fps']))")
s=s.replace("(t<d['end'])","(t<d['end']+.4)")
s=s.replace("f=geom.extract(seq,rx,base);g=next", "original_end=d['last_frame'] if r['id'].startswith(('return_squat_','reviewed_return_squat_')) else d['end']*float(Fraction(inv[sid]['fps']));base=base if base and frames[base['bottom_index']]<=original_end else None;f=geom.extract(seq,rx,base);g=next")
s=s.replace("bank='standing_support_extension_v1'","bank='standing_support_context_v1'").replace("feature_version=geom.VERSION","feature_version='standing-support-bounded-context-v1'")
s=s.replace("reference_before_attempt=True,return_source_frame", "reference_before_attempt=True,original_attempt_end_frame=float(original_end),context_seconds=.4,context_end_frame=int(frames[-1]),return_source_frame")
# The saved original is the immediately preceding, fully frozen Test file.
s=s.replace("assert not (O/'MODEL_SELECTION_PROTOCOL.frozen.json').exists();", "assert not (O/'MODEL_SELECTION_PROTOCOL.frozen.json').exists();")
assert "standing_support_context_v1" in s and "round(.4*d['fps'])" in s
exec(compile(s,str(Path(__file__)), 'exec'))
