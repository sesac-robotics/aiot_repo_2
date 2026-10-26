"""Accept only the two equivalent unscored elbow-angle feature contracts."""
import copy,math
SOURCE_BANKS={'front','side_legacy_roi_v1'}
BANK='observed_arm_extension_shared_v1'
def pool_item(item):
 if item.get('item')!='return_extension' or item.get('bank') not in SOURCE_BANKS:
  raise ValueError('Only degree-angle pushup return measurements may pool')
 f=item.get('feature')
 if f is not None:
  if len(f)!=8 or not all(math.isfinite(v) for v in f):raise ValueError('Expected seven angle percentiles plus observation fraction')
  if any(v<0 or v>180 for v in f[:7]) or not 0<=f[7]<=1:raise ValueError('Invalid angle/visibility units')
  if any(a>b for a,b in zip(f[:6],f[1:7])):raise ValueError('Angle percentiles out of order')
 if item.get('eligible') and f is None:raise ValueError('Eligible measurement cannot be missing')
 out=copy.deepcopy(item);out.update(bank=BANK,feature_version='same_terminal_elbow_angle_degrees_quantiles_visibility_v1',source_feature_bank=item['bank'])
 return out
