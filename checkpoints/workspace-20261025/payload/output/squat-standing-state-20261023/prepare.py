"""Human-selected visible standing/non-standing frames, not fabricated failed reps."""
from pathlib import Path
import json,hashlib,cv2,numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];S=B/'output/squat-return-20261023'
# Frame numbers selected from chronological original-image review sheets.
# These are instantaneous state labels, never return/failure labels.
anchors={
'squat_front_m1_9_4':{'standing':[6,13,48,55,61,96,103,149,192,196,233,240,280,286,327,333,376,382,428,435,481,488,599,643,650],'not_standing':[20,27,34,41,68,75,82,89,116,122,129,136,160,167,173,179,203,209,216,222,246,253,260,267,293,300,306,313,340,347,354,362,390,398,405,412,442,450,458,466,494,501,508,515,522,528,535,548,558,568,578,588,615,622,629,636]},
'squat_front_m3_9_3':{'standing':[8,14,44,50,56,92,98,134,140,178,184,221,227,264,270,308,315,355,362,403,410,448,455,500,507,540],'not_standing':[26,32,67,73,79,109,115,152,158,164,189,195,202,208,232,238,244,251,275,282,288,295,321,328,334,341,368,375,382,389,415,422,428,435,462,470,477,484,517,522,528]},
'squat_front_m4_6_6':{'standing':[0,6,37,43,49,82,89,96,126,132,138,170,177,183,218,225,230,260,266,270,310,314,351,357,393,398,438,444,486,493,545,552,593,600],'not_standing':[12,18,24,31,55,62,68,75,100,107,113,120,144,150,157,163,190,197,204,211,236,242,248,254,277,284,290,296,321,327,334,340,363,369,375,381,404,411,418,425,451,458,465,472,559,566,572,579]},
'squat_front_w1_9_3':{'standing':[25,31,62,69,75,103,109,115,144,151,157,187,194,200,228,234,240,266,271,277,305,311,317,350,356,383,389,395,425,432,438,466,472,478,506,512,517],'not_standing':[38,44,50,56,79,85,91,97,120,126,132,138,162,168,174,181,204,210,216,222,243,249,254,260,281,287,293,299,321,327,332,338,360,366,372,377,401,406,412,419,443,449,454,460,485,488,494,499]},
'squat_front_w2_9_3':{'standing':[3,10,44,50,57,87,94,100,132,139,146,180,186,217,224,230,262,269,276,310,316,355,362,397,404,445,452,492,498,505,539,545],'not_standing':[16,23,30,37,62,68,74,81,106,112,119,126,150,156,162,168,191,198,204,210,236,242,249,256,280,286,292,299,322,329,336,343,369,376,383,390,417,424,431,438,464,471,478,485,509,515,521,527]},
'squat_diag_m1_6_7':{'standing':[5,12,58,65,103,109,148,155,191,196,238,245,299,310,358,365,403,410,446,452],'not_standing':[20,28,35,42,50,70,77,83,90,97,115,122,128,135,142,160,167,174,181,203,210,217,224,231,258,268,278,289,325,332,338,345,370,377,384,390,397,414,421,427,433,440,460,468,476,483,491,499,506,513,520,527,534,541,548,555,562,570,577,584,592,600,607,614,621,628,635,642,650,657]},
'squat_diag_m2_8_6':{'standing':[47,55,94,101,109,136,142,148,178,185,191,220,227,233,272,279,318,325,365,372,405,411,450,457,501,508,645,652,697],'not_standing':[62,70,78,86,113,119,124,130,153,159,166,172,196,202,208,214,238,245,252,259,266,284,291,298,305,312,331,338,344,351,358,376,382,388,393,399,416,423,430,437,444,464,471,478,486,493,513,519,526,532,538,545,551,557,564,571,578,585,592,599,606,614,622,630,638,658,664,671,678,684]},
'squat_diag_w1_9_3':{'standing':[17,24,62,69,67,102,108,145,153,190,188,223,229,264,270,301,307,340,346,381,387,421,429,464,470,506,512],'not_standing':[30,36,43,50,56,73,79,84,90,96,114,121,128,134,140,156,162,169,176,182,194,200,206,211,217,233,239,246,252,258,273,279,284,290,296,311,317,322,328,334,350,356,362,369,375,392,398,404,411,416,433,439,446,452,458,474,481,487,493,500]},
'squat_diag_w2_5_7':{'standing':[18,26,71,79,118,125,161,167,204,210,247,253,510,516],'not_standing':[33,41,48,56,64,84,91,98,105,112,130,136,142,148,154,172,178,184,191,197,215,221,228,234,240,257,263,268,274,280,286,292,295,300,306,312,317,322,328,331,337,342,348,354,359,365,368,374,379,384,390,396,401,404,410,416,422,428,433,439,442,448,454,459,465,470,476,480,486,492,498,504]}}
rows=[];receipts=[]
for name,groups in anchors.items():
 path=S/'cfrep'/f'{name}.mp4';z=np.load(S/f'{name}-heavy.npz');frames=z['frame_numbers'];person='cfrep_'+name.split('_')[2]
 # Baseline consists only of directly seen initial standing frames.
 ref_frames=groups['standing'][:2];ref_indices=[int(np.argmin(abs(frames-f))) for f in ref_frames];ref=z['image_xy'][ref_indices];np.savez_compressed(O/f'{name}-reference.npz',image_xy=ref)
 cap=cv2.VideoCapture(str(path));canvas=np.full((2*230,8*180,3),245,np.uint8)
 for state,chosen in groups.items():
  for f in sorted(set(chosen)):
   nearest=int(np.argmin(abs(frames-f)))
   rows.append(dict(id=f'{name}_f{int(frames[nearest])}',source=name,person=person,split='validation' if person=='cfrep_w2' else 'train',frame=int(frames[nearest]),original_review_frame=f,label=int(state=='standing'),label_kind='instantaneous_standing_state',reference_frames=ref_frames,path=str(path)))
 for row,state in enumerate(['standing','not_standing']):
  chosen=groups[state];selection=np.linspace(0,len(chosen)-1,8).round().astype(int)
  for col,i in enumerate(selection):
   f=chosen[i];cap.set(1,f);ok,im=cap.read()
   if not ok:continue
   cols=np.where((im.max(2)>20).mean(0)>.1)[0];im=im[:,cols[0]:cols[-1]+1] if len(cols) else im;k=min(180/im.shape[1],202/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*k),round(im.shape[0]*k)));x=col*180;y=row*230;canvas[y+25:y+25+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'{state} f{f}',(x+2,y+16),0,.36,(0,0,0),1)
 cap.release();cv2.imwrite(str(O/f'{name}-states.jpg'),canvas)
 # Conflicting nearest-frame labels are excluded instead of resolved by model output.
seen={};conflicts=set()
for r in rows:
 if r['id'] in seen and seen[r['id']]['label']!=r['label']:conflicts.add(r['id'])
 seen[r['id']]=r
rows=[r for k,r in seen.items() if k not in conflicts];data=dict(version='standing-state-labels-v1',target='현재 자세가 본인의 초기 기립 상태에 가까운지',expert_validated=False,not_return_failure_labels=True,rows=rows,conflicts_excluded=sorted(conflicts))
p=O/'labels.frozen.json';p.write_text(json.dumps(data,ensure_ascii=False,indent=2));(O/'labels.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n');print('labels',len(rows),'conflicts',len(conflicts),flush=True)
