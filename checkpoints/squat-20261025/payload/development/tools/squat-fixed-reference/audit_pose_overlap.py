"""Original-pixel/pose alignment audit; no fitting, new thresholds or labels."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

B=Path(__file__).resolve().parents[2]
P=B/'output/squat-fixed-reference-combined-v3-20261024'
R=B/'output/squat-pose-alignment-v5-20261024'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    R.mkdir(exist_ok=True)
    protocol=R/'protocol.frozen.json'
    assert not protocol.exists()
    protocol.write_text(json.dumps(dict(scope='previously_exposed_development_pose_QA_only',
        selection=['r05_nearest_normal','r09_missed_error','r12_probability_order_conflict'],
        model_sha256=sha(P/'new-head.joblib'),fit=False,threshold_selection=False,
        independent_final_Test_opened=False,grade_changes=False),indent=2))
    meta=json.loads((P/'feature-provenance.json').read_text())
    results=[]
    for suffix in ['r05','r09','r12']:
        row=next(x for x in meta if x['id'].endswith('diag_w2_5_7_'+suffix))
        z=np.load(row['cache']);xy=z['image_xy'];fn=z['frame_numbers']
        assert sha(row['cache'])==row['cache_sha256']
        f=row['return_source_frame'];frames=[0,f,f+3,f+6]
        indices=[int(np.where(fn==a)[0][0]) for a in frames]
        assert len(set(indices))==4
        # One shared source-coordinate crop and scaling for all comparisons;
        # each raw frame and its overlay have identical pixels and transforms.
        points=xy[indices][:,[11,12,23,24,25,26,27,28]].reshape(-1,2)
        finite=points[np.isfinite(points).all(1)]
        left=max(0,int(np.floor(finite[:,0].min()-90)))
        right=min(int(z['width']),int(np.ceil(finite[:,0].max()+90)))
        top=max(0,int(np.floor(finite[:,1].min()-100)))
        bottom=min(int(z['height']),int(np.ceil(finite[:,1].max()+50)))
        cap=cv2.VideoCapture(row['source_path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        canvas=Image.new('RGB',(1600,1100),'white');draw=ImageDraw.Draw(canvas)
        diagnostics=[]
        for col,(fr,k) in enumerate(zip(frames,indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,a=cap.read();assert ok
            assert a.shape[:2]==(int(z['height']),int(z['width']))
            raw=Image.fromarray(cv2.cvtColor(a[top:bottom,left:right],cv2.COLOR_BGR2RGB))
            for rr in [0,1]:
                im=raw.copy();pen=ImageDraw.Draw(im)
                if rr:
                    for chain,color in [([11,23,25,27],'red'),([12,24,26,28],'cyan')]:
                        for j in chain:
                            x,y=xy[k,j]-[left,top]
                            if np.isfinite([x,y]).all():
                                pen.ellipse((x-3,y-3,x+3,y+3),fill=color)
                                pen.text((x+4,y),str(j),fill=color)
                        for j,jj in zip(chain,chain[1:]):
                            p,q=xy[k,[j,jj]]-[left,top]
                            if np.isfinite([p,q]).all():pen.line([tuple(p),tuple(q)],fill=color,width=2)
                im.thumbnail((390,510))
                canvas.paste(im,(col*400+(400-im.width)//2,rr*550+30))
                draw.text((col*400+4,rr*550+5),f'{suffix} f{fr} '+('original' if rr==0 else 'cached estimate; not truth'),fill='black')
            diagnostics.append(dict(frame=fr,xy_required=xy[k,[11,12,23,24,25,26,27,28]].tolist()))
        cap.release();page=R/(suffix+'-original-pose.jpg');canvas.save(page,quality=97)
        results.append(dict(id=row['id'],source_sha256=sha(row['source_path']),
            cache_sha256=sha(row['cache']),frames=frames,source_width=int(z['width']),
            source_height=int(z['height']),crop=[left,top,right,bottom],
            sheet=str(page),sheet_sha256=sha(page),samples=diagnostics,
            parent_grade=row['grade'],parent_grade_changed=False))
    (R/'pose-evidence.frozen.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print('Prepared original/pose comparisons:',len(results))

if __name__=='__main__':main()
