import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;r=json.loads((O/'analysis.json').read_text());z=np.load(Path(r['session'])/'pose.npz');ticks=z['frame_numbers'];raw=z['image_xy'];lookup={int(t):p for t,p in zip(ticks,raw)}
fig,axes=plt.subplots(2,1,figsize=(11,5.4),sharex=True,layout='constrained');colors=['#126b80']*7;colors[5]='#b54d22';maximum_ratio=1.4
for s,c in zip(r['segments'],colors):
 ts=np.arange(s['start_tick'],s['end_tick']+1);x=np.array([lookup[int(t)]for t in ts]);sec=(ts-ticks[0])/10;hip=x[:,[23,24]].mean(1);sh=x[:,[11,12]].mean(1);scale=np.median(np.linalg.norm(hip-sh,axis=1));depth=(hip[:,1]-np.median(hip[:3,1]))/scale;space=np.abs(x[:,26,0]-x[:,25,0]);ratio=space/np.median(space[:3]);maximum_ratio=max(maximum_ratio,float(np.max(ratio)));axes[0].plot(sec,depth,color=c,lw=2);axes[1].plot(sec,ratio,color=c,lw=2);axes[0].annotate(str(s['number']),(sec[np.argmax(depth)],max(depth)),xytext=(0,6),textcoords='offset points',ha='center',weight='bold',color=c)
for ax in axes:
 ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].set(ylabel='Observed hip drop / torso length',title='Saved squat test: 7 observed descent-return cycles')
axes[1].set(ylabel='Knee spacing / segment start',xlabel='Seconds from measurement start');axes[1].axhline(1,color='#666',ls='--',lw=1);axes[1].annotate('Rep 6: narrowed observed knee spacing',xy=(33,.76),xytext=(17,.7),arrowprops={'arrowstyle':'->','color':'#b54d22'},color='#b54d22');axes[1].set_ylim(.65,maximum_ratio+.05)
fig.savefig(O/'observed-cycles.png',dpi=160)
