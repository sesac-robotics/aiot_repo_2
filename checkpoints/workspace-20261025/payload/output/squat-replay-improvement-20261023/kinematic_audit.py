import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
p=Path('output/live-webcam-20261023/sessions/1792727328056605892');z=np.load(p/'pose.npz');x=z['image_xy'];t=(z['frame_numbers']-z['frame_numbers'][0])/10
h=x[:,[23,24]].mean(1);s=x[:,[11,12]].mean(1);width=abs(x[:,12,0]-x[:,11,0]);tilt=np.degrees(np.arctan2(h[:,0]-s[:,0],h[:,1]-s[:,1]));a=np.isfinite(h).all(1)
for lo,hi in [(18.7,22),(26.8,29.2),(30.5,35.6),(37.1,47.9),(49.5,52.9)]:
 q=(t>=lo)&(t<=hi);v=x[q];tt=t[q];valid=np.isfinite(v[:,[11,12,23,24]]).all((1,2));hh=v[:,[23,24]].mean(1);ss=v[:,[11,12]].mean(1);ang=np.degrees(np.arctan2(hh[:,0]-ss[:,0],hh[:,1]-ss[:,1]));sl=np.degrees(np.arctan2(v[:,12,1]-v[:,11,1],abs(v[:,12,0]-v[:,11,0])));print('\n',lo,hi,'tilt',np.percentile(ang[valid],[0,10,50,90,100]),'shoulder',np.percentile(sl[valid],[0,10,50,90,100]))
 for n in [29,30,31,32]:print(n,'coverage',np.isfinite(v[:,n]).all(1).mean())
 print('hip_y',[(round(float(tt[i]),1),round(float(hh[i,1]),1)) for i in range(0,len(tt),3)])
