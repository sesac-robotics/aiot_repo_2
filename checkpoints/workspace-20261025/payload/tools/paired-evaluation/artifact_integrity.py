"""Compare persisted model values, ignoring undefined NumPy record padding bytes."""
import numpy as np

def equal_state(a,b,seen=None):
 if type(a) is not type(b):return False
 if seen is None:seen=set()
 pair=(id(a),id(b))
 if pair in seen:return True
 seen.add(pair)
 if isinstance(a,np.ndarray):
  if a.dtype!=b.dtype or a.shape!=b.shape:return False
  if a.dtype.names:return all(equal_state(a[n],b[n],seen) for n in a.dtype.names)
  return bool(np.array_equal(a,b,equal_nan=True)) if a.dtype.kind in 'fc' else bool(np.array_equal(a,b))
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k],seen) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_state(x,y,seen) for x,y in zip(a,b))
 if isinstance(a,(float,np.floating)) and np.isnan(a):return bool(np.isnan(b))
 if a is None or isinstance(a,(str,int,float,bool,np.generic)):return bool(a==b)
 if hasattr(a,'__dict__'):return equal_state(a.__dict__,b.__dict__,seen)
 if hasattr(a,'__getstate__'):return equal_state(a.__getstate__(),b.__getstate__(),seen)
 raise TypeError('Unsupported persisted value '+str(type(a)))
