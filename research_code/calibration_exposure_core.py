"""Map continuous six-hour events to the actual marginal calibration group."""
import numpy as np

def window_fallback(blocks,fallback,count):
    b=np.asarray(blocks);f=np.asarray(fallback)
    if b.ndim!=1 or not np.issubdtype(b.dtype,np.integer) or f.ndim!=2 or f.shape[1]!=4 or f.dtype!=bool or count not in [3,17] or (b<0).any() or (b>=len(f)).any():raise ValueError('valid week ids, four lead-group flags and 3/17 event count required')
    return np.repeat(f[b][:,[0,2,3,3]],count,axis=1)
