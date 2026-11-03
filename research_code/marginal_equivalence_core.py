"""Correct finite-precision equality contract when only scenario order changes."""
import numpy as np

def check_equivalence(current,previous,keys,permuted):
 maximum=0.
 for key in keys:
  a=np.asarray(current[key]);b=np.asarray(previous[key])
  if key=='crps' and permuted:
   np.testing.assert_allclose(a,b,rtol=0,atol=3e-12,equal_nan=True);valid=np.isfinite(a)&np.isfinite(b);maximum=float(np.max(abs(a[valid]-b[valid]))) if valid.any() else 0.
  else:np.testing.assert_array_equal(a,b)
 return maximum
