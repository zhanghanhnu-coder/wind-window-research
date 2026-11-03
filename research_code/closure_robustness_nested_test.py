"""Additional endpoint check where independent window labels are infeasible."""
import unittest
import itertools
import numpy as np
import pandas as pd
from closure_robustness import evaluate_bounds
from iteration2_bounds_core import window_term
from iteration3_matched_core import CONTRASTS

class NestedWindow(unittest.TestCase):
 def test_nested_and_excludes_independent_window_extreme(self):
  raw=np.array([[0.],[np.nan],[np.nan]],dtype=np.float64)
  d=pd.DataFrame([dict(issue=0,start=0,node=0,duration=2,AI=.1,GI=.7,AC=.3,GC=.8),dict(issue=0,start=0,node=0,duration=1,AI=.3,GI=.6,AC=.2,GC=.8)])
  terms=[window_term(raw,0,0,int(t),0) for t in d.duration]
  p=d[['AI','GI','AC','GC']].to_numpy();w=np.array([.5,.5]);sign=np.array(CONTRASTS['interaction'])
  result,err=evaluate_bounds(d,w,sign,terms,raw)
  exact=[float(w@(((p-np.array([a*b,a])[:,None])**2)@sign)) for a,b in itertools.product([0,1],repeat=2)]
  coarse=[float(w@(((p-np.array([a,b])[:,None])**2)@sign)) for a,b in itertools.product([0,1],repeat=2)]
  self.assertAlmostEqual(result['lower'],min(exact));self.assertAlmostEqual(result['upper'],max(exact))
  self.assertGreater(result['lower'],min(coarse)+.05);self.assertLess(err,1e-12)

if __name__=='__main__':unittest.main()
