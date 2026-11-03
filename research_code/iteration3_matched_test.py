"""Synthetic validity tests; observed RED before implementation."""
import unittest
import numpy as np
import pandas as pd
from iteration3_matched_core import affine, matched, weights, CONTRASTS

class Validity(unittest.TestCase):
 def test_four_cell_algebra(self):
  p=np.array([.1,.7,.3,.8])
  for v in CONTRASTS.values():
   c,b=affine(p,v)
   for y in [0.,1.]:self.assertAlmostEqual(c+b*y,float(np.dot(v,(p-y)**2)))
 def test_interaction_sign(self):
  # Both initial losses .25; Graph improves to zero, Analog remains .25.
  p=np.array([.5,.5,.5,1.]);c,b=affine(p,CONTRASTS['interaction'])
  self.assertAlmostEqual(c+b,.25)
 def test_square_before_seed_mean(self):
  p=np.array([0.,1.,.5]);loss=np.mean((p-1)**2)
  self.assertNotAlmostEqual(loss,(p.mean()-1)**2)
  self.assertAlmostEqual(loss,5/12)
 def test_start0_zero(self):
  p=np.array([.2,.7,.2,.7]);c,b=affine(p,CONTRASTS['interaction'])
  self.assertAlmostEqual(c,0);self.assertAlmostEqual(b,0)
 def test_equal_origin_window_weights(self):
  d=pd.DataFrame([dict(fold='f',issue=i,seed=s,task=t,start=0) for i,n in [(1,1),(2,3)] for s in [1,2,3] for t in range(n)])
  w=weights(d);self.assertAlmostEqual(w.sum(),1)
  for i in [1,2]:self.assertAlmostEqual(w[d.issue==i].sum(),.5)
 def test_matched_current_queue(self):
  rows=[]
  for pred in ['AnalogFixed','GraphFixed']:
   for stage in ['initial_candidate','current_candidate']:
    for start in ([0,6] if stage=='initial_candidate' else [0]):
     rows.append(dict(fold='f',issue=1,seed=1,task=0,start=start,prediction=pred,stage=stage,probability=.5,label=1,complete=True))
  m=matched(pd.DataFrame(rows));self.assertEqual(len(m),1);self.assertEqual(m.start.iloc[0],0)

if __name__=='__main__':unittest.main()
