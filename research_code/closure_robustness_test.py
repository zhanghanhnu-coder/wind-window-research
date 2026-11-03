"""Small mathematical counterexamples before frozen-data analysis."""
import unittest
import itertools
import numpy as np
import pandas as pd
from closure_robustness import aggregation_weights, partial_terms, evaluate_bounds
from iteration3_matched_core import CONTRASTS

class MathematicalExamples(unittest.TestCase):
 def test_physical_event_weights(self):
  d=pd.DataFrame([dict(fold='f',issue=i,seed=s,task=0,start=0,event=e) for e,issues in [('a',[1,2]),('b',[3])] for i in issues for s in [1,2,3]])
  w=aggregation_weights(d,'physical_event');self.assertAlmostEqual(w.sum(),1)
  self.assertAlmostEqual(w[d.event=='a'].sum(),.5)
  self.assertAlmostEqual(w[d.issue==1].sum(),.25)
  self.assertAlmostEqual(aggregation_weights(d,'equal_origin')[d.event=='a'].sum(),2/3)
 def test_partial_only_fixes_unknown_bad(self):
  raw=np.array([[2.],[np.nan],[np.nan],[11.]],dtype=np.float64)
  bad=np.array([[True],[True],[False],[True]])
  result,filled=partial_terms(raw,bad)
  self.assertEqual(filled,1);self.assertEqual(result[0,0],2.);self.assertEqual(result[1,0],11.)
  self.assertTrue(np.isnan(result[2,0]));self.assertEqual(result[3,0],11.)
 def test_shared_and_endpoints_square_direct(self):
  raw=np.array([[0.],[np.nan],[np.nan],[np.nan]],dtype=np.float64)
  d=pd.DataFrame([dict(issue=0,start=0,node=0,duration=2,AI=.1,GI=.7,AC=.3,GC=.8),dict(issue=0,start=1,node=0,duration=2,AI=.6,GI=.3,AC=.8,GC=.2)])
  from iteration2_bounds_core import window_term
  terms=[window_term(raw,int(r.issue),int(r.start),int(r.start+r.duration),0) for r in d.itertuples()]
  w=np.array([.5,.5]);v=CONTRASTS['interaction'];res,err=evaluate_bounds(d,w,v,terms,raw)
  brute=[]
  for states in itertools.product([0,1],repeat=3):
   y=np.array([states[0]*states[1],states[1]*states[2]])
   brute.append(float(w@(((d[['AI','GI','AC','GC']].to_numpy()-y[:,None])**2)@v)))
  self.assertAlmostEqual(res['lower'],min(brute));self.assertAlmostEqual(res['upper'],max(brute));self.assertLess(err,1e-12)
  bad=np.zeros_like(raw,dtype=bool);bad[2,0]=True;partial,_=partial_terms(raw,bad)
  pt=[window_term(partial,int(r.issue),int(r.start),int(r.start+r.duration),0) for r in d.itertuples()]
  sub,_=evaluate_bounds(d,w,v,pt,partial)
  self.assertGreaterEqual(sub['lower']+1e-12,res['lower']);self.assertLessEqual(sub['upper']-1e-12,res['upper'])

if __name__=='__main__': unittest.main()
