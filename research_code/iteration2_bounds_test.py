"""Independent exhaustive references for shared physical-hour cost bounds."""
import itertools
import unittest
import numpy as np
from iteration2_bounds_core import solve_bounds, window_term

def exhaustive(constant, terms):
    keys=sorted(set().union(*(set(k) for k in terms)))
    vals=[]
    for bits in itertools.product([0,1], repeat=len(keys)):
        assignment=dict(zip(keys,bits))
        vals.append(constant+sum(c*all(assignment[k] for k in ks) for ks,c in terms.items()))
    return min(vals),max(vals)

class SharedBoundsTests(unittest.TestCase):
    def check(self,constant,terms):
        result=solve_bounds(constant,terms)
        np.testing.assert_allclose([result['lower'],result['upper']],exhaustive(constant,terms),atol=1e-7)
        for side in ['lower','upper']:
            self.assertLessEqual(result[side+'_certificate']['absolute_gap'],1e-7)
    def test_overlapping_windows(self):
        self.check(7,{((1,0),(2,0)):-12,((2,0),(3,0)):10,((1,0),):3})
    def test_identical_window_cancellation(self):
        self.check(19,{((1,0),(2,0)):0})
    def test_cross_seed_shared_weather(self):
        # Two seeds: -12/2 and +8/2 on the same physical hour.
        self.check(1,{((1,0),):-2})
        self.assertEqual(solve_bounds(1,{((1,0),):-2})['lower'],-1)
    def test_known_failure_precedes_unknown(self):
        raw=np.array([[np.nan],[10.0000000001],[9.]],dtype=np.float64)
        self.assertEqual(window_term(raw,0,0,2,0),False)
    def test_all_known_and_threshold_equality(self):
        raw=np.array([[0.],[10.],[9.]],dtype=np.float64)
        self.assertEqual(window_term(raw,0,0,2,0),True)
        self.check(8,{})
    def test_unknown_absolute_identity(self):
        raw=np.full((9,2),np.nan,dtype=np.float64)
        self.assertEqual(window_term(raw,3,0,2,1),((4,1),(5,1)))
    def test_random_small_exhaustive(self):
        rng=np.random.default_rng(142)
        for _ in range(30):
            terms={}
            for j in range(8):
                key=tuple((int(i),0) for i in np.flatnonzero(rng.random(6)<.4))
                if key:terms[key]=terms.get(key,0)+int(rng.integers(-9,10))
            self.check(3,terms)

if __name__=='__main__':unittest.main()
