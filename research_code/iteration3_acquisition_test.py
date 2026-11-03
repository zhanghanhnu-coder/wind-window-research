"""Exhaustive tests of query worst-case residual width, independent enumeration."""
import itertools, unittest
from iteration3_acquisition_core import worst_width

def exhaustive(c, terms, query):
    hours=sorted(set().union(*map(set,terms))) if terms else []
    vals={}
    for bits in itertools.product([0,1],repeat=len(hours)):
        z=dict(zip(hours,bits));key=tuple(z.get(h,0) for h in query if h in z)
        value=c+sum(v*all(z[h] for h in ks) for ks,v in terms.items())
        vals.setdefault(key,[]).append(value)
    return max(max(v)-min(v) for v in vals.values())

class QueryTests(unittest.TestCase):
    def test_exhaustive(self):
        import numpy as np
        rng=np.random.default_rng(317)
        for trial in range(24):
            hh=[(h,0) for h in range(5)];terms={}
            for _ in range(8):
                ks=tuple(hh[i] for i in range(5) if rng.integers(2))
                if ks:terms[ks]=int(rng.integers(-7,8))
            previous=None
            for k in range(6):
                q=hh[:k];r=worst_width(3,terms,q)
                self.assertEqual(r['certificate'],'PASS')
                self.assertAlmostEqual(r['worst_remaining_width'],exhaustive(3,terms,q),places=7)
                if previous is not None:self.assertLessEqual(r['worst_remaining_width'],previous+1e-7)
                previous=r['worst_remaining_width']
            self.assertAlmostEqual(previous,0,places=7)
    def test_irrelevant(self):
        terms={((0,0),(1,0)):3,((1,0),):-2}
        self.assertAlmostEqual(worst_width(9,terms,[])['worst_remaining_width'],worst_width(9,terms,[(99,0)])['worst_remaining_width'])

if __name__=='__main__':unittest.main()
