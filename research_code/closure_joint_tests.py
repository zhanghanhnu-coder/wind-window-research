"""Real RED/GREEN mutation checks for the frozen H2 mathematical invariants."""
import itertools,json
import numpy as np
from closure_joint_core import intervene,probabilities,direct_probabilities,materialize
from missing_window_bounds_core import coefficients,sharp_bounds
from precision_truth_core import window_labels
from closure_joint import OUT,js

def main():
    log=[]
    def check(name,fn,red=False):
        try:fn()
        except (AssertionError,ValueError) as e:
            assert red,(name,str(e));log.append(dict(name=name,phase='RED',expected_failure=type(e).__name__));return
        assert not red,name+' failed to detect mutation'
        log.append(dict(name=name,phase='GREEN',passed=True))
    rng=np.random.default_rng(17);q=np.sort(rng.uniform(0,20,(2,72,17,19)),axis=-1)
    ranks=np.stack([rng.permutation(57) for _ in range(72*17)],1).reshape(57,72,17)
    def marginal(x):np.testing.assert_array_equal(np.sort(x,axis=0),np.sort(ranks,axis=0))
    bad=ranks.copy();bad[0,0,0]=bad[1,0,0]
    check('marginal duplicate-index mutation',lambda:marginal(bad),True)
    for kind in ['time','space','both','global']:
        rr,ix=intervene(ranks,kind,23)
        check(kind+' exact index multiset',lambda:marginal(rr))
        check(kind+' actual values',lambda:np.testing.assert_array_equal(np.sort(materialize(q,rr),1),np.sort(materialize(q,ranks),1)))
        a,b=probabilities(q,rr,10);da,db=direct_probabilities(q[0],rr,10)
        check(kind+' direct probability',lambda:np.testing.assert_array_equal(a[0],da))
        check(kind+' direct group probability',lambda:np.testing.assert_array_equal(b[0],db))
        if kind=='global':
            ba,bg=probabilities(q,ranks,10)
            check('global probability negative control',lambda:np.testing.assert_array_equal(a,ba))
    pa=np.array([[[0.]],[[1.]]]);y=np.array([[0.]])
    check('Brier probability-averaging mutation',lambda:np.testing.assert_equal(float((pa.mean(0)**2).mean()),.5),True)
    check('Brier score-then-average',lambda:np.testing.assert_equal(float((pa**2).mean()),.5))
    raw=np.full((80,17),10.,dtype=np.float64);raw[1,0]=np.nextafter(10.,np.inf)
    check('threshold float32 mutation',lambda:np.testing.assert_equal(window_labels(raw.astype(np.float32).astype(np.float64),np.array([0]))[0,0],0),True)
    check('threshold float64 strict exceedance',lambda:np.testing.assert_equal(window_labels(raw,np.array([0]))[0,0],0))
    a=np.array([[[.2,.8]],[[.4,.6]]]);b=np.array([[[.4,.4]],[[.5,.5]]]);yy=np.array([[np.nan,np.nan]]);ix=np.array([[0,0]])
    c,d=coefficients(a,b);bounds=sharp_bounds(c,d,yy,ix)
    losses=[np.mean((a-label)**2-(b-label)**2) for label in (0,1)]
    check('shared-label paired bound enumerated min',lambda:np.testing.assert_allclose(bounds['lower'],min(losses),atol=1e-15))
    check('shared-label paired bound enumerated max',lambda:np.testing.assert_allclose(bounds['upper'],max(losses),atol=1e-15))
    OUT.mkdir(parents=True,exist_ok=True);js(OUT/'math_test_log.json',log)
    print(json.dumps(log,indent=2));print('ALL',len(log),'RED/GREEN checks passed')

if __name__=='__main__':main()
