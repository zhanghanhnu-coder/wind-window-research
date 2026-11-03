"""Small causal input masking and shared-label bound checks."""
import unittest
import numpy as np
from closure_outage_core import outage_history, paired_bounds

class OutageTests(unittest.TestCase):
    def test_causal_future_poison(self):
        x=np.arange(230*2*3,dtype=float).reshape(230,2,3);x[:,:,2]=1
        a=outage_history(x,np.array([180]),6)
        x[181:]=np.nan
        np.testing.assert_array_equal(a,outage_history(x,np.array([180]),6))
    def test_zero_preserves_and_masks_all_channels(self):
        x=np.ones((230,2,3),np.float32);x[175,0]=0
        a=outage_history(x,np.array([180]),0)
        np.testing.assert_array_equal(a[0],x[13:181])
        b=outage_history(x,np.array([180]),6)
        self.assertTrue((b[:,-6:]==0).all());np.testing.assert_array_equal(a[:,:-6],b[:,:-6])
    def test_reject_bad_length(self):
        with self.assertRaises(ValueError):outage_history(np.ones((230,2,3)),np.array([180]),25)
    def test_shared_unknown_label_cancellation(self):
        a=np.array([[[.9,.1]]]);b=np.array([[[.1,.9]]]);y=np.array([[np.nan,np.nan]]);ix=np.zeros((1,2),int)
        lo,hi=paired_bounds(a,b,y,ix);self.assertAlmostEqual(lo,0);self.assertAlmostEqual(hi,0)
    def test_seed_losses_not_ensemble(self):
        a=np.array([[[0.]],[[1.]]]);b=np.array([[[.5]],[[.5]]]);y=np.array([[1.]]);ix=np.zeros((1,1),int)
        lo,hi=paired_bounds(a,b,y,ix);self.assertAlmostEqual(lo,.25);self.assertAlmostEqual(hi,.25)

if __name__=='__main__':unittest.main()
