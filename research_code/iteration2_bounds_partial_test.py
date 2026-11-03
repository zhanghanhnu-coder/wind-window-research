"""Partial observed failure evidence must never create success evidence."""
import unittest
import numpy as np
from iteration2_bounds_partial import classify_window

class PartialEvidenceTests(unittest.TestCase):
 def test_observed_bad_partial_hour_fixes_failure(self):
  raw=np.array([[0.],[np.nan],[np.nan]],dtype=np.float64)
  bad=np.array([[False],[True],[False]])
  self.assertIs(classify_window(raw,bad,0,0,2,0),False)
 def test_partial_safe_remains_unknown(self):
  raw=np.array([[0.],[np.nan],[10.]],dtype=np.float64);bad=np.zeros_like(raw,dtype=bool)
  self.assertEqual(classify_window(raw,bad,0,0,2,0),((1,0),))
 def test_known_raw_failure_still_dominates(self):
  raw=np.array([[0.],[10.1],[np.nan]],dtype=np.float64);bad=np.zeros_like(raw,dtype=bool)
  self.assertIs(classify_window(raw,bad,0,0,2,0),False)

if __name__=='__main__':unittest.main()
