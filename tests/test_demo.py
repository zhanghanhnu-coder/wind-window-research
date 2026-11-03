import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
from core import truth,probability,snapshot
class Contracts(unittest.TestCase):
 def test_known_failure_over_missing(self):self.assertEqual(truth([None,11]),0)
 def test_unknown_not_safe(self):self.assertIsNone(truth([None,8]))
 def test_complete_success(self):self.assertEqual(truth([8]*6),1)
 def test_same_marginals_different_joint(self):
  self.assertEqual(probability([[8,8],[12,12]],duration=2),.5)
  self.assertEqual(probability([[8,12],[12,8]],duration=2),0)
 def test_resource_cap(self):self.assertEqual(snapshot(0,crews=0)['selected'],[])
 def test_synthetic_determinism(self):self.assertEqual(snapshot(),snapshot())
