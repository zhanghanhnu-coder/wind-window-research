"""Preserve raw atom values while using float64 forecast arithmetic for scoring."""
import numpy as np
from crps_distribution_core import couple_atoms

def scoring_coupling(q,ranks):
 return couple_atoms(q,ranks).astype(np.float64,copy=False)
