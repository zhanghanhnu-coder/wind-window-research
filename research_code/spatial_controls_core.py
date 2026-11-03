"""Fixed mean and permuted graphs, with matched initial learnable weights."""
import numpy as np
import torch
from crps_distribution_core import AtomNet

def control_graphs(adjacency):
 a=np.asarray(adjacency,float)
 if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<2 or not np.isfinite(a).all() or (a<0).any():raise ValueError('finite nonnegative square graph with at least two nodes required')
 n=len(a);permutation=np.random.default_rng(20241101).permutation(n)
 return {'mean':(np.ones((n,n))-np.eye(n))/(n-1),'random':a[permutation][:,permutation].copy()},permutation

def make_control_network(adjacency,scale,seed):
 torch.manual_seed(seed);return AtomNet(adjacency,scale,hidden=24,gated=True)
