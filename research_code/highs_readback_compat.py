"""Process-local SciPy 1.15.2 readback optimization, without solver changes.

The installed wrapper reads a pybind vector property inside its column loop.
That property returns a new list each time. Cache the two completed-result
vectors once. Refuse any unreviewed upstream source; never edit site-packages.
"""
import hashlib,inspect,importlib
import numpy as np
import scipy
from scipy.optimize._highspy._core import HighsBasisStatus
SOURCE_SHA256='9dca9bc8bdcfe602d7f47adc3d927aecd05ee8cfccd28c96522f54c48ae3f0fc'
OLD='''    marg_bnds = np.zeros((2, numcol))
    for ii in range(numcol):
        if basis.col_status[ii] == _h.HighsBasisStatus.kLower:
            marg_bnds[0, ii] = solution.col_dual[ii]
        elif basis.col_status[ii] == _h.HighsBasisStatus.kUpper:
            marg_bnds[1, ii] = solution.col_dual[ii]
'''
NEW='''    marg_bnds = _cached_bound_marginals(basis, solution, numcol)
'''

def cached_bound_marginals(basis,solution,numcol):
 statuses=basis.col_status;dual=solution.col_dual;out=np.zeros((2,numcol))
 for ii in range(numcol):
  if statuses[ii]==HighsBasisStatus.kLower:out[0,ii]=dual[ii]
  elif statuses[ii]==HighsBasisStatus.kUpper:out[1,ii]=dual[ii]
 return out

def build_wrapper(source=None):
 upstream=importlib.import_module('scipy.optimize._highspy._highs_wrapper')._highs_wrapper
 source=inspect.getsource(upstream) if source is None else source;digest=hashlib.sha256(source.encode()).hexdigest()
 if scipy.__version__!='1.15.2' or digest!=SOURCE_SHA256 or source.count(OLD)!=1:raise RuntimeError('unreviewed SciPy wrapper; compatibility layer refuses to apply')
 patched=source.replace(OLD,NEW);namespace=dict(upstream.__globals__);namespace['_cached_bound_marginals']=cached_bound_marginals;exec(compile(patched,'<scipy_1_15_2_cached_result_readback>','exec'),namespace);fn=namespace['_highs_wrapper'];meta=dict(scipy_version=scipy.__version__,upstream_function_sha256=digest,patched_function_sha256=hashlib.sha256(patched.encode()).hexdigest(),change='cache two immutable result arrays once; no solver change',scope='current process only; installed SciPy files unchanged');fn._readback_compat_metadata=meta;return fn,meta

def install():
 module=importlib.import_module('scipy.optimize._milp');existing=getattr(module._highs_wrapper,'_readback_compat_metadata',None)
 if existing:return dict(existing)
 upstream=importlib.import_module('scipy.optimize._highspy._highs_wrapper')._highs_wrapper
 if module._highs_wrapper is not upstream:raise RuntimeError('another MILP wrapper is already installed')
 fn,meta=build_wrapper();module._highs_wrapper=fn;return dict(meta)
