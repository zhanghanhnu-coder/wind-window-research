"""Join only genuine pre-test raw forecasts to the fixed test-origin history."""
import numpy as np

def join_history(archive_q,archive_y,archive_ids,test_q,test_y,test_ids):
    aq=np.asarray(archive_q);ay=np.asarray(archive_y);ai=np.asarray(archive_ids);tq=np.asarray(test_q);ty=np.asarray(test_y);ti=np.asarray(test_ids)
    if aq.ndim!=4 or tq.ndim!=4 or aq.shape[1:]!=tq.shape[1:] or aq.shape[1]!=72 or aq.shape[-1]!=19 or ay.shape!=aq.shape[:-1] or ty.shape!=tq.shape[:-1]:raise ValueError('aligned archive/test 72-hour 19-atom forecasts and truth required')
    for ids,n in [(ai,len(aq)),(ti,len(tq))]:
        if ids.shape!=(n,) or n==0 or not np.issubdtype(ids.dtype,np.integer) or (np.diff(ids)<=0).any():raise ValueError('strictly chronological nonempty issue ids required')
    positions=np.searchsorted(ai,ti)
    if (positions>=len(ai)).any() or not np.array_equal(ai[positions],ti):raise ValueError('test issues must appear in archive')
    if not np.array_equal(aq[positions],tq,equal_nan=True) or not np.array_equal(ay[positions],ty,equal_nan=True):raise ValueError('archive must match original test forecast and truth version')
    pre=ai<ti[0]
    return dict(q=np.concatenate([aq[pre],tq]),y=np.concatenate([ay[pre],ty]),ids=np.concatenate([ai[pre],ti]),pretest_origins=int(pre.sum()))
