"""New, isolated follow-up: frozen B/W28 forecasts crossed with executable rules."""
import numpy as np
from causal_execution_core import dispatch_now
from lookahead_execution_core import plan_future
from two_stage_core import audit_schedule, booking_cost

def selected_positions(n):
    """First eleven of round33's twelve; last lacks 66h of cached updates."""
    if n < 12:
        raise ValueError('at least twelve candidate forecasts required')
    return np.floor(np.linspace(0, n-1, 12)).astype(int)[:11]

def probability_grid(scenarios, config, threshold=10.):
    sc = np.asarray(scenarios, float)
    if sc.ndim != 3 or sc.shape[1] != 72 or not np.isfinite(sc).all():
        raise ValueError('complete future scenarios required')
    result = np.full((12, len(config['tasks'])), np.nan)
    for i, start in enumerate(range(0, 72, 6)):
        for j, task in enumerate(config['tasks']):
            end = start + task['duration']
            if end <= 72:
                result[i,j] = (sc[:,start:end,task['node']] <= threshold).all(1).mean()
    return result

def execute(problem, x, grids, quality, executor):
    """Only current forecast and prior irrevocable commitments enter each action."""
    grids = np.asarray(grids, float)
    quality = np.asarray(quality, bool)
    if grids.shape != (12,12,len(problem.config['tasks'])) or quality.shape != (12,):
        raise ValueError('unaligned update forecasts')
    if executor not in ['myopic', 'lookahead72']:
        raise ValueError('unknown executor')
    attempts, steps = [], []
    for k, now in enumerate(range(0,72,6)):
        if not quality[k]:
            decision = dict(selected=[], planned=[], quality_ok=False)
        elif executor == 'myopic':
            decision = dispatch_now(problem, x, attempts, now, grids[k,0], 'economic')
            decision['quality_ok'] = True
        else:
            decision = plan_future(problem, x, attempts, now, grids[k], 72, 'economic')
            decision['quality_ok'] = True
        attempts.extend(decision['selected'])
        steps.append(dict(now=now, **decision))
    audit_schedule(problem,x,np.ones(len(problem.choices),bool),attempts)
    return dict(x=list(x), executor=executor, attempts=attempts, steps=steps)

def task_ledger(problem, trajectory, truth, threshold=10.):
    """Independent accounting from raw64 observations; no production evaluation call."""
    y = np.asarray(truth)
    if y.dtype != np.float64 or y.shape[0] != problem.H:
        raise ValueError('original float64 hourly truth required')
    attempts = trajectory['attempts']
    audit_schedule(problem,trajectory['x'],np.ones(len(problem.choices),bool),attempts)
    bytask = {problem.choices[i]['task']:i for i in attempts}
    rows=[]
    for j,t in enumerate(problem.config['tasks']):
        choice=bytask.get(j)
        row=dict(task=j,id=t['id'],node=t['node'],duration=t['duration'],
                 penalty=float(t['penalty']),choice=choice,start=None,end=None,
                 operation_cost=0.,status='unattempted',penalty_lower=float(t['penalty']),
                 penalty_upper=float(t['penalty']),known_hours=0)
        if choice is not None:
            o=problem.choices[choice];z=y[o['start']:o['end'],o['node']]
            # Explicit comparisons on unrounded float64, including >10 by <1e-6.
            status='failed' if np.any(z[np.isfinite(z)]>threshold) else 'successful' if np.isfinite(z).all() else 'unknown'
            cost=(problem.config['boat_hourly'][o['boat']]+problem.config['crew_hourly'][o['crew']])*t['duration']+problem.config['late_fraction']*t['penalty']*o['start']/problem.H
            row.update(start=o['start'],end=o['end'],operation_cost=float(cost),status=status,
                       penalty_lower=float(t['penalty']) if status=='failed' else 0.,
                       penalty_upper=0. if status=='successful' else float(t['penalty']),
                       known_hours=int(np.isfinite(z).sum()))
        rows.append(row)
    book=booking_cost(problem,trajectory['x'])
    op=sum(r['operation_cost'] for r in rows)
    counts={s:sum(r['status']==s for r in rows) for s in ['successful','failed','unknown','unattempted']}
    summary=dict(booking_cost=book,operating_cost=op,cost_lower=book+op+sum(r['penalty_lower'] for r in rows),cost_upper=book+op+sum(r['penalty_upper'] for r in rows),**counts)
    return rows,summary

def difference_bounds(a,b):
    """Conservative shared-case outer bound, never a confidence interval."""
    return a['cost_lower']-b['cost_upper'],a['cost_upper']-b['cost_lower']
