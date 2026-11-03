"""Frozen-action ex post repricing and shared-weather service accounting."""
from collections import defaultdict
from itertools import product
import numpy as np
from iteration2_bounds_core import window_term, solve_bounds


def expression(e, case, raw, metric):
    """Penalty loss or unit-weight success expression on absolute physical hours."""
    constant = 0.
    terms = defaultdict(float)
    for t in e['tasks']:
        penalty = float(t['penalty'])
        key = False if t['choice'] is None else window_term(raw, case['issue'], t['start'], t['end'], t['node'])
        expected = 'unattempted' if t['choice'] is None else 'failed' if key is False else 'successful' if key is True else 'unknown'
        assert t['status'] == expected
        weight = penalty if metric == 'penalty' else 1.
        if metric == 'penalty':
            constant += weight
        sign = -1 if metric == 'penalty' else 1
        if key is True:
            constant += sign * weight
        elif key is not False:
            terms[key] += sign * weight
    return constant, terms


def direct(e, case, raw, witness):
    """Read task/hour ledger without using symbolic window classification."""
    penalty = success = hours = 0.
    for t in e['tasks']:
        good = t['choice'] is not None
        if good:
            hours += t['end'] - t['start']
            for h in range(case['issue'] + t['start'] + 1, case['issue'] + t['end'] + 1):
                v = raw[h, t['node']]
                if not (bool(v <= 10.) if np.isfinite(v) else bool(witness.get((h, t['node']), 0))):
                    good = False
        success += int(good)
        penalty += float(t['penalty']) * (not good)
    operation = sum(t['operation_cost'] for t in e['tasks'])
    return dict(penalty=penalty, success=success, operation=operation,
                constant=float(e['metrics']['booking_cost']) + operation, hours=hours)


def affine_direction(constant, lower_slope, upper_slope):
    """Exact root representation and open-interval signs for lambda >= 0 only."""
    def direction(x):
        lo, hi = constant + x * lower_slope, constant + x * upper_slope
        if hi < -1e-9: return 'candidate_strictly_lower_cost'
        if lo > 1e-9: return 'candidate_strictly_higher_cost'
        if abs(lo) < 1e-9 and abs(hi) < 1e-9: return 'equal'
        return 'not_strictly_identified'
    roots = sorted(set(-constant / v for v in [lower_slope, upper_slope] if v != 0 and -constant / v > 0))
    edges = [0.] + roots + [None]
    intervals = []
    for left, right in zip(edges[:-1], edges[1:]):
        mid = left + max(1., left) if right is None else (left + right) / 2
        intervals.append(dict(lambda_left=left, lambda_right=right, endpoints='open', direction=direction(mid)))
    return dict(domain='lambda >= 0', constant=constant, lower_slope=lower_slope,
                upper_slope=upper_slope, positive_roots=roots, intervals=intervals,
                points=[dict(lambda_value=x, lower=constant+x*lower_slope,
                             upper=constant+x*upper_slope, direction=direction(x)) for x in [0.] + roots])


def exhaustive_tests():
    rng = np.random.default_rng(1102)
    errors = []
    for index in range(35):
        hours = [(i, 0) for i in range(5)]
        terms = defaultdict(float)
        for _ in range(9):
            subset = tuple(h for h in hours if rng.random() < .5)
            terms[subset] += float(rng.integers(-9, 10))
        constant = float(rng.integers(-8, 9))
        values = [constant + sum(v * all(dict(zip(hours, bits))[h] for h in key) for key, v in terms.items())
                  for bits in product([0, 1], repeat=5)]
        opt = solve_bounds(constant, terms)
        errors.extend([abs(min(values)-opt['lower']), abs(max(values)-opt['upper'])])
        for lam in [0., .5, 1., 2., 4.]:
            assert abs(min(7+lam*v for v in values) - (7+lam*opt['lower'])) < 1e-7
            assert abs(max(7+lam*v for v in values) - (7+lam*opt['upper'])) < 1e-7
    raw = np.array([[0.], [np.nan], [np.nan], [11.], [9.]], dtype=np.float64)
    t = dict(penalty=120., choice=1, start=0, end=2, node=0, status='unknown', operation_cost=12.)
    a = dict(tasks=[t], metrics=dict(booking_cost=5.))
    b = dict(tasks=[dict(t, penalty=100.)], metrics=dict(booking_cost=5.))
    case = dict(issue=0)
    # Same physical completion applies across arms and any repeated seeds.
    ca, ta = expression(a, case, raw, 'penalty'); cb, tb = expression(b, case, raw, 'penalty')
    terms = {key: ta[key] - tb[key] for key in ta}
    opt = solve_bounds(ca-cb, terms)
    values = [direct(a, case, raw, dict(zip([(1,0),(2,0)], bits)))['penalty']-
              direct(b, case, raw, dict(zip([(1,0),(2,0)], bits)))['penalty'] for bits in product([0,1], repeat=2)]
    assert (opt['lower'], opt['upper']) == (min(values), max(values)) == (0., 20.)
    attempted=dict(tasks=[t],metrics=dict(booking_cost=5.))
    unattempted=dict(tasks=[dict(t,choice=None,start=None,end=None,status='unattempted',operation_cost=0.)],metrics=dict(booking_cost=5.))
    cross_seed_pairs=[(attempted,unattempted),(unattempted,attempted),(attempted,attempted)]
    seed_constant=0.;seed_terms=defaultdict(float)
    for left,right in cross_seed_pairs:
        for sign,e in [(1,left),(-1,right)]:
            c,tt=expression(e,case,raw,'penalty');seed_constant+=sign*c/3
            for key,v in tt.items():seed_terms[key]+=sign*v/3
    cancelled=solve_bounds(seed_constant,seed_terms)
    assert cancelled['lower']==cancelled['upper']==0
    for bits in product([0,1],repeat=2):
        w=dict(zip([(1,0),(2,0)],bits))
        assert sum(direct(left,case,raw,w)['penalty']-direct(right,case,raw,w)['penalty'] for left,right in cross_seed_pairs)==0
    # Known exceedance dominates a missing hour, threshold equality counts as good.
    assert window_term(raw, 0, 0, 3, 0) is False
    assert window_term(np.array([[0.],[10.]], dtype=np.float64), 0, 0, 1, 0) is True
    p = affine_direction(-2., 1., 3.)
    assert p['positive_roots'] == [2./3, 2.]
    assert not affine_direction(2., 1., 3.)['positive_roots']
    assert max(errors) < 1e-7
    return dict(status='PASS', exhaustive_random_problems=35, completions_each=32,
                paired_task_ledger_completions=4, lambdas=[0,.5,1,2,4], max_error=max(errors),
                cross_seed_cancellation_cases=3, same_hour_across_arms_seeds=True, negative_lambda_excluded=True)
