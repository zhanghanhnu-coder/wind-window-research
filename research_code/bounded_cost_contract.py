"""Public entry for the bounded formulation with an explicit cost-scale contract."""
import bounded_distribution_core as _core
import math
MIN_POSITIVE_COST=1e-4
def validate_cost_scale(problem):
 values=[t['penalty'] for t in problem.config['tasks']]
 for key in ['boat_booking','boat_hourly','crew_booking','crew_hourly','spare_cost']:values.extend(problem.config[key])
 if any(not math.isfinite(v) or v<0 or 0<v<MIN_POSITIVE_COST for v in values):
  raise ValueError('费用尺度超出当前有界求解支持范围：金额须为0或至少0.0001；这不表示业务不可行，请统一换算费用单位后重新验证')
 return dict(min_positive_cost=MIN_POSITIVE_COST,zero_allowed=True,scope='Numerical input restriction only; no blanket solver-stability guarantee.')
def solve_distribution_master(problem,*args,**kwargs):
 validate_cost_scale(problem);return _core.solve_distribution_master(problem,*args,**kwargs)
def solve_bounded_dominance(problem,*args,**kwargs):
 validate_cost_scale(problem);return _core.solve_bounded_dominance(problem,*args,**kwargs)
