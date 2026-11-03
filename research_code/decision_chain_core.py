"""Round20 forecast-to-decision bridge. No future observations in planning."""
import json
from pathlib import Path
import numpy as np
from model import ART
from probability_core import break_dependence
from crps_distribution_core import couple_atoms
from lead_calibration_core import apply_groups
from two_stage_core import weather_options,booking_cost,SolverFailure
from regret_core import solve_master,solve_recourse,solve_ccg,perfect_information
FAMILIES=['old57','gated_graph57','gated_temporal57','old57_time_broken','old57_space_broken']
POLICIES=['deterministic','stochastic','regret']

def checked_scenarios(sc):
 x=np.asarray(sc)
 if x.ndim!=3 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('finite nonempty scenario/hour/node array required')
 return x

def scenario_mean(sc):return checked_scenarios(sc).mean(axis=0,dtype=np.float64)

def dependency_variant(sc,kind):return break_dependence(checked_scenarios(sc),kind)

def sorted_rows(a):return a[np.lexsort(a.T[::-1])]

def audit_dependency(original,changed,kind):
 a=checked_scenarios(original);b=checked_scenarios(changed);np.testing.assert_array_equal(np.sort(a,axis=0),np.sort(b,axis=0))
 if kind=='time':
  for h in range(a.shape[1]):np.testing.assert_array_equal(sorted_rows(a[:,h,:]),sorted_rows(b[:,h,:]))
 elif kind=='space':
  for n in range(a.shape[2]):np.testing.assert_array_equal(sorted_rows(a[:,:,n]),sorted_rows(b[:,:,n]))
 else:raise ValueError('unknown dependence control')
 return True

class ScenarioBank:
 def __init__(self,root=ART):
  root=Path(root);self.root=root;self.meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'));src=root/'crps_distribution_v1'
  with np.load(src/'evaluation_data.npz') as e:self.complete=e['complete_positions'];self.selected=e['selected57'];self.test_ids=e['test_ids'];self.source_ids=e['selected57_ids']
  assert len(self.selected)==57 and max(self.source_ids+72)<min(self.test_ids)
  self.old=np.load(root/'probability_v1/predictions.npz');self.cache={}
 def get(self,family,seed,position):
  if family not in FAMILIES:raise ValueError('unknown family')
  if seed not in [20241101,20241102,20241103] or not isinstance(position,(int,np.integer)) or not 0<=position<len(self.test_ids):raise ValueError('invalid frozen forecast key')
  base='old57' if family.startswith('old57') else family;key=(base,seed)
  if key not in self.cache:
   if base=='old57':self.cache[key]=(self.old[f'test_q_{seed}'][...,1],(self.old['cal_truth']-self.old[f'cal_q_{seed}'][...,1])[self.complete][self.selected])
   else:
    model='gated_graph_atoms' if base=='gated_graph57' else 'gated_temporal_atoms';src=self.root/'crps_distribution_v1'
    with np.load(src/f'predictions_{model}_{seed}_budget40.npz') as p,np.load(src/f'bank_{model}_{seed}.npz') as b,np.load(self.root/'crps_lead_calibration_v1'/f'parameters_{model}_{seed}.npz') as t:self.cache[key]=(p['test'],b['ranks'],t['affine'])
  data=self.cache[key]
  sc=np.maximum(0,data[0][position]+data[1]) if base=='old57' else couple_atoms(apply_groups(data[0][position],data[2],self.meta['scale'],True),data[1])
  if family.endswith('_broken'):
   kind=family.split('_')[1];z=dependency_variant(sc,kind);audit_dependency(sc,z,kind);sc=z
  assert sc.shape==(57,72,17)
  return sc.copy()

def plan_scenarios(p,sc,guard=None,progress=None,partial=None):
 sc=checked_scenarios(sc);a=weather_options(p,sc);check=guard or (lambda:None);result={}
 def save():
  if partial:partial(result)
 check();benchmark=perfect_information(p,a,guard=guard);v=np.asarray(benchmark['values']);result['benchmark']=benchmark;save();plans={};result['plans']=plans
 check();plans['deterministic']=solve_master(p,weather_options(p,scenario_mean(sc)[None]),'stochastic');save()
 check();plans['stochastic']=solve_master(p,a,'stochastic');save()
 check();plans['regret']=solve_ccg(p,a,offsets=v,guard=guard,progress=progress);save()
 check();full=solve_master(p,a,offsets=v);result['full_regret']=full;save()
 if abs(full['objective']-plans['regret']['objective'])>1e-6:raise SolverFailure(4,'full regret and CCG disagree')
 for policy,plan in plans.items():
  solutions=[]
  for allowed in a:check();solutions.append(solve_recourse(p,allowed,plan['x']))
  total=booking_cost(p,plan['x'])+np.array([s['recourse_cost'] for s in solutions]);regrets=total-v
  if regrets.min()<-1e-6:raise SolverFailure(4,'cost below perfect-information optimum')
  plan.update(evaluated_scenario_solutions=solutions,scenario_costs=total.tolist(),scenario_regrets=regrets.tolist(),forecast_expected_cost=float(total.mean()),forecast_worst_cost=float(total.max()),forecast_mean_regret=float(regrets.mean()),forecast_worst_regret=float(regrets.max()));save()
 assert abs(plans['regret']['objective']-plans['regret']['forecast_worst_regret'])<1e-6
 assert all(plans['regret']['forecast_worst_regret']<=x['forecast_worst_regret']+1e-6 for x in plans.values())
 assert all(plans['stochastic']['forecast_expected_cost']<=x['forecast_expected_cost']+1e-6 for x in plans.values())
 result['no_task_window_fraction']=float((~a.any(1)).mean());return result
