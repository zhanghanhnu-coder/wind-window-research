"""Audit all3888 frozen source cases and all50544 visible-time projections."""
from pathlib import Path
import csv,gzip,hashlib,io,json
import numpy as np
import pandas as pd
from model import ART
from v4.history import HistoryStore,case_key,state_csv
OUT=ART/'v4_release_v1';DEST=ART.parent/'v4/data/history.json.gz'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 asset=json.loads(gzip.decompress(DEST.read_bytes()));store=HistoryStore(DEST);asset_hash=sha(DEST);counts=dict(source_files=0,cases=0,state_projections=0,export_rows=0,forecast_cells=0)
 for file,h in asset['source_files'].items():assert sha(ART/file)==h,file;counts['source_files']+=1
 cfg=asset['config'];choices=asset['choices'];frame=pd.read_csv(ART/'lookahead_execution_v1/cases.csv');lookup={(r.family,r.seed,r.position,r.policy,r.rule,r.planner):r for r in frame.itertuples()};B=len(cfg['boat_booking']);C=len(cfg['crew_booking']);cost=np.array(cfg['boat_booking']+cfg['crew_booking']+cfg['spare_cost']);assert len(asset['cases'])==3888 and len(asset['origins'])==24 and sum(o['complete'] for o in asset['origins'])==17
 for flowid,flow in asset['flows'].items():
  z=np.load(ART/f'lookahead_execution_v1/window_grids_{flowid}.npz');np.testing.assert_array_equal(np.array(flow['grids'],float),z['grids']);np.testing.assert_array_equal(flow['quality'],z['quality']);counts['forecast_cells']+=z['grids'].size
 for key,record in asset['cases'].items():
  t=read(ART/record['trajectory_source']);e=read(ART/record['evaluation_source']);r=t['result'];identity=record['identity'];expectedkey=(identity['family'],identity['seed'],identity['position'],identity['policy'],identity['rule'],identity['planner']);row=lookup[expectedkey];assert key==case_key(identity);assert record['resources']==r['resources'] and record['attempts']==r['attempts'] and record['update_indices']==t['update_indices'];assert record['evaluation']==e['evaluation'];assert record['oracle_cost_lower']==e['oracle_cost_lower'] and record['oracle_cost_upper']==e['oracle_cost_upper'];assert record['complete']==row.complete;assert e['trajectory_sha256']==sha(ART/record['trajectory_source']);np.testing.assert_allclose(record['evaluation']['booking_cost'],np.dot(cost,record['resources']),rtol=0,atol=1e-10)
  for step,original in zip(record['steps'],r['steps']):
   for field in ['now','quality_ok','selected','expected_benefit']:assert step[field]==original[field]
   assert step['planned']==original.get('planned',[])
  flow=asset['flows'][f"{identity['family']}_{identity['seed']}"];outcomes={o['choice']:o for o in e['evaluation']['outcomes']}
  for now in range(0,73,6):
   state=store.state(key,now);assert state['case_id']==key and state['identity']==identity and state['data_sha256']==asset_hash;assert 'evaluation' not in state and 'steps' not in state and 'outcomes' not in state and 'complete' not in state;selected=[i for i in r['attempts'] if choices[i]['start']<=now];assert [x['choice'] for x in state['committed']]==selected;assert len(set(choices[i]['task'] for i in selected))==len(selected);used=np.zeros(len(cfg['spare_max']),int);activeboats=[];activecrews=[];bytask={}
   for i in selected:
    o=choices[i];used[o['spare']]+=o['quantity'];bytask[o['task']]=i
    if o['end']>now:activeboats.append(o['boat']);activecrews.append(o['crew'])
   assert len(activeboats)==len(set(activeboats)) and len(activecrews)==len(set(activecrews));assert state['active_boats']==sorted(activeboats) and state['active_crews']==sorted(activecrews);assert state['remaining_spares']==(np.array(r['resources'][B+C:])-used).tolist();assert min(state['remaining_spares'])>=0;np.testing.assert_allclose(state['committed_operating_cost'],sum(choices[i]['cost'] for i in selected),rtol=0,atol=1e-10)
   for j,task in enumerate(state['tasks']):
    assert task['id']==cfg['tasks'][j]['id'] and task['turbine']==asset['nodes'][cfg['tasks'][j]['node']]
    if j not in bytask:assert task['status']==('deferred' if now==72 else 'pending') and 'attempt' not in task;continue
    i=bytask[j];o=choices[i];actual=task['attempt'];assert actual['choice']==i
    for field,v in o.items():assert actual[field]==v
    assert actual['probability_at_start']==flow['grids'][t['update_indices'][o['start']//6]][0][j]
    if o['end']<=now:assert actual['outcome']==outcomes[i] and task['status']==outcomes[i]['status']
    else:assert task['status']=='active' and 'outcome' not in actual and 'known_hours' not in actual
   if now<72:
    step=r['steps'][now//6];planned=[i for i in step.get('planned',[]) if choices[i]['start']>now];assert [x['choice'] for x in state['tentative']]==planned;assert state['quality_ok']==step['quality_ok'] and state['expected_benefit']==step['expected_benefit'];assert len(state['window_grid'])==(72-now)//6
    for offset,cell in enumerate(state['window_grid']):
     start=now+offset*6;assert cell['start']==start
     for j,task in enumerate(cfg['tasks']):assert cell['probabilities'][j]==(flow['grids'][t['update_indices'][now//6]][offset][j] if step['quality_ok'] and start+task['duration']<=72 else None)
   else:assert state['tentative']==[] and state['window_grid']==[] and state['quality_ok'] is None
   expected_clock=pd.Timestamp(t['timestamp'])+pd.Timedelta(hours=now);assert pd.Timestamp(state['current_time'])==expected_clock;json.dumps(state,allow_nan=False);counts['state_projections']+=1
   if now in [0,72]:
    exported=list(csv.DictReader(io.StringIO(state_csv(state))));assert len(exported)==6
    for line,task in zip(exported,state['tasks']):assert line['snapshot_id']==state['snapshot_id'] and line['case_id']==key and line['status']==task['status']
    counts['export_rows']+=len(exported)
  result=store.evaluate(key);assert result['evaluation']==e['evaluation'] and result['complete']==row.complete and result['case_id']==key and result['kind']=='historical_expost_evaluation';counts['cases']+=1
  if counts['cases']%500==0:print('VERIFIED',counts['cases'],counts['state_projections'],flush=True)
 assert counts['state_projections']==50544;counts.update(status='PASS',asset_sha256=sha(DEST),complete_origins=17,incomplete_origins=7);(OUT/'history_verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':main()
