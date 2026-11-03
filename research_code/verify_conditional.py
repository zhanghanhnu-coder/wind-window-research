import hashlib,json,re,subprocess,sys,shutil
import numpy as np
import pandas as pd
import joblib
from research_conditional import ROOT,ART,OUT,RESULT,PAIRS,load,identity,masks
from conditional_core import design,changes,error
from event_graph import KINDS,state_at

def run():
 tested=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'风电运维Demo/tests'),'-p','test_*.py'],capture_output=True,text=True);text=tested.stdout+tested.stderr;(OUT/'unit_tests.txt').write_text(text,encoding='utf-8')
 if tested.returncode:raise RuntimeError(text)
 testcount=int(re.search(r'Ran (\d+) tests?',text).group(1));previous=json.loads((OUT/'previous_hashes.json').read_text(encoding='utf-8'));assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in previous.items())
 fine,times,nodes=load();sensor=np.load(OUT/'sensor_data.npz');token=identity();metrics=pd.read_csv(OUT/'prediction_metrics.csv',dtype={'source':str,'target':str});counts=pd.read_csv(OUT/'sample_counts.csv',dtype={'source':str,'target':str});tuning=pd.read_csv(OUT/'validation_tuning.csv',dtype={'source':str,'target':str});modelcount=0;recomputed=0
 events=json.loads((ART/'event_propagation_v1/events.json').read_text(encoding='utf-8'));states=np.array([KINDS.index(state_at(events,i+1)) for i in range(len(times))],np.int8)
 for source,target in PAIRS:
  si=nodes.index(source);ti=nodes.index(target)
  for signal in ['direction','wind']:
   z=design(fine['wind'].astype(float),fine['direction'].astype(float),si,ti,signal);xx=z['features'];yy=z['target'];cur=z['current'];valid=np.isfinite(xx).all(1)&np.isfinite(yy)
   for collection in ['main','sensor']:
    path=OUT/f'{collection}_{signal}_{source}_{target}_predictions.npz'
    if not path.exists():continue
    x=xx.copy();v=valid.copy()
    if collection=='sensor':
     k=list(sensor['nodes']).index(target);extra=np.column_stack([changes(sensor['heading'][:,k],True),changes(sensor['relative'][:,k],True)]);x=np.column_stack([x,extra]);v &= np.isfinite(extra).all(1)
    split=masks(times,v);s=np.load(path);assert str(s['identity'])==token;ids=s['indices'];assert np.array_equal(ids,np.flatnonzero(split['oct']|split['test']));np.testing.assert_array_equal(s['truth'],yy[ids]);np.testing.assert_array_equal(s['state'],states[ids])
    assert (times[np.flatnonzero(split['train'])]+pd.Timedelta(minutes=5)<pd.Timestamp('2024-09-01',tz=times.tz)).all()
    selected=np.unique(np.r_[0,np.arange(0,len(ids),997),len(ids)-1]);models=[name[5:] for name in s.files if name.startswith('pred_')]
    for name in models:
     e=error(s['pred_'+name],s['truth'],signal,s['current']);np.testing.assert_allclose(e,s['error_'+name],rtol=0,atol=1e-12);recomputed+=1
     if name!='persistence':
      ck=joblib.load(OUT/f'{collection}_{signal}_{source}_{target}_{name}.joblib');assert ck['identity']==token;cols=ck['columns'];pred=ck['model'].predict(x[ids[selected]][:,cols]);np.testing.assert_allclose(pred,s['pred_'+name][selected],rtol=1e-9,atol=1e-9)
      np.testing.assert_allclose(ck['model'][0].mean_,x[split['train']][:,cols].mean(0),rtol=1e-9,atol=1e-9)
      ts=tuning[(tuning.source==source)&(tuning.target==target)&(tuning.signal==signal)&(tuning.collection==collection)&(tuning.model==name)];assert ck['alpha']==ts.loc[ts.validation_MAE.idxmin(),'alpha'];modelcount+=1
     rows=metrics[(metrics.source==source)&(metrics.target==target)&(metrics.signal==signal)&(metrics.collection==collection)&(metrics.model==name)]
     for row in rows.itertuples():
      if row.scope=='oct':m=times[ids].month==10
      elif row.scope=='test_all':m=times[ids].month>=11
      elif row.scope in ['nov','dec']:m=times[ids].month=={'nov':11,'dec':12}[row.scope]
      else:m=(times[ids].month>=11)&(states[ids]==KINDS.index(row.scope[5:]))
      assert row.n==m.sum()
      if m.any():assert np.isclose(row.MAE,e[m].mean(),rtol=0,atol=1e-10)
 matched=pd.read_csv(OUT/'smoothing_sensitivity_matched.csv');assert matched.valid.equals(matched.matched_timestamps>=36)
 result={'unit_tests_passed':testcount,'selected_prediction_models_verified':modelcount,'prediction_error_series_recomputed':recomputed,'residual_models':len(list(OUT.glob('residual_*.joblib'))),'prediction_case_groups':len(list(OUT.glob('*_predictions.npz'))),'previous_artifacts_unchanged':len(previous),'matched_sensitivity_event_pairs':int(matched.valid.sum()),'experiment_identity':token,'train_only_scaler_and_september_selection_checked':True,'scope':'next 5min record; conditional linear predictive information; exploratory; not clock synchronization proof or physical causation'}
 (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 for p in OUT.iterdir():
  if p.suffix in ['.csv','.json','.txt','.md'] and p.name not in ['previous_hashes.json','manifest.json','smoothing_sensitivity.csv']:shutil.copy2(p,RESULT/p.name)
 files=[p for p in OUT.iterdir() if p.suffix in ['.csv','.npz','.joblib','.json','.txt','.md'] and p.name!='manifest.json']
 files += [ROOT/'风电运维Demo'/n for n in ['conditional_core.py','research_conditional.py','sensitivity_conditional.py','verify_conditional.py','tests/test_conditional.py','tests/test_matched_sensitivity.py']]
 files += [ROOT/'研究笔记/测点与条件预测研究方案_2026-11-01.md']
 pdf=ROOT/'10_测点核查与条件预测_2026-11-01.pdf'
 if pdf.exists():files.append(pdf)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(OUT/'manifest.json',RESULT/'manifest.json')
 print(json.dumps(result,ensure_ascii=False));print(text)

if __name__=='__main__':run()
