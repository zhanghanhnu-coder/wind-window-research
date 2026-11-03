"""Original-support schedule/LP audit of exact feasibility-equivalence runs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from v5.store import StudyStore
from verify_distributional_schedule import lp,options_reference
from verify_two_stage import independent_schedule
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'scenario_equivalence_v1';PARENT=ART/'v5_release_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-6):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def check_table(name,rows,keys):
 a=pd.read_csv(OUT/name,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for col in b:
  if b[col].dtype.kind in 'biufc':close(a[col],b[col],3e-10)
  else:assert a[col].fillna('').tolist()==b[col].fillna('').tolist(),(name,col)
 return len(a)
def run():
 guard();install();assert read(OUT/'complete.json')['status']=='COMPLETE';identity=read(OUT/'identity.json');parent=read(PARENT/'final_manifest.json')
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in parent.items():assert sha(file)==h,file
 recipe=read(OUT/'recipe.json');cfg=recipe['config'];inventory=read(OUT/'inventory.json');assert len(inventory)==8 and len(recipe['jobs'])==32;store=StudyStore();store.observations=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('future observations forbidden'));meta=store.metadata();selected=[]
 for fold in ['M05','M07','M09','M11']:
  cases=[r for r in meta['cases'] if r['fold']==fold];selected.extend([cases[0]['id'],cases[-1]['id']])
 assert [r['case_id'] for r in inventory]==selected;counts=dict(forecasts=0,runs=0,successful=0,solver_failures=0,original_schedule_audits=0,reference_schedule_audits=0,independent_probability_lps=0,full_certificates=0,master_certificates=0,nonuniform_cases=0);rows=[];forecasts={}
 for ci,item in enumerate(inventory):
  assert item['index']==ci and item['position']==('first' if ci%2==0 else 'last') and item['request']==dict(case_id=item['case_id'],model='window_pool',seed=meta['seeds'][0],threshold=10);assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source']);sc,w,info=store.scenarios(item['request']);np.testing.assert_array_equal(sc,z['scenarios']);np.testing.assert_array_equal(w,z['probability']);available=options_reference(cfg,sc);np.testing.assert_array_equal(available,z['available']);assert len(sc)==len(w)==228
  unique=np.unique(available,axis=0);assert len(unique)==item['classes'];masses=[w[np.all(available==row,axis=1)].sum() for row in unique];assert sum(v==0 for v in masses)==item['zero_mass_classes'];nonuniform=bool(np.ptp(w)>1e-14);assert item['nonuniform']==nonuniform;close(item['expert_weights'],info['expert_weights'],1e-14);counts['forecasts']+=1;counts['nonuniform_cases']+=int(nonuniform);forecasts[item['case_id']]=(sc,w,available)
 for ji,job in enumerate(recipe['jobs']):
  guard();item=next(r for r in inventory if r['case_id']==job['case_id']);alg=job['algorithm'];comp=job['compressed'];ai=['ccg','full'].index(alg);expected_order=[False,True] if (item['index']+ai)%2==0 else [True,False];assert comp==expected_order[ji%2];key=f'{item["case_id"]}__{alg}__'+('grouped' if comp else 'raw');path=OUT/'cases'/('solve_'+key+'.json');record=read(path);assert record['key']==key and record['order']==ji and all(record[k]==job[k] for k in job) and np.isfinite(record['seconds']) and record['seconds']>0;sc,w,available=forecasts[item['case_id']];r=record.get('result',{})
  if record['status']=='SOLVER_FAILURE':assert 'result' not in record and isinstance(record['error'],dict) and 'status' in record['error'];counts['solver_failures']+=1
  else:
   assert record['status']=='COMPLETE' and r['certificate']['passed'] and r['compressed']==comp and r['algorithm']==alg and r['scenario_count']==228 and r['equivalence_classes']==item['classes'];inverse=np.asarray(r['class_inverse']);assert inverse.shape==(228,) and inverse.dtype.kind in 'iu' and inverse.min()==0 and inverse.max()==item['classes']-1
   for a in range(228):
    np.testing.assert_array_equal(inverse==inverse[a],np.all(available==available[a],axis=1))
   classmass=np.bincount(inverse,weights=w,minlength=item['classes']);close(classmass,r['class_probability'],1e-14);assert int((classmass==0).sum())==r['zero_mass_classes'];optinv=np.asarray(r['optimization_inverse']);G=item['classes'] if comp else 228;assert r['optimization_scenarios']==G;np.testing.assert_array_equal(optinv,inverse if comp else np.arange(228));prob=np.bincount(optinv,weights=w,minlength=G);close(prob,r['optimization_probability'],1e-14);representatives=[int(np.flatnonzero(optinv==g)[0]) for g in range(G)]
   close(r['scenario_weights'],w,1e-14);assert len(r['scenario_solutions'])==228;costs=[]
   for s,sol in enumerate(r['scenario_solutions']):book,q=independent_schedule(cfg,r['x'],sc[s],sol);costs.append(book+q);counts['original_schedule_audits']+=1
   costs=np.asarray(costs);close(costs,r['scenario_costs']);close(book,r['booking_cost']);close(w@costs,r['nominal_total']);v,oracle=lp(costs,w,.25);counts['independent_probability_lps']+=1;adv=np.asarray(r['adversary_weights']);assert adv.shape==(228,) and adv.min()>=0;close(adv.sum(),1,1e-12);tv=.5*abs(adv-w).sum();assert tv<=.25+1e-12;close(tv,r['adversary_tv'],1e-12);close(adv@costs,v);close(v,r['objective']);groupcost=costs[representatives];close(costs,groupcost[optinv]);groupadv=np.bincount(optinv,weights=adv,minlength=G);close(groupadv,r['reduced_adversary']['weights'],1e-12);close(.5*abs(groupadv-prob).sum(),tv,1e-12);gv,_=lp(groupcost,prob,.25);counts['independent_probability_lps']+=1;close(gv,v)
   def account(sol,x):
    values=[];b=None;assert len(sol)==G
    for gi,solution in enumerate(sol):b,q=independent_schedule(cfg,x,sc[representatives[gi]],solution);values.append(q);counts['reference_schedule_audits']+=1
    return b,np.asarray(values)
   full=r['full_reference'];assert full['solver_status']==0 and full['mip_gap']<=1e-9;close(full['objective'],full['dual_bound']);fb,fc=account(full['scenario_solutions'],full['x']);du=full['dual_variables'];eta=du['eta'];lam=du['lambda_value'];u=np.array(du['u']);assert lam>=-1e-8 and (abs(u)<=lam+1e-6).all() and (eta+u>=fc-1e-6).all();fv,_=lp(fc,prob,.25);counts['independent_probability_lps']+=1;close(fv,eta+prob@u+.5*lam);close(fb+fv,full['objective']);close(full['objective'],r['objective'],1e-5);counts['full_certificates']+=1
   if alg=='ccg':
    plan=r['ccg_plan'];close(plan['objective'],r['objective']);close(plan['objective'],full['objective'],1e-5);assert plan['x']==r['x'] and plan['solver_status']==0;close(plan['recourse_costs'],groupcost-book);pb,pc=account(plan['scenario_solutions'],plan['x']);close(pb,book);close(pc,groupcost-book);trace=plan['trace'];assert len(trace)==r['iterations'];lbs=[];ubs=[]
    for ti,h in enumerate(trace):
     m=h['master'];assert m['solver_status']==0 and m['mip_gap']<=1e-9;close(m['objective'],m['dual_bound']);close(m['objective'],h['lower_bound']);mb,mc=account(m['scenario_solutions'],m['x']);cuts=np.array(h['cuts']);assert len(cuts)==ti+1 and cuts.min()>=0;close(cuts[0],prob,1e-12);close(cuts.sum(1),1,1e-12);assert (.5*abs(cuts-prob).sum(1)<=.25+1e-12).all();close(mb+max(cuts@mc),m['objective']);hv=np.array(h['recourse_costs']);hw=np.array(h['worst_weights']);assert hw.min()>=0;close(hw.sum(),1,1e-12);assert .5*abs(hw-prob).sum()<=.25+1e-12;ov,_=lp(hv,prob,.25);close(hw@hv,ov);close(mb+ov,h['candidate_upper']);counts['independent_probability_lps']+=1;counts['master_certificates']+=1;lbs.append(h['lower_bound']);ubs.append(h['upper_bound'])
     if ti+1<len(trace):close(trace[ti+1]['cuts'],np.vstack([cuts,hw]),1e-12)
    assert (np.diff(lbs)>=-1e-6).all() and (np.diff(ubs)<=1e-6).all();close(lbs[-1],r['lower_bound']);close(ubs[-1],r['upper_bound'])
   else:assert r['ccg_plan'] is None and r['iterations'] is None and r['x']==full['x'];close(r['lower_bound'],full['dual_bound']);close(r['upper_bound'],full['objective'])
   close(r['upper_bound']-r['lower_bound'],r['gap']);assert r['gap']>=0 and r['gap']<=1e-7*(1+abs(r['objective']));counts['successful']+=1
  rows.append(dict(key=key,order=ji,case_id=item['case_id'],fold=item['fold'],position=item['position'],algorithm=alg,compressed=comp,nonuniform=item['nonuniform'],status=record['status'],seconds=record['seconds'],scenario_count=228,classes=item['classes'],zero_mass_classes=item['zero_mass_classes'],optimization_scenarios=item['classes'] if comp else 228,objective=r.get('objective',np.nan),booking_cost=r.get('booking_cost',np.nan),iterations=r.get('iterations'),gap=r.get('gap',np.nan),source=str(path),source_sha256=sha(path)));counts['runs']+=1;print('AUDIT',ji,key,record['status'],flush=True)
 counts['run_rows']=check_table('runs.csv',rows,['key']);df=pd.DataFrame(rows);pairs=[]
 for item in inventory:
  for alg in ['ccg','full']:
   g=df[(df.case_id==item['case_id'])&(df.algorithm==alg)];a=g[~g.compressed].iloc[0];b=g[g.compressed].iloc[0];both=a.status==b.status=='COMPLETE';difference=b.objective-a.objective if both else np.nan
   if both:assert abs(difference)<=1e-5
   pairs.append(dict(case_id=item['case_id'],fold=item['fold'],position=item['position'],algorithm=alg,nonuniform=item['nonuniform'],classes=item['classes'],raw_status=a.status,grouped_status=b.status,both_complete=both,objective_difference=difference,raw_seconds=a.seconds,grouped_seconds=b.seconds,raw_over_grouped_time=a.seconds/b.seconds if both else np.nan))
 counts['pair_rows']=check_table('paired.csv',pairs,['case_id','algorithm']);summary=[]
 for scope in ['ALL','uniform','nonuniform']:
  selected=df if scope=='ALL' else df[df.nonuniform==(scope=='nonuniform')]
  for (alg,comp),g in selected.groupby(['algorithm','compressed']):summary.append(dict(scope=scope,algorithm=alg,compressed=bool(comp),runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),seconds_total=float(g.seconds.sum()),median_seconds_including_failures=float(g.seconds.median())))
 counts['summary_rows']=check_table('summary.csv',summary,['scope','algorithm','compressed']);assert counts['runs']==32 and counts['forecasts']==8 and counts['successful']+counts['solver_failures']==32
 for file,h in identity.items():assert sha(file)==h,file
 counts.update(status='PASS',source_files_unchanged=len(identity),parent_manifest_files=len(parent),future_observations_used=False,scope='Exact-grouping arithmetic and recorded runtime audit; not a forecast or field-savings study');(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':run()
