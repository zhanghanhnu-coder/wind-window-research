"""Independent monotonicity witnesses, original schedules and LP/solver certificates."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem
from regret_core import solve_recourse
from verify_distributional_schedule import lp,options_reference
from verify_two_stage import independent_schedule
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'support_dominance_v1';PARENT=ART/'scenario_equivalence_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-6):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def table(name,rows,keys):
 a=pd.read_csv(OUT/name,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k],3e-10)
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)
def run():
 guard();install();assert read(OUT/'complete.json')['status']=='COMPLETE';identity=read(OUT/'identity.json');parent=read(PARENT/'final_manifest.json')
 for file,h in {**identity,**parent}.items():assert sha(file)==h,file
 recipe=read(OUT/'recipe.json');cfg=recipe['config'];p=make_problem(cfg);inventory=read(OUT/'inventory.json');old=read(PARENT/'inventory.json');assert len(inventory)==len(old)==8;assert recipe['jobs']==[dict(case_id=i['case_id'],algorithm=a) for i in old for a in ['ccg','full']];assert recipe['radius']==.25 and recipe['solver_time_limit']==60
 counts=dict(forecasts=0,runs=0,successful=0,solver_failures=0,dominance_witnesses=0,original_schedule_audits=0,reference_schedule_audits=0,probability_lps=0,full_certificates=0,master_certificates=0,fresh_removed_recourse=0,nonuniform_cases=0);data={};rows=[];comparisons=[];fresh=[]
 for item,prior in zip(inventory,old):
  for key,value in prior.items():assert item[key]==value,key
  assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source']);sc=z['scenarios'];w=z['probability'];a=options_reference(cfg,sc);np.testing.assert_array_equal(a,z['available']);assert len(a)==228
  # Independent stable representatives and partial-order witnesses, no producer helper.
  _,first,inv_sorted=np.unique(a,axis=0,return_index=True,return_inverse=True);order=np.argsort(first);representatives=first[order];rank=np.empty(len(order),int);rank[order]=np.arange(len(order));inverse=rank[inv_sorted];ga=a[representatives];gp=np.bincount(inverse,weights=w,minlength=len(ga));keep=list(np.flatnonzero(gp>0));witness=np.arange(len(ga));removed=[]
  for g in sorted(np.flatnonzero(gp==0),key=lambda j:(int(np.count_nonzero(ga[j])),int(j))):
   hard=[k for k in keep if not np.any(ga[k]&~ga[g])]
   if hard:witness[g]=hard[0];removed.append(int(g))
   else:keep.append(int(g))
  removed=sorted(removed);np.testing.assert_array_equal(item['kept_classes'],keep);np.testing.assert_array_equal(item['removed_classes'],removed);np.testing.assert_array_equal(item['dominance_witness'],witness);assert item['optimization_scenarios']==len(keep) and item['kept_zero_mass_classes']==int((gp[keep]==0).sum())
  for g in removed:assert gp[g]==0 and witness[g] in keep and not np.any(ga[witness[g]]&~ga[g]);counts['dominance_witnesses']+=1
  assert set(keep)|set(removed)==set(range(len(ga))) and not set(keep)&set(removed);close(gp[keep].sum(),1,1e-12);counts['forecasts']+=1;counts['nonuniform_cases']+=int(np.ptp(w)>1e-14);data[item['case_id']]=(sc,w,a,representatives,inverse,ga,gp,np.array(keep),np.array(removed,int),witness)
 for ji,job in enumerate(recipe['jobs']):
  guard();item=inventory[ji//2];key=job['case_id']+'__'+job['algorithm'];path=OUT/'cases'/('solve_'+key+'.json');record=read(path);assert record['key']==key and record['order']==ji and all(record[k]==job[k] for k in job);assert np.isfinite(record['seconds']) and record['seconds']>0;r=record.get('result',{});sc,w,a,reps,inv,ga,gp,keep,removed,witness=data[job['case_id']];K=len(keep);prob=gp[keep]
  if record['status']=='SOLVER_FAILURE':assert 'result' not in record and isinstance(record['error'],dict) and 'status' in record['error'];counts['solver_failures']+=1
  else:
   assert record['status']=='COMPLETE' and r['algorithm']==job['algorithm'] and r['certificate']['passed'];assert r['scenario_count']==228 and r['optimization_scenarios']==K and r['equivalence_classes']==len(ga)
   for keyname,value in [('class_inverse',inv),('class_probability',gp),('kept_classes',keep),('removed_classes',removed),('dominance_witness',witness),('optimization_probability',prob),('scenario_weights',w)]:close(r[keyname],value,1e-14)
   costs=[];assert len(r['scenario_solutions'])==228
   for si,sol in enumerate(r['scenario_solutions']):book,q=independent_schedule(cfg,r['x'],sc[si],sol);costs.append(book+q);counts['original_schedule_audits']+=1;assert sol['solver_status']==0 and sol['mip_gap']<=1e-9
   costs=np.array(costs);classcost=costs[reps];close(costs,classcost[inv]);close(costs,r['scenario_costs']);close(book,r['booking_cost']);close(w@costs,r['nominal_total']);v,_=lp(costs,w,.25);counts['probability_lps']+=1;adv=np.array(r['adversary_weights']);assert adv.shape==(228,) and adv.min()>=0;close(adv.sum(),1,1e-12);tv=.5*abs(adv-w).sum();assert tv<=.25+1e-12;close(tv,r['adversary_tv'],1e-12);close(adv@costs,v);close(v,r['objective'])
   groupq=np.bincount(inv,weights=adv,minlength=len(ga));close(groupq[removed],0,1e-14);iv,_=lp(classcost[keep],prob,.25);counts['probability_lps']+=1;close(iv,v);inner=r['inner'];assert inner['scenario_count']==K and inner['optimization_scenarios']==K and inner['compressed'] is False;close(inner['scenario_costs'],classcost[keep]);close(inner['scenario_weights'],prob,1e-14);close(inner['objective'],v);close(inner['adversary_weights'],groupq[keep],1e-12);assert inner['x']==r['x']
   for g in removed:
    assert classcost[g]<=classcost[witness[g]]+1e-6;guard();freshsol=solve_recourse(p,ga[g],r['x']);close(freshsol['recourse_cost']+book,classcost[g]);fresh.append(dict(key=key,group=int(g),x=r['x'],recourse_cost=freshsol['recourse_cost'],solver_status=freshsol['solver_status'],mip_gap=freshsol['mip_gap']));counts['fresh_removed_recourse']+=1
   def account(solutions,x):
    assert len(solutions)==K;vals=[];b=None
    for k,solution in enumerate(solutions):b,q=independent_schedule(cfg,x,sc[reps[keep[k]]],solution);vals.append(q);counts['reference_schedule_audits']+=1
    return b,np.array(vals)
   ib,ic=account(inner['scenario_solutions'],inner['x']);close(ib,book);close(ic,classcost[keep]-book)
   full=inner['full_reference'];assert full['solver_status']==0 and full['mip_gap']<=1e-9;close(full['objective'],full['dual_bound']);fb,fc=account(full['scenario_solutions'],full['x']);du=full['dual_variables'];u=np.array(du['u']);eta=du['eta'];lam=du['lambda_value'];assert lam>=-1e-8 and (abs(u)<=lam+1e-6).all() and (eta+u>=fc-1e-6).all();fv,_=lp(fc,prob,.25);counts['probability_lps']+=1;close(fv,eta+prob@u+.5*lam);close(fb+fv,full['objective']);close(full['objective'],r['objective'],1e-5);counts['full_certificates']+=1
   if job['algorithm']=='ccg':
    plan=inner['ccg_plan'];assert plan['x']==r['x'] and plan['solver_status']==0;close(plan['objective'],v);pb,pc=account(plan['scenario_solutions'],plan['x']);close(pb+pc,classcost[keep]);trace=plan['trace'];assert len(trace)==r['iterations'];lbs=[];ubs=[]
    for ti,h in enumerate(trace):
     m=h['master'];assert m['solver_status']==0 and m['mip_gap']<=1e-9;close(m['objective'],m['dual_bound']);close(m['objective'],h['lower_bound']);mb,mc=account(m['scenario_solutions'],m['x']);cuts=np.array(h['cuts']);assert len(cuts)==ti+1 and cuts.min()>=0;close(cuts[0],prob,1e-12);close(cuts.sum(1),1,1e-12);assert (.5*abs(cuts-prob).sum(1)<=.25+1e-12).all();close(mb+max(cuts@mc),m['objective']);hv=np.array(h['recourse_costs']);hw=np.array(h['worst_weights']);assert hw.min()>=0;close(hw.sum(),1,1e-12);assert .5*abs(hw-prob).sum()<=.25+1e-12;ov,_=lp(hv,prob,.25);close(hw@hv,ov);close(mb+ov,h['candidate_upper']);counts['probability_lps']+=1;counts['master_certificates']+=1;lbs.append(h['lower_bound']);ubs.append(h['upper_bound'])
     if ti+1<len(trace):close(trace[ti+1]['cuts'],np.vstack([cuts,hw]),1e-12)
    assert (np.diff(lbs)>=-1e-6).all() and (np.diff(ubs)<=1e-6).all();close(lbs[-1],r['lower_bound']);close(ubs[-1],r['upper_bound'])
   else:assert inner['ccg_plan'] is None and r['iterations'] is None and r['x']==full['x'];close(r['lower_bound'],full['dual_bound']);close(r['upper_bound'],full['objective'])
   close(r['upper_bound']-r['lower_bound'],r['gap']);assert 0<=r['gap']<=1e-7*(1+abs(v));counts['successful']+=1
  rows.append(dict(key=key,order=ji,case_id=job['case_id'],fold=item['fold'],position=item['position'],algorithm=job['algorithm'],nonuniform=item['nonuniform'],scenario_count=228,classes=item['classes'],optimization_scenarios=K,removed=len(removed),kept_zero_mass_classes=item['kept_zero_mass_classes'],status=record['status'],seconds=record['seconds'],objective=r.get('objective',np.nan),gap=r.get('gap',np.nan),source=str(path),source_sha256=sha(path)))
  for mode in ['raw','grouped']:
   oldrec=read(PARENT/'cases'/('solve_'+key+'__'+mode+'.json'));both=record['status']==oldrec['status']=='COMPLETE';diff=r['objective']-oldrec['result']['objective'] if both else np.nan
   if both:assert abs(diff)<=1e-5
   comparisons.append(dict(key=key,case_id=job['case_id'],algorithm=job['algorithm'],reference_mode=mode,reference_status=oldrec['status'],current_status=record['status'],both_complete=both,objective_difference=diff))
  counts['runs']+=1;print('AUDIT',ji,key,record['status'],flush=True)
 counts['run_rows']=table('runs.csv',rows,['key']);counts['comparison_rows']=table('comparisons.csv',comparisons,['key','reference_mode']);df=pd.DataFrame(rows);summary=[]
 for scope in ['ALL','uniform','nonuniform']:
  part=df if scope=='ALL' else df[df.nonuniform==(scope=='nonuniform')]
  for alg,g in part.groupby('algorithm'):summary.append(dict(scope=scope,algorithm=alg,runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),median_seconds_including_failures=float(g.seconds.median()),seconds_total=float(g.seconds.sum())))
 counts['summary_rows']=table('summary.csv',summary,['scope','algorithm']);assert counts['runs']==16 and counts['forecasts']==8 and counts['successful']+counts['solver_failures']==16
 for file,h in {**identity,**parent}.items():assert sha(file)==h,file
 (OUT/'fresh_removed_recourse.json').write_text(json.dumps(fresh,indent=2),encoding='utf-8');counts.update(status='PASS',source_files_unchanged=len(identity),parent_manifest_files=len(parent),future_observations_used=False,scope='strict zero-mass dominance engineering audit; no field/forecast/savings validation');(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()
