"""Rebuild probabilities, schedules, full/cut certificates and all tables independently."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem
from regret_core import solve_recourse
from verify_distributional_schedule import lp,options_reference
from verify_two_stage import independent_schedule
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'bounded_distribution_v1';PARENT=ART/'support_dominance_v1'
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
 for f,h in {**identity,**parent}.items():assert sha(f)==h,f
 recipe=read(OUT/'recipe.json');cfg=recipe['config'];problem=make_problem(cfg);F=sum(t['penalty'] for t in cfg['tasks']);assert F==910 and recipe['radius']==.25 and recipe['solver_time_limit']==60;inventory=read(OUT/'inventory.json');expected=[(c,r) for c in ['M07_235','M09_206'] for r in ['raw','ui_normalized']];assert [(i['case_id'],i['representation']) for i in inventory]==expected
 counts=dict(inputs=0,runs=0,successful=0,solver_failures=0,dominance_witnesses=0,original_schedule_audits=0,reference_schedule_audits=0,probability_lps=0,full_certificates=0,master_certificates=0,bounded_certificates=0,fresh_removed_recourse=0);data={};expected_jobs=[];rows=[];fresh=[]
 for index,item in enumerate(inventory):
  assert sha(item['source'])==item['source_sha256'] and sha(item['input'])==item['input_sha256'];raw=np.load(item['source']);z=np.load(item['input']);w=raw['probability'].copy()
  if item['representation']=='ui_normalized':w=w/w.sum()
  np.testing.assert_array_equal(w,z['probability']);np.testing.assert_array_equal(z['scenarios'],raw['scenarios']);a=options_reference(cfg,z['scenarios']);np.testing.assert_array_equal(a,z['available']);np.testing.assert_array_equal(a,raw['available']);close(item['max_difference_from_raw'],max(abs(w-raw['probability'])),1e-30)
  _,first,iv=np.unique(a,axis=0,return_index=True,return_inverse=True);order=np.argsort(first);reps=first[order];rank=np.empty(len(order),int);rank[order]=np.arange(len(order));inverse=rank[iv];ga=a[reps];gp=np.bincount(inverse,weights=w,minlength=len(ga));keep=list(np.flatnonzero(gp>0));witness=np.arange(len(ga));removed=[]
  for g in sorted(np.flatnonzero(gp==0),key=lambda j:(int(ga[j].sum()),int(j))):
   candidates=[k for k in keep if not np.any(ga[k]&~ga[g])]
   if candidates:witness[g]=candidates[0];removed.append(int(g))
   else:keep.append(int(g))
  for g in removed:assert gp[g]==0 and witness[g] in keep and not np.any(ga[witness[g]]&~ga[g]);counts['dominance_witnesses']+=1
  data[item['key']]=(z['scenarios'],w,reps,inverse,ga,gp,np.array(keep),np.array(sorted(removed),int),witness);counts['inputs']+=1
  for algorithm in ['ccg','full']:
   order=['original','bounded'] if (index+1+(algorithm=='full'))%2 else ['bounded','original']
   expected_jobs.extend(dict(key=item['key']+'__'+algorithm+'__'+f,case_id=item['case_id'],representation=item['representation'],algorithm=algorithm,formulation=f,input_key=item['key']) for f in order)
 assert recipe['jobs']==expected_jobs
 for ji,job in enumerate(expected_jobs):
  guard();path=OUT/'cases'/('solve_'+job['key']+'.json');rec=read(path);assert all(rec[k]==v for k,v in job.items()) and rec['order']==ji and rec['seconds']>0;r=rec.get('result',{});sc,w,reps,inv,ga,gp,keep,removed,witness=data[job['input_key']];prob=gp[keep];K=len(keep)
  if rec['status']=='SOLVER_FAILURE':assert 'result' not in rec and rec['error']['status']!=0;counts['solver_failures']+=1
  else:
   assert rec['status']=='COMPLETE' and r['certificate']['passed'] and r['scenario_count']==228 and r['optimization_scenarios']==K
   for name,value in [('class_inverse',inv),('class_probability',gp),('kept_classes',keep),('removed_classes',removed),('dominance_witness',witness),('scenario_weights',w)]:close(r[name],value,1e-14)
   costs=[]
   for si,sol in enumerate(r['scenario_solutions']):
    book,q=independent_schedule(cfg,r['x'],sc[si],sol);costs.append(book+q);assert sol['solver_status']==0 and sol['mip_gap']<=1e-9;counts['original_schedule_audits']+=1
   assert len(costs)==228;costs=np.array(costs);close(costs,r['scenario_costs']);classcost=costs[reps];close(costs,classcost[inv]);close(book,r['booking_cost']);close(w@costs,r['nominal_total']);v,_=lp(costs,w,.25);counts['probability_lps']+=1;adv=np.array(r['adversary_weights']);assert adv.min()>=0;close(adv.sum(),1,1e-12);assert .5*abs(adv-w).sum()<=.25+1e-12;close(adv@costs,v);close(v,r['objective']);groupq=np.bincount(inv,weights=adv,minlength=len(ga));close(groupq[removed],0,1e-14);inner=r['inner'];close(inner['objective'],v);close(inner['scenario_costs'],classcost[keep]);close(inner['scenario_weights'],prob,1e-14);assert inner['x']==r['x']
   for g in removed:
    assert classcost[g]<=classcost[witness[g]]+1e-6;guard();sol=solve_recourse(problem,ga[g],r['x']);close(sol['recourse_cost']+book,classcost[g]);fresh.append(dict(key=job['key'],group=int(g),x=r['x'],recourse_cost=sol['recourse_cost'],solver_status=sol['solver_status'],mip_gap=sol['mip_gap']));counts['fresh_removed_recourse']+=1
   def account(solutions,x):
    assert len(solutions)==K;values=[]
    for k,s in enumerate(solutions):b,q=independent_schedule(cfg,x,sc[reps[keep[k]]],s);values.append(q);counts['reference_schedule_audits']+=1
    return b,np.array(values)
   def boundcheck(m,b,values):
    if job['formulation']!='bounded':return
    proof=m['bounds_certificate'];assert proof['active'] and proof['all_defer_cost']==F and not proof['zero_radius_expected_form'];assert max(values)<=F+1e-6 and m['objective']<=F+1e-6
    if m['dual_variables'] is not None:
     L=F/.5;assert proof['lambda_upper']==L;d=m['dual_variables'];assert d['lambda_value']<=L+1e-6 and -L-1e-6<=d['eta']<=F+L+1e-6 and max(abs(np.array(d['u'])))<=L+1e-6
    else:assert proof['lambda_upper'] is None
    counts['bounded_certificates']+=1
   ib,ic=account(inner['scenario_solutions'],inner['x']);close(ib+ic,classcost[keep]);full=inner['full_reference'];assert full['solver_status']==0 and full['mip_gap']<=1e-9;close(full['objective'],full['dual_bound']);fb,fc=account(full['scenario_solutions'],full['x']);d=full['dual_variables'];u=np.array(d['u']);eta=d['eta'];lam=d['lambda_value'];assert lam>=-1e-8 and (abs(u)<=lam+1e-6).all() and (eta+u>=fc-1e-6).all();fv,_=lp(fc,prob,.25);counts['probability_lps']+=1;close(fv,eta+prob@u+.5*lam);close(fb+fv,full['objective']);close(full['objective'],v,1e-5);boundcheck(full,fb,fc);counts['full_certificates']+=1
   if job['algorithm']=='ccg':
    plan=inner['ccg_plan'];close(plan['objective'],v);assert plan['x']==r['x'];pb,pc=account(plan['scenario_solutions'],plan['x']);close(pb+pc,classcost[keep]);trace=plan['trace'];assert len(trace)==r['iterations'];lbs=[];ubs=[]
    for ti,h in enumerate(trace):
     m=h['master'];assert m['solver_status']==0 and m['mip_gap']<=1e-9;close(m['objective'],m['dual_bound']);mb,mc=account(m['scenario_solutions'],m['x']);cuts=np.array(h['cuts']);assert len(cuts)==ti+1 and cuts.min()>=0;close(cuts[0],prob,1e-12);close(cuts.sum(1),1,1e-12);assert (.5*abs(cuts-prob).sum(1)<=.25+1e-12).all();close(mb+max(cuts@mc),m['objective']);close(m['objective'],h['lower_bound']);hw=np.array(h['worst_weights']);assert hw.min()>=0;close(hw.sum(),1,1e-12);assert .5*abs(hw-prob).sum()<=.25+1e-12;ov,_=lp(h['recourse_costs'],prob,.25);counts['probability_lps']+=1;close(hw@np.array(h['recourse_costs']),ov);close(mb+ov,h['candidate_upper']);boundcheck(m,mb,mc);counts['master_certificates']+=1;lbs.append(h['lower_bound']);ubs.append(h['upper_bound'])
     if ti+1<len(trace):close(trace[ti+1]['cuts'],np.vstack([cuts,hw]),1e-12)
    assert (np.diff(lbs)>=-1e-6).all() and (np.diff(ubs)<=1e-6).all();close(lbs[-1],r['lower_bound']);close(ubs[-1],r['upper_bound'])
   else:assert inner['ccg_plan'] is None and r['x']==full['x'];close(r['lower_bound'],full['dual_bound']);close(r['upper_bound'],full['objective'])
   close(r['upper_bound']-r['lower_bound'],r['gap']);assert 0<=r['gap']<=1e-7*(1+abs(v));counts['successful']+=1
  rows.append(dict(**job,order=ji,status=rec['status'],seconds=rec['seconds'],objective=r.get('objective',np.nan),gap=r.get('gap',np.nan),source=str(path),source_sha256=sha(path)));counts['runs']+=1;print('AUDIT',ji,rec['status'],flush=True)
 counts['run_rows']=table('runs.csv',rows,['key']);df=pd.DataFrame(rows);comparisons=[]
 for item in inventory:
  for algorithm in ['ccg','full']:
   pair=df[(df.input_key==item['key'])&(df.algorithm==algorithm)];a=pair[pair.formulation=='original'].iloc[0];b=pair[pair.formulation=='bounded'].iloc[0];both=a.status==b.status=='COMPLETE';diff=b.objective-a.objective if both else np.nan
   if both:assert abs(diff)<=1e-5
   comparisons.append(dict(input_key=item['key'],algorithm=algorithm,original_status=a.status,bounded_status=b.status,both_complete=both,objective_difference=diff))
 counts['comparison_rows']=table('comparisons.csv',comparisons,['input_key','algorithm']);summary=[]
 for (form,rep,alg),g in df.groupby(['formulation','representation','algorithm']):summary.append(dict(formulation=form,representation=rep,algorithm=alg,runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),seconds_total=float(g.seconds.sum())))
 counts['summary_rows']=table('summary.csv',summary,['formulation','representation','algorithm']);assert counts['runs']==16 and counts['successful']+counts['solver_failures']==16
 for f,h in {**identity,**parent}.items():assert sha(f)==h,f
 (OUT/'fresh_removed_recourse.json').write_text(json.dumps(fresh,indent=2),encoding='utf-8');counts.update(status='PASS',source_files_unchanged=len(identity),parent_manifest_files=len(parent),future_observations_used=False,scope='exact bounded formulation / floating representation diagnostic; fixed selected cases, exploratory, no blanket performance guarantee');(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()
