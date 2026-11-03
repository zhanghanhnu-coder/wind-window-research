"""Independent all-case support/cost accounting, LP certificates and fixed resolves."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem
from regret_core import solve_recourse
from distributional_schedule_core import solve_distribution_master
from verify_distributional_schedule import lp,options_reference
from verify_two_stage import independent_schedule
from highs_readback_compat import install
from research_crps_distribution import guard

OUT=ART/'support_stress_v1';PARENT=ART/'distributional_schedule_v1';FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};SEEDS=[20241101,20241102,20241103];EPS={'stress_e010':.01,'stress_e001':.001,'stress_e050':.05}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-6):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def table(name,rows,keys):
 a=pd.read_csv(OUT/name,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b),(name,len(a),len(b))
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k],3e-10)
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)

def check_tables(df,times):
 metrics=[]
 for scope in ['ALL',*FOLDS]:
  selected=df if scope=='ALL' else df[df.fold==scope]
  for case,g in selected.groupby('case'):
   metrics.append(dict(scope=scope,case=case,n=len(g),unique_origins=g.issue_id.nunique(),complete=int(g.complete.sum()),complete_unique_origins=g[g.complete].issue_id.nunique(),**{k:float(g[k].mean()) for k in ['cost_lower','cost_upper','exact_cost','exact_completed','booking_cost','objective','nominal_total','no_booking']}))
 nmetrics=table('metrics.csv',metrics,['scope','case']);saved=np.load(OUT/'bootstrap_draws.npz');rng=np.random.default_rng(20261125);weights={};blocks={}
 for fold,start in FOLDS.items():
  ids=np.sort(df[df.fold==fold].issue_id.unique());blocks[fold]=np.array([(times[i]-pd.Timestamp(start,tz=times.tz)).days//7 for i in ids]);k=max(blocks[fold])+1;draw=rng.integers(0,k,(2000,k));np.testing.assert_array_equal(saved[fold],draw)
  weights[fold]=np.array([[(row==block).sum() for block in blocks[fold]] for row in draw])
 pairs=[]
 for base in ['base_tv025','base_tv000']:
  for metric in ['all_cost_bounds','exact_cost','exact_completed']:
   foldvals={}
   for fold in FOLDS:
    a=df[(df.case=='stress_e010')&(df.fold==fold)].set_index(['issue_id','seed']).sort_index();b=df[(df.case==base)&(df.fold==fold)].set_index(['issue_id','seed']).sort_index();assert len(a)==len(b)==36 and a.index.equals(b.index)
    if metric=='all_cost_bounds':d=pd.DataFrame({'lo':a.cost_lower-b.cost_upper,'hi':a.cost_upper-b.cost_lower})
    else:d=pd.DataFrame({'lo':a[metric]-b[metric],'hi':a[metric]-b[metric]})
    # Every seed must have the same truth availability; group mean cannot hide a missing seed.
    assert (d.notna().groupby(level=0).sum().isin([0,3])).all().all();v=d.groupby(level=0).mean();foldvals[fold]=v.lo.to_numpy(),v.hi.to_numpy()
   for scope in ['ALL',*FOLDS]:
    fs=list(FOLDS) if scope=='ALL' else [scope];numlo=np.zeros(2000);numhi=np.zeros(2000);den=np.zeros(2000);alllo=[];allhi=[];ks=[]
    for fold in fs:
     lo,hi=foldvals[fold];valid=np.isfinite(lo)&np.isfinite(hi);alllo.extend(lo[valid]);allhi.extend(hi[valid]);numlo+=weights[fold][:,valid]@lo[valid];numhi+=weights[fold][:,valid]@hi[valid];den+=weights[fold][:,valid].sum(1);ks.append(len(set(blocks[fold][valid])))
    ok=den>0;supported=min(ks)>=4 and ok.sum()>=1900;ql=np.quantile(numlo[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan];qu=np.quantile(numhi[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan]
    pairs.append(dict(scope=scope,candidate='stress_e010',baseline=base,metric=metric,primary=scope=='ALL' and metric=='all_cost_bounds',difference_lower=np.mean(alllo) if len(alllo) else np.nan,difference_upper=np.mean(allhi) if len(allhi) else np.nan,outer_low=ql[0],outer_high=qu[1],lower_percentile_high=ql[1],upper_percentile_low=qu[0],n=len(alllo),minimum_blocks_in_fold=min(ks),valid_draws=int(ok.sum()),resampling_supported=supported))
 npairs=table('paired_comparisons.csv',pairs,['scope','baseline','metric']);tails=[];tp=[]
 def cvar_reference(y):
  y=np.asarray(y,float);return min(float(t+np.maximum(y-t,0).mean()/.1) for t in y)
 for seed_scope,cases in [('three_seed_mean',['stress_e010','base_tv025','base_tv000']),('first_seed',['stress_e001','stress_e050','base_tv025','base_tv000'])]:
  subset=df if seed_scope=='three_seed_mean' else df[df.seed==20241101]
  for case in cases:
   g=subset[subset.case==case].groupby(['fold','issue_id'])[['cost_lower','cost_upper']].mean();assert len(g)==48
   tails.append(dict(case=case,seed_scope=seed_scope,n_origins=48,alpha=.9,cvar_lower=cvar_reference(g.cost_lower),cvar_upper=cvar_reference(g.cost_upper)))
 for case,ss in [('stress_e010','three_seed_mean'),('stress_e001','first_seed'),('stress_e050','first_seed')]:
  a=next(r for r in tails if (r['case'],r['seed_scope'])==(case,ss))
  for base in ['base_tv025','base_tv000']:
   b=next(r for r in tails if (r['case'],r['seed_scope'])==(base,ss));tp.append(dict(candidate=case,baseline=base,seed_scope=ss,difference_lower=a['cvar_lower']-b['cvar_upper'],difference_upper=a['cvar_upper']-b['cvar_lower'],scope='fixed 48 origins; descriptive tail cost; no population guarantee'))
 return dict(metric_rows=nmetrics,paired_rows=npairs,tail_rows=table('tail_risk.csv',tails,['case','seed_scope']),tail_comparison_rows=table('tail_comparisons.csv',tp,['candidate','baseline']))

def run():
 guard();install();assert read(OUT/'planning_complete.json')['status']==read(OUT/'evaluation_complete.json')['status']=='COMPLETE' and read(OUT/'failures.json')==[];parent_counts={}
 for root in [PARENT,ART/'support_diagnostic_v1']:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(file)==h,file
  parent_counts[root.name]=len(manifest)
 for file,h in read(OUT/'identity.json').items():assert sha(file)==h,file
 cfg=read(OUT/'recipe.json')['config'];p=make_problem(cfg);F=sum(t['penalty'] for t in cfg['tasks']);raw=np.load(ART/'data.npz')['raw'];times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');nodes=sorted({t['node'] for t in cfg['tasks']});parent={r['key']:r for r in read(PARENT/'inventory.json') if r['case']=='graph_rolling28'};inventory=read(OUT/'inventory.json');expected={(r['key'],case) for r in parent.values() for case in (list(EPS) if r['seed']==SEEDS[0] else ['stress_e010'])};assert len(inventory)==len(expected)==240 and {(r['parent_key'],r['case']) for r in inventory}==expected
 counts=dict(forecasts=0,plans=0,scenario_schedule_audits=0,probability_lps=0,master_certificates=0,full_dual_certificates=0,fresh_full_solves=0,fresh_recourse_solves=0,actual_recourse_solves=0);outcomes=[];planning=[];cache={};fresh_rows=[]
 for item in inventory:
  guard();old=parent[item['parent_key']];assert all(item[k]==old[k] for k in ['fold','seed','position','issue_id','timestamp']);assert item['epsilon']==EPS[item['case']] and item['source_forecast']==old['forecast_source'] and item['source_sha256']==old['forecast_sha256']==sha(old['forecast_source']);z=np.load(item['forecast_source']);original=np.load(old['forecast_source']);sc=z['scenarios'];assert sc.shape==(58,72,17);np.testing.assert_array_equal(sc[:57],original['scenarios']);np.testing.assert_array_equal(sc[57],np.full((72,17),11.));prob=np.r_[np.ones(57)*(1-item['epsilon'])/57,item['epsilon']];close(prob,z['probability'],2e-16);prob=z['probability'];available=options_reference(cfg,sc);np.testing.assert_array_equal(available,z['available']);assert not available[-1].any();assert sha(item['forecast_source'])==item['forecast_sha256'];counts['forecasts']+=1
  path=OUT/'cases'/('plan_'+item['key']+'.json');record=read(path);r=record['plan'];full=record['dual_reference'];x=r['x'];assert r['solver_status']==0 and r['radius']==.25;close(r['probability'],prob,1e-14);close(r['objective'],r['upper_bound']);close(r['upper_bound']-r['lower_bound'],r['gap']);assert 0<=r['gap']<=1e-7*(1+abs(r['objective']));close(r['objective'],full['objective'],1e-5)
  def account(sol,resources):
   values=[];book=None
   assert len(sol)==58
   for s,solution in enumerate(sol):book,cost=independent_schedule(cfg,resources,sc[s],solution);values.append(cost);counts['scenario_schedule_audits']+=1
   return book,np.array(values)
  book,costs=account(r['scenario_solutions'],x);close(costs,r['recourse_costs']);close(costs[-1],F);assert max(costs)<=F+1e-6;close(record['closed_bound']['upper_total'],book+F);close(record['closed_bound']['all_defer'],F)
  worst,_=lp(costs,prob,.25);counts['probability_lps']+=1;w=np.asarray(r['adversary']['weights']);assert w.min()>=0 and .5*abs(w-prob).sum()<=.25+1e-12;close(w.sum(),1,1e-12);close(w@costs,worst);close(book+worst,r['objective']);close(book+prob@costs,r['nominal_total'])
  fb,fc=account(full['scenario_solutions'],full['x']);assert full['solver_status']==0 and full['mip_gap']<1e-9;close(full['objective'],full['dual_bound']);du=full['dual_variables'];u=np.array(du['u']);eta=du['eta'];lam=du['lambda_value'];assert lam>=-1e-8 and (abs(u)<=lam+1e-6).all() and (eta+u>=fc-1e-6).all();fv,_=lp(fc,prob,.25);counts['probability_lps']+=1;close(eta+prob@u+2*.25*lam,fv);close(fb+fv,full['objective']);counts['full_dual_certificates']+=1
  hist=r['trace'];assert len(hist)==r['iterations']<=513;lbs=[];ubs=[]
  for i,h in enumerate(hist):
   m=h['master'];assert m['solver_status']==0 and m['mip_gap']<1e-9;close(m['objective'],m['dual_bound']);close(m['objective'],h['lower_bound']);mb,mc=account(m['scenario_solutions'],m['x']);cuts=np.array(h['cuts']);assert len(cuts)==i+1 and cuts.min()>=0;close(cuts[0],prob,1e-14);close(cuts.sum(1),1,1e-12);assert (.5*abs(cuts-prob).sum(1)<=.25+1e-12).all();close(mb+max(cuts@mc),m['objective']);wv=np.array(h['worst_weights']);hv=np.array(h['recourse_costs']);assert wv.min()>=0 and .5*abs(wv-prob).sum()<=.25+1e-12;close(wv.sum(),1,1e-12);opt,_=lp(hv,prob,.25);counts['probability_lps']+=1;close(wv@hv,opt);close(mb+opt,h['candidate_upper']);lbs.append(h['lower_bound']);ubs.append(h['upper_bound']);counts['master_certificates']+=1
   if i+1<len(hist):close(hist[i+1]['cuts'],np.vstack([cuts,wv]),1e-14)
  assert (np.diff(lbs)>=-1e-6).all() and (np.diff(ubs)<=1e-6).all() and (np.array(lbs)<=r['objective']+1e-6).all() and (np.array(ubs)>=r['objective']-1e-6).all();close(lbs[-1],r['lower_bound']);close(ubs[-1],r['upper_bound'])
  positions=[v['position'] for v in parent.values() if v['fold']==item['fold'] and v['seed']==SEEDS[0]]
  if item['case']=='stress_e010' and item['seed']==SEEDS[0] and item['position'] in [min(positions),max(positions)]:
   target=OUT/'cases'/('fresh_'+item['key']+'.json')
   if target.exists():fr=read(target);assert fr['plan_sha256']==sha(path)
   else:
    fresh=solve_distribution_master(p,available,prob,.25);values=[solve_recourse(p,a,x)['recourse_cost'] for a in available];fr=dict(plan_sha256=sha(path),objective=fresh['objective'],status=fresh['solver_status'],gap=fresh['mip_gap'],costs=values);target.write_text(json.dumps(fr,indent=2),encoding='utf-8')
   assert fr['status']==0 and fr['gap']<1e-9;close(fr['objective'],r['objective'],1e-5);close(fr['costs'],costs);counts['fresh_full_solves']+=1;counts['fresh_recourse_solves']+=58;fresh_rows.append(dict(key=item['key'],objective_difference=fr['objective']-r['objective']))
  truth=raw[item['issue_id']+1:item['issue_id']+73];lo=options_reference(cfg,truth[None],True)[0];hi=options_reference(cfg,truth[None],False)[0];complete=bool(np.isfinite(truth[:,nodes]).all());ck=item['issue_id'],tuple(x)
  if ck not in cache:cache[ck]=solve_recourse(p,lo,x),solve_recourse(p,hi,x);counts['actual_recourse_solves']+=2
  a,b=cache[ck];opath=OUT/'cases'/('outcome_'+item['key']+'.json');out=read(opath);assert opath.stat().st_mtime>=(OUT/'planning_complete.json').stat().st_mtime and out['complete']==complete and out['resources']==x;_,al=independent_schedule(cfg,x,truth,out['optimistic_solution'],'optimistic');_,ah=independent_schedule(cfg,x,truth,out['pessimistic_solution'],'pessimistic');close(al,a['recourse_cost']);close(ah,b['recourse_cost']);close(out['cost_lower'],book+al);close(out['cost_upper'],book+ah);assert al<=ah+1e-6 and ah<=F+1e-6;close(out['known_fraction'],np.isfinite(truth[:,nodes]).mean(),1e-14)
  if complete:close(out['exact_cost'],book+al);close(al,ah);assert out['exact_completed']==out['optimistic_solution']['completed']
  else:assert out['exact_cost'] is None and out['exact_completed'] is None
  outcomes.append({k:v for k,v in out.items() if k not in ['resources','optimistic_solution','pessimistic_solution']}|dict(objective=r['objective'],nominal_total=r['nominal_total'],no_booking=int(sum(x)==0),outcome_source=str(opath),outcome_sha256=sha(opath)));planning.append(dict(key=item['key'],case=item['case'],fold=item['fold'],seed=item['seed'],issue_id=item['issue_id'],epsilon=item['epsilon'],objective=r['objective'],booking_cost=r['booking_cost'],nominal_total=r['nominal_total'],no_booking=int(sum(x)==0),iterations=r['iterations'],gap=r['gap'],seconds=record['seconds'],plan_source=str(path),plan_sha256=sha(path)));counts['plans']+=1;print('AUDIT',counts['plans'],item['key'],flush=True)
 counts['planning_rows']=table('planning_summary.csv',planning,['key']);counts['outcome_rows']=table('new_outcomes.csv',outcomes,['key']);old=pd.read_csv(PARENT/'outcomes.csv',float_precision='round_trip');old=old[(old.case=='graph_rolling28')&old.policy.isin(['tv000','tv025'])].copy();old['case']='base_'+old.policy;old['epsilon']=0.;combined=pd.concat([pd.DataFrame(outcomes),old],ignore_index=True);counts['combined_rows']=table('outcomes.csv',combined.to_dict('records'),['case','seed','issue_id']);counts.update(check_tables(combined,times));pd.DataFrame(fresh_rows).to_csv(OUT/'independent_resolves.csv',index=False,encoding='utf-8-sig');assert counts['plans']==240 and counts['fresh_full_solves']==8 and counts['fresh_recourse_solves']==464
 for file,h in read(OUT/'identity.json').items():assert sha(file)==h,file
 counts.update(status='PASS',parent_manifest_files=parent_counts,frozen_identities=len(read(OUT/'identity.json')));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':run()
