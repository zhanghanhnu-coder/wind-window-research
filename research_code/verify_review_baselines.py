"""Independent reconstruction of post-review baseline outputs; old studies read-only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from precision_truth_core import load_truth
from crps_distribution_core import AtomNet
from multiscale_context_core import MultiScaleAtomNet,multiscale_inputs

BASE=Path(__file__).resolve().parent;OUT=BASE/'artifacts/review_followup_v1/baselines'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE'
    times,nodes,raw=load_truth();inventory=read(OUT/'inventory.json');population={};design={};saved={};prediction_checks=0
    for fold in FOLDS:
        z=dict(np.load(OUT/fold/'population.npz'));ids=z['ids'];population[fold]=z
        # Truth classification does not call the experiment window label function.
        ys=[];yg=[]
        for off in OFFSETS:
            obs=raw[ids[:,None]+np.arange(off+1,off+7)]
            failure=(np.isfinite(obs)&(obs>10)).any(1);complete=np.isfinite(obs).all(1)
            ys.append(np.where(failure,0.,np.where(complete,1.,np.nan)))
            for group in GROUPS:
                sub=obs[:,:,group];failed=(np.isfinite(sub)&(sub>10)).any((1,2));observed=np.isfinite(sub).all((1,2))
                yg.append(np.where(failed,0.,np.where(observed,1.,np.nan)))
        np.testing.assert_array_equal(z['single_labels'],np.concatenate(ys,1));np.testing.assert_array_equal(z['group_labels'],np.stack(yg,1))
        f=np.load(BASE/'artifacts/seasonal_probability_v1'/fold/'data.npz');m=np.load(OUT/fold/'mask_design.npz')
        assert np.isin(m['template_origin_ids'],f['train_ids']).all() and m['template_origin_ids'].max()<ids.min()
        for i,t in enumerate(m['template_origin_ids']):
            np.testing.assert_array_equal(m['template_keep'][i],f['x'][t-167:t+1,:,2].astype(bool))
        d=np.load(OUT/fold/'target_time_draws.npz');anchor=pd.Timestamp(str(d['anchor']));design[fold]=(anchor,d['draws'])
        assert anchor==times[z['single_keys'][:,0].min()].normalize()
    stress_checked=set();calibration_checked=set()
    for entry in inventory:
        f=entry['fold'];qraw=np.load(entry['source'])[entry['source_key']].astype(float);theta=np.load(entry['parameters'])[entry['method']]
        if entry['parameters'] not in calibration_checked:
            family=entry['family'];producer=entry['producer_seed'];isold=family.startswith('context1_')
            original_name={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'}.get(family,family)
            original=BASE/'artifacts'/('seasonal_probability_v1' if isold else 'multiscale_context_v1')/f/(f'predictions_{original_name}_{producer}_budget40.npz' if isold else f'predictions_{family}_{producer}.npz')
            calq=np.load(original)['cal'];folddata=np.load(BASE/'artifacts/seasonal_probability_v1'/f/'data.npz');calids=folddata['cal_ids'];scale=float(folddata['scale']);assert (calids+72).max()<population[f]['ids'].min()
            cy=np.stack([raw[i+1:i+73] for i in calids]);fit=np.load(entry['parameters'])['refit64'];info=read(Path(entry['parameters']).with_suffix('.json'))
            for j,(start,end) in enumerate([(0,6),(6,24),(24,48),(48,72)]):
                valid=np.isfinite(cy[:,start:end]);qq=calq[:,start:end][valid].astype(float)/scale;yy=cy[:,start:end][valid]/scale
                candidate=np.maximum(0,qq[:,9,None]+fit[j,0]+fit[j,1]*(qq-qq[:,9,None]));coef=(2*np.arange(1,20)-20)/361
                objective=np.mean(np.mean(abs(candidate-yy[:,None]),axis=-1)-np.sum(candidate*coef,axis=-1))+.001*(fit[j,0]**2+(fit[j,1]-1)**2)
                np.testing.assert_allclose(objective,info[j]['objective_used'],atol=1e-11,rtol=0);assert valid.sum()==info[j]['valid_targets']
            calibration_checked.add(entry['parameters'])
        if entry['regime']=='empirical_mask' and entry['source'] not in stress_checked:
            family=entry['family'];producer=entry['producer_seed'];isold=family.startswith('context1_')
            original_name={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'}.get(family,family)
            ck=torch.load(BASE/'artifacts'/('seasonal_probability_v1' if isold else 'multiscale_context_v1')/f/f'{original_name}_{producer}_budget40.pt',weights_only=False,map_location='cpu')
            span=1 if isold else (3 if family=='context3_graph' else 7)
            net=AtomNet(ck['adj'],ck['scale']) if isold else MultiScaleAtomNet(ck['adj'],ck['scale'],span)
            net.load_state_dict(ck['state']);net.eval();fold=np.load(BASE/'artifacts/seasonal_probability_v1'/f/'data.npz');md=np.load(OUT/f/'mask_design.npz');ids=population[f]['ids'];pos=[0,len(ids)//2,len(ids)-1]
            inputs=[]
            for i in pos:
                h=fold['x'][ids[i]-167:ids[i]+1].copy();keep=md['template_keep'][i]&h[...,2].astype(bool);h[~keep]=0
                fine,coarse=multiscale_inputs(h,np.array([167]),span);inputs.append((fine,coarse))
            with torch.no_grad():
                fine=torch.from_numpy(np.concatenate([x[0] for x in inputs]));coarse=torch.from_numpy(np.concatenate([x[1] for x in inputs]))
                check=(net(fine) if isold else net(fine,coarse)).numpy()
            np.testing.assert_allclose(check,qraw[pos],atol=1e-5,rtol=0);stress_checked.add(entry['source'])
        scale=float(np.load(BASE/'artifacts/seasonal_probability_v1'/f/'data.npz')['scale'])
        params=np.column_stack([np.interp(np.arange(1,73),[3.5,15.5,36.5,60.5],theta[:,j]) for j in range(2)])
        q=np.maximum(0,qraw[...,9,None]+params[:,0][None,:,None,None]*scale+params[:,1][None,:,None,None]*(qraw-qraw[...,9,None]))
        ranks=np.load(entry['ranks'])['ranks'];np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape))
        atom_index=ranks//3
        # Gather atom indexes directly, avoiding the experiment's repeated-atom implementation.
        safe=np.take_along_axis(q.transpose(0,3,1,2),np.broadcast_to(atom_index,(len(q),)+atom_index.shape),axis=1)<=10
        single=np.concatenate([safe[:,:,off:off+6].all(2).sum(1)/57 for off in OFFSETS],axis=1)
        group=np.stack([safe[:,:,off:off+6,g].all((2,3)).sum(1)/57 for off in OFFSETS for g in GROUPS],axis=1)
        probability=dict(np.load(entry['probabilities']));np.testing.assert_array_equal(single,probability['single']);np.testing.assert_array_equal(group,probability['group'])
        ids=population[f]['ids'];truth=np.stack([raw[i+1:i+73] for i in ids]);valid=np.isfinite(truth)
        # E|X-X'| over 19 atoms on three fixed origins, not the sorted-coefficient identity.
        for i in [0,len(ids)//2,len(ids)-1]:
            direct=np.mean(abs(q[i]-truth[i,...,None]),axis=-1)-np.mean(abs(q[i,...,None,:]-q[i,...,:,None]),axis=(-1,-2))/2
            np.testing.assert_allclose(np.nansum(direct),probability['crps_sums'][i],atol=1e-9,rtol=0)
        np.testing.assert_array_equal(valid.sum((1,2)),probability['crps_counts'])
        saved[f,entry['family'],entry['method'],entry['regime'],entry['seed']]=probability
        prediction_checks+=1
        if prediction_checks%48==0:print('VERIFY FLOWS',prediction_checks,flush=True)
    pairs=pd.read_csv(OUT/'paired_brier.csv');checked=0
    for row in pairs.itertuples():
        folds=FOLDS if row.scope=='ALL' else [row.scope];pl=pu=0.;n=0;bl=np.zeros(2000);bu=np.zeros(2000);bn=np.zeros(2000)
        for f in folds:
            z=population[f];kind=row.kind;ix=z[kind+'_inverse'];y=z[kind+'_labels'];count=np.bincount(ix.ravel());pa=np.stack([saved[f,row.candidate,row.method,row.regime,s][kind] for s in SEEDS]);pb=np.stack([saved[f,row.baseline,row.method,row.regime,s][kind] for s in SEEDS])
            # Evaluate both binary outcomes directly, then share one outcome per physical event.
            loss0=np.mean(pa**2-pb**2,axis=0);loss1=np.mean((1-pa)**2-(1-pb)**2,axis=0)
            e0=np.bincount(ix.ravel(),weights=loss0.ravel());e1=np.bincount(ix.ravel(),weights=loss1.ravel())
            label=np.full(len(e0),np.nan);known=np.isfinite(y);label[ix[known]]=y[known]
            lo=np.where(label==0,e0,np.where(label==1,e1,np.minimum(e0,e1)));hi=np.where(label==0,e0,np.where(label==1,e1,np.maximum(e0,e1)))
            anchor,draw=design[f];block=((times[z[kind+'_keys'][:,0]]-anchor).total_seconds()//(7*86400)).astype(int);k=draw.shape[1]
            lo_b=np.bincount(block,weights=lo,minlength=k);hi_b=np.bincount(block,weights=hi,minlength=k);n_b=np.bincount(block,weights=count,minlength=k)
            pl+=lo.sum();pu+=hi.sum();n+=count.sum();bl+=lo_b[draw].sum(1);bu+=hi_b[draw].sum(1);bn+=n_b[draw].sum(1)
        np.testing.assert_allclose([pl/n,pu/n,*np.quantile(bl/bn,[.025]),*np.quantile(bu/bn,[.975])],[row.lower,row.upper,row.outer_low,row.outer_high],atol=1e-12,rtol=0)
        assert row.n==n and row.valid_draws==(bn>0).sum();checked+=1
    identity=read(OUT/'source_identity.json');frozen={}
    for folder in ['seasonal_probability_v1','multiscale_context_v1','partial_copula_v1']:
        frozen.update(read(BASE/'artifacts'/folder/'final_manifest.json'))
    frozen_checked=0
    for name in ['model.py','direct_window_core.py','crps_distribution_core.py','lead_calibration_core.py','multiscale_context_core.py','seasonal_probability_core.py','missing_window_bounds_core.py']:
        identity[str(BASE/name)]=sha(BASE/name)
    for p,h in identity.items():
        assert sha(p)==h,p
        if p in frozen:assert frozen[p]==h,p;frozen_checked+=1
    assert frozen_checked>100
    result=dict(status='PASS',probability_streams=prediction_checks,float64_calibration_groups_objective_checked=len(calibration_checked)*4,masked_raw_streams_reinferred=len(stress_checked),paired_rows=checked,truth_population_folds=4,raw_truth_dtype=str(raw.dtype),direct_CRPS_origins=prediction_checks*3,source_files_unchanged=len(identity),sources_matching_previous_frozen_manifest=frozen_checked,independent_paths=['direct binary-outcome loss enumeration','direct atom-index scenario reconstruction','pairwise CRPS identity','original float64 raw classification','stress re-inference using the original multiscale input path','calibration objective and strict calendar maturity reconstruction'])
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'}
    manifest.update({str(BASE/n):sha(BASE/n) for n in ['research_review_baselines.py','review_baselines_core.py','verify_review_baselines.py','summarize_review_baselines.py','tests/test_review_baselines.py','precision_truth_core.py']})
    manifest.update(identity)
    (OUT/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result,flush=True)
if __name__=='__main__':verify()
