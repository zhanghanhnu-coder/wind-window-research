"""Post-review benchmark: frozen weights, float64 truth, one empirical-mask stress.

Writes exclusively artifacts/review_followup_v1/baselines; no training or model selection.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
import torch
from precision_truth_core import load_truth,window_labels
from crps_distribution_core import AtomNet
from multiscale_context_core import MultiScaleAtomNet
from lead_calibration_core import fit_groups,apply_groups
from seasonal_probability_core import FOLDS
from missing_window_bounds_core import physical_mapping
from review_baselines_core import training_masks,masked_histories,coarse_inputs,window_probabilities,loss_event_totals,OFFSETS,GROUPS

BASE=Path(__file__).resolve().parent
ART=BASE/'artifacts';OUT=ART/'review_followup_v1/baselines'
OLD={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'}
FAMILIES=[*OLD,'context3_graph','context7_graph','context7_own','context7_mean','climate28','analog112']
NEURAL=FAMILIES[:7];SEEDS=[20241101,20241102,20241103]
PAIRS=[('context1_graph','context1_mean'),('context1_graph','context1_own'),('context1_graph','climate28'),('context1_graph','analog112'),('context7_graph','context7_mean'),('context7_graph','context7_own'),('context7_graph','climate28'),('context7_graph','analog112'),('context7_graph','context1_graph'),('context3_graph','context1_graph')]
METHODS=['legacy_frozen32','refit64']
MASK_SEED=20261104;BOOT_SEED=20261103
TRACK={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
    p=Path(p);TRACK[str(p)]=sha(p);return p
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def source(fold,family,seed):
    producer=seed if family in NEURAL else 0
    return ART/'seasonal_probability_v1'/fold/f'predictions_{OLD[family]}_{producer}_budget40.npz' if family in OLD else ART/'multiscale_context_v1'/fold/f'predictions_{family}_{producer}.npz'
def legacy_theta(fold,family,seed):
    producer=seed if family in NEURAL else 0
    p=ART/'seasonal_probability_v1'/fold/f'calibration_{OLD[family]}_{producer}.npz' if family in OLD else ART/'multiscale_context_v1'/fold/f'calibration_{family}_{producer}.npz'
    return np.load(track(p))['theta']
def checkpoint(fold,family,seed):
    name=OLD[family] if family in OLD else family
    parent='seasonal_probability_v1' if family in OLD else 'multiscale_context_v1'
    return ART/parent/fold/f'{name}_{seed}_budget40.pt'
def infer(ck,family,hist,batch=32):
    span=1 if family in OLD else (3 if family=='context3_graph' else 7)
    net=AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True) if span==1 else MultiScaleAtomNet(ck['adj'],ck['scale'],span)
    net.load_state_dict(ck['state']);net.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(hist),batch):
            fine,coarse=coarse_inputs(hist[start:start+batch],span)
            out.append((net(torch.from_numpy(fine)) if span==1 else net(torch.from_numpy(fine),torch.from_numpy(coarse))).numpy())
    return np.concatenate(out)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    recipe=dict(families=FAMILIES,pairs=PAIRS,primary='ALL natural/refit64 single six-hour Brier context1_graph minus context1_mean',secondary='own and simple probability controls; context3/7 are sensitivities, never a winner search',methods=METHODS,offsets=OFFSETS,threshold=10,bootstrap_seed=BOOT_SEED,mask_seed=MASK_SEED,replicates=2000,mask='one training-origin 168-hour joint validity template per test origin, same template across neural models, AND current validity; no future inputs; no label masking',calibration='four preceding-month CRPS affine groups with fixed smooth interpolation; legacy float32-target fit versus new float64-target refit; stress uses natural-input fit',scope='same-year development evidence; fixed existing weights trained/selected on legacy float32 labels; no new independent test',statistical_stress='not evaluated; statistical controls only compared under natural input',code_hashes={str(BASE/n):sha(BASE/n) for n in ['research_review_baselines.py','review_baselines_core.py','precision_truth_core.py','tests/test_review_baselines.py']})
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==recipe,'protocol changed'
    else:js(p,recipe)
    times,nodes,raw=load_truth();assert raw.dtype==np.float64 and len(nodes)==17
    track(BASE/'precision_truth_core.py')
    track(BASE.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl')
    data=np.load(track(ART/'data.npz'));oldtimes=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
    np.testing.assert_array_equal(times,oldtimes)
    np.testing.assert_array_equal(raw.astype(np.float32),data['raw'])
    return times,nodes,raw

def forecast():
    times,nodes,raw=prepare();inventory=[];audits=[];started=time.monotonic()
    for fi,(fold,cfg) in enumerate(FOLDS.items()):
        folder=OUT/fold;folder.mkdir(exist_ok=True)
        f=np.load(track(ART/'seasonal_probability_v1'/fold/'data.npz'));ids=f['test_ids'];calids=f['cal_ids'];scale=float(f['scale'])
        assert int((calids+72).max())<ids.min()
        truth=np.stack([raw[i+1:i+73] for i in ids]);caltruth=np.stack([raw[i+1:i+73] for i in calids])
        ys=np.concatenate([window_labels(raw,ids,off) for off in OFFSETS],axis=1)
        yg=np.stack([window_labels(raw,ids,off,node_indices=list(g),joint=True) for off in OFFSETS for g in GROUPS],axis=1)
        fields=dict(ids=ids,single_labels=ys,group_labels=yg)
        for kind in ['single','group']:
            keys,inverse=physical_mapping(ids,kind);fields[kind+'_keys']=keys;fields[kind+'_inverse']=inverse
        np.savez_compressed(folder/'population.npz',**fields)
        empirical,selected=training_masks(f['x'][...,2].astype(bool),f['train_ids'],ids,MASK_SEED+fi)
        hist=masked_histories(f['x'],ids,empirical)
        naturalhist=np.stack([f['x'][i-167:i+1] for i in ids])
        np.savez_compressed(folder/'mask_design.npz',template_origin_ids=selected,template_keep=empirical,ids=ids)
        audits.append(dict(fold=fold,origins=len(ids),train_template_max=int(selected.max()),first_test=int(ids.min()),natural_valid_fraction=float(naturalhist[...,2].mean()),masked_valid_fraction=float(hist[...,2].mean()),natural_last24_valid_fraction=float(naturalhist[:,-24:,...,2].mean()),masked_last24_valid_fraction=float(hist[:,-24:,...,2].mean()),natural_fully_observed_24h_origins=int((naturalhist[:,-24:,...,2]==1).all((1,2)).sum())))
        for family in FAMILIES:
            for producer in (SEEDS if family in NEURAL else [0]):
                sp=source(fold,family,producer);pack=np.load(track(sp));qraw=pack['test'];calq=pack['cal']
                assert qraw.shape==(len(ids),72,17,19) and calq.shape[:-1]==caltruth.shape
                paramfile=folder/f'calibration_{family}_{producer}.npz'
                if paramfile.exists():theta=np.load(paramfile)['refit64'];details=read(paramfile.with_suffix('.json'))
                else:
                    theta,details=fit_groups(calq,caltruth,scale)
                    np.savez_compressed(paramfile,refit64=theta,legacy_frozen32=legacy_theta(fold,family,producer));js(paramfile.with_suffix('.json'),details)
                # Always track the legacy fit source, including resumed runs.
                oldtheta=legacy_theta(fold,family,producer)
                for method,params in [('legacy_frozen32',oldtheta),('refit64',theta)]:
                    for regime in (['natural','empirical_mask'] if family in NEURAL else ['natural']):
                        qp=sp;key='test'
                        if regime=='empirical_mask':
                            qp=folder/f'masked_raw_{family}_{producer}.npz';key='q'
                            ck=torch.load(track(checkpoint(fold,family,producer)),map_location='cpu',weights_only=False)
                            if not qp.exists():
                                # Native input checkpoint equivalence on three fixed origins.
                                pos=np.array([0,len(ids)//2,len(ids)-1]);clean=infer(ck,family,naturalhist[pos])
                                np.testing.assert_allclose(clean,qraw[pos],atol=1e-5,rtol=0)
                                np.savez_compressed(qp,q=infer(ck,family,hist),ids=ids)
                        q=apply_groups(np.load(qp)[key],params,scale,True)
                        coef=(2*np.arange(1,20)-20)/361
                        crps=np.mean(abs(q-truth[...,None]),axis=-1)-np.sum(q*coef,axis=-1)
                        for seed in (SEEDS if producer==0 else [producer]):
                            rankpath=ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz';ranks=np.load(track(rankpath))['ranks']
                            single,group=window_probabilities(q,ranks)
                            dest=folder/f'probabilities_{family}_{seed}_{method}_{regime}.npz'
                            np.savez_compressed(dest,single=single,group=group,crps_sums=np.nansum(crps,(1,2)),crps_counts=np.isfinite(crps).sum((1,2)))
                            inventory.append(dict(fold=fold,family=family,seed=seed,producer_seed=producer,method=method,regime=regime,source=str(qp),source_key=key,parameters=str(paramfile),ranks=str(rankpath),probabilities=str(dest),origins=len(ids)))
                print('BENCHMARK',fold,family,producer,'elapsed',round(time.monotonic()-started,1),flush=True)
    pd.DataFrame(audits).to_csv(OUT/'mask_support.csv',index=False,encoding='utf-8-sig')
    for p,h in TRACK.items():assert sha(p)==h,p
    js(OUT/'source_identity.json',TRACK);js(OUT/'inventory.json',inventory)
    js(OUT/'forecast_complete.json',dict(status='COMPLETE',streams=len(inventory),unique_origins=880,seconds=time.monotonic()-started))

def evaluate():
    assert read(OUT/'forecast_complete.json')['status']=='COMPLETE'
    times,_,_=load_truth();inventory=read(OUT/'inventory.json');rng=np.random.default_rng(BOOT_SEED)
    design={};pop={};probs={}
    for fold in FOLDS:
        z=dict(np.load(OUT/fold/'population.npz'));pop[fold]=z
        anchor=times[z['single_keys'][:,0].min()].normalize()
        k=int(((times[z['single_keys'][:,0]].max()-anchor).total_seconds())//(7*86400))+1
        draws=rng.integers(0,k,(2000,k));design[fold]=(anchor,draws)
        np.savez_compressed(OUT/fold/'target_time_draws.npz',draws=draws,anchor=str(anchor))
    for r in inventory:
        probs[r['fold'],r['family'],r['method'],r['regime'],r['seed']]=dict(np.load(r['probabilities']))
    paired=[];absolute=[];crpsrows=[]
    for scope in ['ALL',*FOLDS]:
        folds=list(FOLDS) if scope=='ALL' else [scope]
        for method in METHODS:
            for regime in ['natural','empirical_mask']:
                families=FAMILIES if regime=='natural' else NEURAL
                for kind in ['single','group']:
                    for family in families:
                        lo=hi=knownloss=0.;n=validn=unknownn=0
                        for f in folds:
                            y=pop[f][kind+'_labels'];p=np.stack([probs[f,family,method,regime,s][kind] for s in SEEDS]);a,b,c=loss_event_totals(p,None,y,pop[f][kind+'_inverse']);lo+=a.sum();hi+=b.sum();n+=int(c.sum());known=np.isfinite(y);knownloss+=np.mean((p-y)**2,axis=0)[known].sum();validn+=int(known.sum());unknownn+=int((~known).sum())
                        absolute.append(dict(scope=scope,method=method,regime=regime,kind=kind,family=family,lower=lo/n,upper=hi/n,known_only_brier=knownloss/validn,n=n,known_records=validn,unknown_records=unknownn))
                    for a,b in PAIRS:
                        if a not in families or b not in families:continue
                        pointlo=pointhi=0.;n=0;bn=np.zeros(2000);bl=np.zeros(2000);bu=np.zeros(2000);blockcounts=[]
                        for f in folds:
                            z=pop[f];y=z[kind+'_labels'];pa=np.stack([probs[f,a,method,regime,s][kind] for s in SEEDS]);pb=np.stack([probs[f,b,method,regime,s][kind] for s in SEEDS]);lo,hi,count=loss_event_totals(pa,pb,y,z[kind+'_inverse']);anchor,draws=design[f]
                            eb=((times[z[kind+'_keys'][:,0]]-anchor).total_seconds()//(7*86400)).astype(int);k=draws.shape[1]
                            ls=np.bincount(eb,weights=lo,minlength=k);us=np.bincount(eb,weights=hi,minlength=k);ns=np.bincount(eb,weights=count,minlength=k)
                            pointlo+=lo.sum();pointhi+=hi.sum();n+=int(count.sum());bl+=ls[draws].sum(1);bu+=us[draws].sum(1);bn+=ns[draws].sum(1);blockcounts.append(int((ns>0).sum()))
                        good=bn>0;supported=min(blockcounts)>=4 and good.sum()>=1900
                        paired.append(dict(scope=scope,method=method,regime=regime,kind=kind,candidate=a,baseline=b,primary=scope=='ALL' and method=='refit64' and regime=='natural' and kind=='single' and (a,b)==PAIRS[0],lower=pointlo/n,upper=pointhi/n,outer_low=float(np.quantile((bl/bn)[good],.025)) if supported else np.nan,outer_high=float(np.quantile((bu/bn)[good],.975)) if supported else np.nan,n=n,minimum_fold_blocks=min(blockcounts),valid_draws=int(good.sum()),supported=bool(supported)))
                for family in families:
                    sums=counts=0.
                    for f in folds:
                        sums+=np.mean([probs[f,family,method,regime,s]['crps_sums'] for s in SEEDS],axis=0).sum();counts+=probs[f,family,method,regime,SEEDS[0]]['crps_counts'].sum()
                    crpsrows.append(dict(scope=scope,method=method,regime=regime,family=family,crps=sums/counts,known_hours=int(counts)))
    for name,rows in [('paired_brier.csv',paired),('absolute_brier.csv',absolute),('crps.csv',crpsrows)]:pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    assert sum(r['primary'] for r in paired)==1
    js(OUT/'evaluation_complete.json',dict(status='COMPLETE',paired_rows=len(paired),primary_comparisons=1,absolute_rows=len(absolute),crps_rows=len(crpsrows)))
    print('PRIMARY',[r for r in paired if r['primary']],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['forecast','evaluate','all'],default='all');args=parser.parse_args()
    if args.phase in ['forecast','all']:forecast()
    if args.phase in ['evaluate','all']:evaluate()
if __name__=='__main__':main()
